"""
Stronger structural baselines (paper Table II).

Four comparators beyond the ECFP4 random forest, all under the same
drug-disjoint protocol and the same 10 subsampling seeds:

  - MACCS keys + 10 RDKit physicochemical descriptors + ECFP4, random forest
    (2048 + 167 + 10 = 2225 features)
  - gradient boosting on that same 2225-d representation, default settings
  - Tanimoto-similarity kNN over ECFP4, the classic chemoinformatics scorer
  - logistic regression on a Tanimoto kernel, the structural analogue of the
    Gaussian interaction profile kernels used in the drug-disease literature

Writes results/strong_baselines.csv and results/strong_baselines.log.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import repeated_subsample_splits

ROOT = os.environ.get("MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose")
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{ROOT}/BIBM_MAIDRS/results")
os.makedirs(RESULTS, exist_ok=True)

CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
N_REPEATS = 10
SEED0 = 42

DESCRIPTORS = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
               "NumRotatableBonds", "RingCount", "FractionCSP3",
               "HeavyAtomCount", "NumAromaticRings"]


def build_structure_features(smiles_by_uid, n_drugs):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys, Descriptors, Crippen, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")

    getters = {
        "MolWt": Descriptors.MolWt, "MolLogP": Crippen.MolLogP,
        "NumHDonors": rdMolDescriptors.CalcNumHBD,
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
        "TPSA": rdMolDescriptors.CalcTPSA,
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
        "RingCount": rdMolDescriptors.CalcNumRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "HeavyAtomCount": lambda m: m.GetNumHeavyAtoms(),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    }
    maccs = np.zeros((n_drugs, 167), np.float32)
    desc = np.zeros((n_drugs, len(DESCRIPTORS)), np.float32)
    for uid, smi in smiles_by_uid.items():
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(m)
        for b in fp.GetOnBits():
            maccs[uid, b] = 1.0
        for j, name in enumerate(DESCRIPTORS):
            try:
                desc[uid, j] = float(getters[name](m))
            except Exception:
                desc[uid, j] = 0.0
    # descriptors on wildly different scales; standardise so they do not swamp
    # the binary blocks in the distance-free tree models or the GBM
    mu, sd = desc.mean(0), desc.std(0)
    desc = (desc - mu) / np.where(sd > 0, sd, 1.0)
    return maccs, desc


def tanimoto(A, B):
    """Tanimoto similarity between binary fingerprint matrices."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    inter = A @ B.T
    a = A.sum(1)[:, None]
    b = B.sum(1)[None, :]
    return inter / np.maximum(a + b - inter, 1e-8)


def main():
    logf = open(f"{RESULTS}/strong_baselines.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    feats = pd.read_parquet(f"{DATA}/drug_features.parquet")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy")
    n_drugs = ecfp.shape[0]
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_uids = np.array(sorted(drug_level.index.tolist()))

    maccs, desc = build_structure_features(
        dict(zip(feats["drug_uid"].astype(int), feats["smiles"])), n_drugs)
    multi = np.concatenate([ecfp.astype(np.float32), maccs, desc], axis=1)
    log(f"{n_drugs} drugs; multi-representation dim = {multi.shape[1]}")

    rows = []
    for cls in CLASS_COLS:
        pos = drug_level[cls]
        pos_drugs = np.array(sorted(pos[pos == 1].index.tolist()))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_uids)]
        for sp in repeated_subsample_splits(pos_drugs, all_uids,
                                            n_repeats=N_REPEATS, seed0=SEED0):
            seen, test, ty = sp["seen_drugs"], sp["test_drugs"], sp["test_labels"]
            sy = pos.reindex(seen).fillna(0).values
            if len(np.unique(sy)) < 2:
                continue

            def add(name, score):
                rows.append({"class": cls, "seed": sp["seed"], "method": name,
                             "auroc": roc_auc_score(ty, score)})

            clf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                         class_weight="balanced", random_state=0)
            clf.fit(ecfp[seen], sy)
            add("ECFP4+RF (ours)", clf.predict_proba(ecfp[test])[:, 1])

            clf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                         class_weight="balanced", random_state=0)
            clf.fit(multi[seen], sy)
            add("MACCS+desc+ECFP4 RF", clf.predict_proba(multi[test])[:, 1])

            clf = GradientBoostingClassifier(random_state=0)
            clf.fit(multi[seen], sy)
            add("multi-rep GBM", clf.predict_proba(multi[test])[:, 1])

            # Tanimoto kNN: mean similarity to the k nearest seen positives
            S = tanimoto(ecfp[test], ecfp[seen][sy == 1])
            k = min(5, S.shape[1])
            add("Tanimoto kNN", np.sort(S, axis=1)[:, -k:].mean(axis=1))

            # GIP-style: logistic regression on the Tanimoto kernel to seen drugs
            Ktr = tanimoto(ecfp[seen], ecfp[seen])
            Kte = tanimoto(ecfp[test], ecfp[seen])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
            clf.fit(Ktr, sy)
            add("GIP-style kernel LR", clf.predict_proba(Kte)[:, 1])
        log(f"  {cls}: done")

    df = pd.DataFrame(rows)
    df.groupby(["class", "method"])["auroc"].mean().reset_index().to_csv(
        f"{RESULTS}/strong_baselines.csv", index=False)
    log("\n=== mean AUROC over six classes and 10 seeds ===")
    for m, v in df.groupby("method")["auroc"].mean().sort_values(ascending=False).items():
        log(f"  {m:<26} {v:.4f}")
    logf.close()


if __name__ == "__main__":
    main()
