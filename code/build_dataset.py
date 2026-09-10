"""
Build the leak-free multi-disease drug-repurposing dataset from real LINCS L1000
Phase 2 data (GSE70138), real PubChem-resolved SMILES, and real drug-class
membership lists (curated from standard pharmacology / ATC classifications).

Output: ICLR2027_repurpose/data/dataset.parquet with columns:
    sig_id, pert_iname, cell_id, pert_idose, pert_itime, drug_uid,
    <978 landmark gene columns as a single 'expr' array column>,
    class_AED, class_Statin, class_SSRI, class_NSAID, class_BetaBlocker, class_Antipsychotic
plus ICLR2027_repurpose/data/drug_features.parquet with columns:
    drug_uid, pert_iname, smiles, ecfp4 (2048-bit array)
"""
import os
import numpy as np
import pandas as pd

ROOT = "/mnt/c/Users/rishik/epilepsy_repurpose"
OUT = f"{ROOT}/ICLR2027_repurpose/data"

# Real drug-class membership lists (standard pharmacology; ATC-consistent).
# These are the SAME lists validated against the real LINCS Phase 2 compound
# vocabulary earlier (see session log) -- only compounds confirmed present are kept.
DRUG_CLASSES = {
    "AED": ['levetiracetam', 'lamotrigine', 'carbamazepine', 'phenytoin', 'topiramate',
            'gabapentin', 'lacosamide', 'zonisamide', 'perampanel', 'primidone',
            'clonazepam', 'diazepam', 'lorazepam', 'tiagabine', 'felbamate', 'rufinamide'],
    "Statin": ['atorvastatin', 'simvastatin', 'pravastatin', 'rosuvastatin', 'pitavastatin'],
    "SSRI": ['fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram', 'fluvoxamine'],
    "NSAID": ['ibuprofen', 'naproxen', 'diclofenac', 'celecoxib', 'ketoprofen',
              'piroxicam', 'meloxicam', 'aspirin'],
    "BetaBlocker": ['propranolol', 'metoprolol', 'carvedilol', 'bisoprolol', 'nebivolol', 'timolol'],
    "Antipsychotic": ['haloperidol', 'risperidone', 'olanzapine', 'clozapine',
                       'quetiapine', 'aripiprazole', 'chlorpromazine'],
}


def compute_ecfp4(smiles, n_bits=2048, radius=2):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    for b in fp.GetOnBits():
        arr[b] = 1
    return arr


def main():
    print("Loading LINCS sig_info metadata...")
    sig = pd.read_csv(f"{ROOT}/lincs_phase2/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt",
                       sep="\t", low_memory=False)
    cp = sig[sig["pert_type"] == "trt_cp"].copy()
    print(f"  {len(cp)} compound-treatment signatures, {cp['pert_iname'].nunique()} unique compounds")

    print("Loading resolved SMILES...")
    smi_df = pd.read_csv(f"{OUT}/lincs_smiles.csv")
    smi_df = smi_df.dropna(subset=["smiles"])
    smi_map = dict(zip(smi_df["pert_iname"], smi_df["smiles"]))
    print(f"  {len(smi_map)} compounds with resolved SMILES "
          f"({len(smi_map) / cp['pert_iname'].nunique() * 100:.1f}% of unique compounds)")

    cp = cp[cp["pert_iname"].isin(smi_map)].copy()
    print(f"  {len(cp)} signatures retained after SMILES filter, {cp['pert_iname'].nunique()} compounds")

    print("Computing ECFP4 fingerprints (RDKit, real molecular structures)...")
    uniq_compounds = sorted(cp["pert_iname"].unique())
    fps = {}
    n_failed = 0
    for name in uniq_compounds:
        fp = compute_ecfp4(smi_map[name])
        if fp is None:
            n_failed += 1
            continue
        fps[name] = fp
    print(f"  ECFP4 computed for {len(fps)}/{len(uniq_compounds)} compounds ({n_failed} invalid SMILES)")

    # IMPORTANT: drug_uid must be a contiguous 0..n_valid-1 index that exactly
    # matches the row order of the ECFP4 matrix saved below (drug_uid is used
    # directly as a row index everywhere downstream -- no separate lookup
    # table). Assign uids only over compounds with a valid fingerprint, in
    # the same sorted order used to build the matrix.
    valid_compounds = sorted(fps.keys())
    drug_uid_map = {name: i for i, name in enumerate(valid_compounds)}

    cp = cp[cp["pert_iname"].isin(fps)].copy()
    cp["drug_uid"] = cp["pert_iname"].map(drug_uid_map)

    print("Loading landmark gene expression (Level 5, real LINCS profiles)...")
    expr = pd.read_parquet(f"{ROOT}/lincs_phase2/level5_landmark.parquet")
    expr = expr.set_index("inst_id")
    gene_cols = [c for c in expr.columns]
    print(f"  expression matrix: {expr.shape}")

    cp = cp[cp["sig_id"].isin(expr.index)].copy()
    print(f"  {len(cp)} signatures with matching expression rows")

    # Assign multi-label drug-class membership (real pharmacology-based positives)
    for cls, drugs in DRUG_CLASSES.items():
        drugs_lower = set(d.lower() for d in drugs)
        cp[f"class_{cls}"] = cp["pert_iname"].str.lower().isin(drugs_lower).astype(int)

    class_cols = [f"class_{c}" for c in DRUG_CLASSES]
    print("\nClass label counts (signature level):")
    print(cp[class_cols].sum())
    print("\nClass label counts (unique drug level):")
    print(cp.drop_duplicates("pert_iname")[class_cols].sum())

    # Save expression as a separate aligned numpy array (parquet w/ list column is slow)
    X_expr = expr.loc[cp["sig_id"]].values.astype(np.float32)
    np.save(f"{OUT}/expr_signatures.npy", X_expr)

    meta_cols = ["sig_id", "pert_iname", "drug_uid", "cell_id", "pert_idose", "pert_itime"] + class_cols
    cp[meta_cols].reset_index(drop=True).to_parquet(f"{OUT}/dataset_meta.parquet")

    # Drug-level ECFP4 feature table
    drug_rows = []
    for name in uniq_compounds:
        if name not in fps:
            continue
        drug_rows.append({"drug_uid": drug_uid_map[name], "pert_iname": name, "smiles": smi_map[name]})
    drug_df = pd.DataFrame(drug_rows)
    assert (drug_df["drug_uid"].values == np.arange(len(drug_df))).all(), \
        "drug_uid must equal row position (used as direct array index downstream)"
    ecfp_matrix = np.stack([fps[n] for n in drug_df["pert_iname"]])
    np.save(f"{OUT}/drug_ecfp4.npy", ecfp_matrix)
    drug_df.reset_index(drop=True).to_parquet(f"{OUT}/drug_features.parquet")

    with open(f"{OUT}/gene_cols.txt", "w") as f:
        f.write("\n".join(gene_cols))

    print(f"\nSaved: dataset_meta.parquet ({len(cp)} rows), expr_signatures.npy {X_expr.shape}, "
          f"drug_features.parquet ({len(drug_df)} drugs), drug_ecfp4.npy {ecfp_matrix.shape}")


if __name__ == "__main__":
    main()
