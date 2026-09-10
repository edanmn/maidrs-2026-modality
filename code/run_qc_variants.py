"""
QC-filtered representation variants.

These are the exploratory sweeps behind `results/qc_final.log` and
`results/final_push.log`. No claim in the paper rests on them; they are kept
because they are the analyses that led to the representations the paper does
use, and because a log without a script that produces it is not reproducible.

Two questions:

  qc_final.log   Do QC-aware representations (per-cell-line blocks, gold
                 signatures, TAS-weighted consensus) close the gap to structure,
                 either over all compounds or over transcriptionally responsive
                 compounds only? "Responsive" follows Sirci et al.: a compound
                 whose strongest signature reaches TAS >= 0.3. They report that
                 a majority of CMap compounds respond weakly and that class
                 recovery for those is near random, so restricting to responsive
                 compounds is the obvious thing to try.

  final_push.log The same restriction applied to the two task families, with the
                 richest QC representation (gold per cell line) and a fixed
                 50:50 blend.

Model choices, stated rather than implied:

  - Structure is a random forest on ECFP4 throughout, which the grid in
    `run_expression_grid.py` shows is model-robust.
  - In `qc_final.log` the expression arm is also a random forest. That panel
    predates the model-class finding and is kept in that form because it is what
    motivated looking at model class in the first place.
  - In `final_push.log` the expression arm is logistic regression, the corrected
    choice from Section V of the paper.
  - The blend is a fixed 50:50 average of predicted probabilities. It is not the
    nested weight selection of `run_sider.py`; selecting a blend weight on test
    predictions is defect 5's cousin and produced a spurious +0.008 in an
    earlier draft.

All representations fall back to a compound's own naive mean when it has no
signature meeting the criterion, rather than leaving an all-zero vector.

Writes results/qc_final.log and results/final_push.log.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import repeated_subsample_splits

ROOT = os.environ.get("MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose")
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{ROOT}/BIBM_MAIDRS/results")
os.makedirs(RESULTS, exist_ok=True)

CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
QC_REPEATS = 10
PUSH_REPEATS = 40
SEED0 = 42
N_JOBS = int(os.environ.get("MAIDRS_JOBS", "12"))
TAS_RESPONSIVE = 0.3
N_CELL_BLOCKS = 6
N_ENDPOINTS = 50
MIN_POS = MIN_NEG = 25


def rf():
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  class_weight="balanced", random_state=0)


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def build_qc_reps(meta, expr, qc, n_drugs):
    """The five QC-aware representations, each with a naive-mean fallback."""
    duid = meta["drug_uid"].values.astype(int)
    G = expr.shape[1]
    sig_by_drug = {}
    for s, d in enumerate(duid):
        sig_by_drug.setdefault(d, []).append(s)

    tas = qc["tas"].values
    gold = (qc["distil_cc_q75"].values >= 0.2) & (qc["pct_self_rank_q25"].values <= 5)
    hidose = (meta["pert_idose"].values == "10.0 um") & (meta["pert_itime"].values == "24 h")
    cells = meta["cell_id"].values
    top_cells = pd.Series(cells).value_counts().index[:N_CELL_BLOCKS].tolist()

    naive = np.zeros((n_drugs, G), np.float32)
    for d, s in sig_by_drug.items():
        naive[d] = expr[s].mean(axis=0)

    def percell(mask):
        out = np.zeros((n_drugs, G * len(top_cells)), np.float32)
        for bi, c in enumerate(top_cells):
            blk = np.zeros((n_drugs, G), np.float32)
            cm = (cells == c) & mask
            for d, s in sig_by_drug.items():
                sel = [i for i in s if cm[i]]
                blk[d] = expr[sel].mean(axis=0) if sel else naive[d]
            out[:, bi * G:(bi + 1) * G] = blk
        return out

    def best_single(score):
        out = naive.copy()
        for d, s in sig_by_drug.items():
            out[d] = expr[s[int(np.argmax(score[s]))]]
        return out

    def weighted(w):
        out = naive.copy()
        for d, s in sig_by_drug.items():
            ww = np.maximum(w[s], 0)
            if ww.sum() > 0:
                out[d] = (expr[s] * ww[:, None]).sum(0) / ww.sum()
        return out

    max_tas = np.zeros(n_drugs, np.float32)
    for d, s in sig_by_drug.items():
        max_tas[d] = np.nanmax(tas[s])

    reps = {"prev_best_percell_hidose": percell(hidose),
            "gold_percell": percell(gold),
            "gold_hi_percell": percell(gold & hidose),
            "bestTAS_sig": best_single(tas),
            "TASweighted_consensus": weighted(tas)}
    return reps, max_tas


def _pair(Xe, Xs, expr_model, seen, sy, test, ty):
    ms = rf(); ms.fit(Xs[seen], sy)
    ps = ms.predict_proba(Xs[test])[:, 1]
    me = rf() if expr_model == "rf" else lr()
    me.fit(Xe[seen], sy)
    pe = me.predict_proba(Xe[test])[:, 1]
    return (roc_auc_score(ty, ps), roc_auc_score(ty, pe),
            roc_auc_score(ty, 0.5 * ps + 0.5 * pe))


def class_sweep(Xe, Xs, expr_model, drug_level, uids, n_repeats):
    jobs = []
    for cls in CLASS_COLS:
        pos = drug_level[cls]
        pd_ = np.array(sorted(pos[pos == 1].index.tolist()))
        pd_ = pd_[np.isin(pd_, uids)]
        if len(pd_) < 3:
            continue
        for sp in repeated_subsample_splits(pd_, uids, n_repeats=n_repeats, seed0=SEED0):
            seen, test, ty = sp["seen_drugs"], sp["test_drugs"], sp["test_labels"]
            sy = pos.reindex(seen).fillna(0).values
            if len(np.unique(sy)) < 2:
                continue
            jobs.append((seen, sy, test, ty))
    out = Parallel(n_jobs=N_JOBS)(delayed(_pair)(Xe, Xs, expr_model, *j) for j in jobs)
    return np.array(out)      # columns: struct, expr, blend


def main():
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    qc = pd.read_parquet(f"{DATA}/sig_qc.parquet")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    n_drugs = ecfp.shape[0]
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_uids = np.array(sorted(drug_level.index.tolist()))
    reps, max_tas = build_qc_reps(meta, expr, qc, n_drugs)
    responsive = all_uids[max_tas[all_uids] >= TAS_RESPONSIVE]

    # ---------------------------------------------------------- qc_final.log
    with open(f"{RESULTS}/qc_final.log", "w") as f:
        def log(m):
            print(m, flush=True); f.write(m + "\n"); f.flush()

        log("QC-aware representations, expression arm = random forest, "
            f"structure = RF on ECFP4, fixed 50:50 blend, {QC_REPEATS} repeats.")
        for label, uids in (("ALL DRUGS", all_uids),
                            (f"TRANSCRIPTIONALLY RESPONSIVE ONLY "
                             f"(max TAS>={TAS_RESPONSIVE}, Sirci-style)", responsive)):
            log(f"\n########## {label} (n_drugs={len(uids)}) ##########")
            log(f"{'representation':<28}{'expr_only':>10}{'struct':>9}"
                f"{'blend50':>9}{'gain':>8}{'p':>10}")
            ref = None
            for name, Xe in reps.items():
                v = class_sweep(Xe, ecfp, "rf", drug_level, uids, QC_REPEATS)
                s, e, b = v[:, 0], v[:, 1], v[:, 2]
                if ref is None:
                    ref = s.mean()
                    log(f"{'[structure ref]':<28}{'-':>10}{ref:>9.3f}"
                        f"{'-':>9}{'-':>8}{'-':>10}")
                _, p = wilcoxon(b, s)
                log(f"{name:<28}{e.mean():>10.3f}{s.mean():>9.3f}"
                    f"{b.mean():>9.3f}{b.mean()-s.mean():>+8.3f}{p:>10.4f}")

    # -------------------------------------------------------- final_push.log
    with open(f"{RESULTS}/final_push.log", "w") as f:
        def log(m):
            print(m, flush=True); f.write(m + "\n"); f.flush()

        log(f"=== A: DRUG CLASS, responsive drugs, gold_percell, {PUSH_REPEATS} repeats ===")
        v = class_sweep(reps["gold_percell"], ecfp, "lr", drug_level,
                        responsive, PUSH_REPEATS)
        s, e, b = v[:, 0], v[:, 1], v[:, 2]
        _, p = wilcoxon(b, s)
        log(f"n={len(v)} struct={s.mean():.4f} expr={e.mean():.4f} "
            f"blend={b.mean():.4f} gain={b.mean()-s.mean():+.4f} p={p:.4f}")

        smap = pd.read_csv(f"{DATA}/lincs_sider_map.csv")
        pairs = pd.read_csv(f"{DATA}/sider_pairs.csv")
        cid2uid = dict(zip(smap["cid"], smap["drug_uid"].astype(int)))
        su = np.array(sorted(set(cid2uid.values()) & set(responsive.tolist())))
        pos = {}
        for cid, se in zip(pairs["cid_flat"], pairs["se_name"]):
            u = cid2uid.get(cid)
            if u is not None and u in set(su.tolist()):
                pos.setdefault(se, set()).add(u)
        usable = sorted(se for se, t in pos.items()
                        if len(t) >= MIN_POS and len(su) - len(t) >= MIN_NEG)
        rng = np.random.RandomState(0)
        chosen = [usable[i] for i in rng.choice(len(usable),
                                                min(N_ENDPOINTS, len(usable)), replace=False)]
        Xe = reps["gold_percell"]

        def endpoint(se):
            y = np.array([1 if u in pos[se] else 0 for u in su])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            ps, pe = np.zeros(len(y)), np.zeros(len(y))
            for tr, te in skf.split(su, y):
                a, b_ = su[tr], su[te]
                m = rf(); m.fit(ecfp[a], y[tr]); ps[te] = m.predict_proba(ecfp[b_])[:, 1]
                q = lr(); q.fit(Xe[a], y[tr]); pe[te] = q.predict_proba(Xe[b_])[:, 1]
            return (roc_auc_score(y, ps), roc_auc_score(y, pe),
                    roc_auc_score(y, 0.5 * ps + 0.5 * pe))

        out = np.array(Parallel(n_jobs=N_JOBS)(delayed(endpoint)(se) for se in chosen))
        s, e, b = out[:, 0], out[:, 1], out[:, 2]
        _, p = wilcoxon(b, s)
        log(f"\n=== B: SIDER, gold_percell, responsive drugs only (n={len(su)}) ===")
        log(f"n_se={len(out)} struct={s.mean():.4f} expr={e.mean():.4f} blend={b.mean():.4f}")
        log(f"gain={b.mean()-s.mean():+.4f} p={p:.4f} wins={(b>s).sum()}/{len(out)}")


if __name__ == "__main__":
    main()
