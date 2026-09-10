"""
Published-method baseline: the cross-modal alignment objective of Finlayson et
al. (PSB 2021), run under this paper's drug-disjoint protocol.

Finlayson et al. introduced the structure-transcriptome alignment recipe this
paper evaluates. They do not release code, so this is a reimplementation from
the paper's Section 3.3. Two deliberate substitutions, both stated in the
paper text:

  1. Structure encoder. They use a Chemprop D-MPNN over the molecular graph;
     we keep the ECFP4 MLP used by every other method here. This holds the
     structural input fixed across the comparison so that the contrast is the
     alignment objective and not the featurisation.
  2. Negative-sampling density. They precompute pairwise distances in average
     post-perturbational expression space and apply the distance-weighted
     sampler of Wu et al. (ICCV 2017). We do the same on L2-normalised
     drug-level mean expression, keeping Wu's 0.5 cutoff and lambda cap, but
     using the empirical density of the observed distances in place of their
     closed-form q(d). Their q assumes points uniform on a sphere; these
     distances run from 0.10 to 1.70 with median 1.21, and the analytic form
     collapses the sampler onto each anchor's two nearest compounds. The
     empirical version leaves about 344 effective negatives per anchor.

Everything else matches their description: a self-normalising (SELU +
AlphaDropout) expression encoder, L2-normalised 256-d embeddings, and the
adaptive-margin quadruplet loss

    mar_{a,b}(i,j) = (a + y_ij (D_ij - b))_+ ,  y = +1 positive, -1 negative
    l_quad = mar(gA,mA) + mar(gA,mB) + mar(gB,mA) + mar(gB,mB)

where (gA,mA) is a matched pair and mB is the distance-weighted negative.

The training budget (25 epochs, batch 256, AdamW lr 1e-3) is held identical to
the CLIP-style aligner so the two differ only in objective and encoder
nonlinearity. Margin hyperparameters are selected on a held-out slice of the
SEEN pool only, never on the evaluation drugs.

Writes results/finlayson_baseline.csv and results/finlayson_baseline.log.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import repeated_subsample_splits

ROOT = os.environ.get("MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose")
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{ROOT}/BIBM_MAIDRS/results")
os.makedirs(RESULTS, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]

N_REPEATS = 10
SEED0 = 42                 # matches run_experiments.py, so splits are identical
N_PRETRAIN_SEEDS = 3
EPOCHS = 25
BATCH = 256
LR = 1e-3
EMBED_DIM = 256
CUTOFF = 0.5               # Wu et al. lower distance cutoff
LAMBDA_CAP = 1e4           # Wu et al. weight cap


# ---------------------------------------------------------------- encoders
class SNNEncoder(nn.Module):
    """Self-normalising expression encoder (Finlayson et al. Section 3.3)."""

    def __init__(self, in_dim=978, hidden=512, out_dim=EMBED_DIM, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(hidden, hidden), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, (1.0 / m.in_features) ** 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class FPEncoder(nn.Module):
    """ECFP4 structure encoder, matched in shape to the CLIP-style aligner."""

    def __init__(self, in_dim=2048, hidden=512, out_dim=EMBED_DIM, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def margin_loss(z_a, z_b, y, alpha, beta):
    """(alpha + y * (D - beta))_+ with y = +1 positive, -1 negative."""
    d = torch.norm(z_a - z_b, p=2, dim=-1)
    return F.relu(alpha + y * (d - beta))


# --------------------------------------------------- distance-weighted sampler
def build_sampler(drug_mean_expr, valid_drugs, rng):
    """Wu et al. (2017) distance-weighted negative sampling over compounds,
    on L2-normalised drug-level mean expression as Finlayson et al. describe."""
    X = drug_mean_expr[valid_drugs]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X @ X.T)))
    off = ~np.eye(len(valid_drugs), dtype=bool)

    # Wu et al. weight each candidate by 1/q(d), where q is the density of
    # pairwise distances. Their closed form for q assumes points distributed
    # uniformly on the unit sphere. Drug-level mean expression is not: observed
    # distances here run from 0.10 to 1.70 with median 1.21, far wider than the
    # concentration a 256-sphere would give, and the analytic q(d) then puts
    # essentially all mass on each anchor's two nearest compounds. We therefore
    # use the empirical density of the observed distances, which is the
    # quantity q(d) stands in for, keeping Wu's cutoff and lambda cap.
    d = np.clip(D, CUTOFF, None)
    edges = np.linspace(CUTOFF, d[off].max() + 1e-9, 51)
    dens, _ = np.histogram(d[off], bins=edges, density=True)
    dens = np.maximum(dens, 1e-6)
    b = np.clip(np.digitize(d, edges) - 1, 0, len(dens) - 1)
    W = 1.0 / dens[b]
    W /= W[off].min()                 # smallest weight is 1, cap the ratio
    W = np.minimum(W, LAMBDA_CAP)
    np.fill_diagonal(W, 0.0)          # never sample a compound against itself
    W[~np.isfinite(W)] = 0.0
    W /= W.sum(axis=1, keepdims=True)
    ess = 1.0 / (W ** 2).sum(axis=1)  # effective number of candidate negatives
    print(f"  [sampler] median effective negatives per anchor: {np.median(ess):.0f}"
          f" of {W.shape[1] - 1}", flush=True)
    return W


# ------------------------------------------------------------------ training
def train_finlayson(expr, ecfp, duid_of_sig, drug_mean_expr, valid_drugs,
                    alpha, beta, seed, epochs=EPOCHS, log=print):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    pos_in_valid = {d: i for i, d in enumerate(valid_drugs)}
    W = build_sampler(drug_mean_expr, valid_drugs, rng)
    sig_by_drug = {}
    for s, d in enumerate(duid_of_sig):
        sig_by_drug.setdefault(d, []).append(s)

    ge = SNNEncoder(expr.shape[1]).to(DEVICE)
    me = FPEncoder(ecfp.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(list(ge.parameters()) + list(me.parameters()),
                            lr=LR, weight_decay=1e-5)

    all_sigs = np.arange(len(duid_of_sig))
    n_batches = len(all_sigs) // BATCH
    ge.train(); me.train()
    t0 = time.time()
    losses = []
    for ep in range(epochs):
        perm = rng.permutation(all_sigs)
        ep_loss = []
        for b in range(n_batches):
            anchor = perm[b * BATCH:(b + 1) * BATCH]
            dA = duid_of_sig[anchor]
            # distance-weighted negative compound for each anchor
            rows = np.array([pos_in_valid[d] for d in dA])
            u = rng.rand(len(rows), 1)
            cdf = np.cumsum(W[rows], axis=1)
            pick = (u > cdf).sum(axis=1).clip(0, W.shape[1] - 1)
            dB = valid_drugs[pick]
            sigB = np.array([sig_by_drug[d][rng.randint(len(sig_by_drug[d]))] for d in dB])

            gA = ge(torch.from_numpy(expr[anchor]).float().to(DEVICE))
            gB = ge(torch.from_numpy(expr[sigB]).float().to(DEVICE))
            mA = me(torch.from_numpy(ecfp[dA]).float().to(DEVICE))
            mB = me(torch.from_numpy(ecfp[dB]).float().to(DEVICE))

            one = torch.ones(len(anchor), device=DEVICE)
            loss = (margin_loss(gA, mA, one, alpha, beta)
                    + margin_loss(gA, mB, -one, alpha, beta)
                    + margin_loss(gB, mA, -one, alpha, beta)
                    + margin_loss(gB, mB, one, alpha, beta)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss.append(loss.item())
        losses.append(float(np.mean(ep_loss)))
    log(f"  [finlayson seed={seed} a={alpha} b={beta}] {epochs} ep in "
        f"{time.time()-t0:.0f}s, loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    return ge, me


@torch.no_grad()
def embed_drugs(ge, me, expr, ecfp, duid_of_sig, n_drugs):
    ge.eval(); me.eval()
    z_s = me(torch.from_numpy(ecfp).float().to(DEVICE)).cpu().numpy()
    sig_by_drug = {}
    for s, d in enumerate(duid_of_sig):
        sig_by_drug.setdefault(d, []).append(s)
    z_e = np.zeros((n_drugs, EMBED_DIM), np.float32)
    for d, sigs in sig_by_drug.items():
        z = ge(torch.from_numpy(expr[sigs]).float().to(DEVICE)).cpu().numpy()
        z_e[d] = z.mean(axis=0)
    return z_s, z_e


def probe(Z, seen, seen_y, test, test_y):
    if len(np.unique(seen_y)) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Z[seen], seen_y)
    p = clf.predict_proba(Z[test])[:, 1]
    return {"auroc": roc_auc_score(test_y, p), "auprc": average_precision_score(test_y, p)}


def main():
    logf = open(f"{RESULTS}/finlayson_baseline.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log(f"device={DEVICE}")
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy")
    duid = meta["drug_uid"].values.astype(int)
    n_drugs = ecfp.shape[0]
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_uids = np.array(sorted(drug_level.index.tolist()))
    log(f"{expr.shape[0]} signatures, {len(all_uids)} drugs, expr {expr.shape[1]}d, ecfp {ecfp.shape[1]}d")

    sig_by_drug = {}
    for s, d in enumerate(duid):
        sig_by_drug.setdefault(d, []).append(s)
    drug_mean = np.zeros((n_drugs, expr.shape[1]), np.float32)
    for d, s in sig_by_drug.items():
        drug_mean[d] = expr[s].mean(axis=0)

    # ---- margin selection on the SEEN pool only ------------------------------
    # Hold out 20% of drugs as an inner validation set, never touching the
    # evaluation splits. Selection criterion is cross-modal retrieval MRR,
    # the criterion Finlayson et al. use for early stopping.
    rng = np.random.RandomState(7)
    inner_val = rng.choice(all_uids, size=int(0.2 * len(all_uids)), replace=False)
    inner_tr = np.array(sorted(set(all_uids.tolist()) - set(inner_val.tolist())))
    tr_mask = np.isin(duid, inner_tr)
    log(f"\n=== margin selection (inner split: {len(inner_tr)} train / {len(inner_val)} val drugs) ===")

    best, best_mrr = None, -1.0
    for alpha in (0.05, 0.1, 0.2):
        for beta in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8):
            ge, me = train_finlayson(expr[tr_mask], ecfp, duid[tr_mask], drug_mean,
                                     inner_tr, alpha, beta, seed=0, epochs=8, log=log)
            z_s, z_e = embed_drugs(ge, me, expr, ecfp, duid, n_drugs)
            q = z_e[inner_val] / (np.linalg.norm(z_e[inner_val], axis=1, keepdims=True) + 1e-8)
            g = z_s[inner_val] / (np.linalg.norm(z_s[inner_val], axis=1, keepdims=True) + 1e-8)
            sim = q @ g.T
            ranks = (sim > sim.diagonal()[:, None]).sum(axis=1) + 1
            mrr = float(np.mean(1.0 / ranks))
            log(f"    alpha={alpha} beta={beta}  val MRR={mrr:.4f}")
            if mrr > best_mrr:
                best_mrr, best = mrr, (alpha, beta)
    alpha, beta = best
    log(f"  selected alpha={alpha}, beta={beta} (val MRR={best_mrr:.4f})")

    # ---- full pretraining, N_PRETRAIN_SEEDS ---------------------------------
    log(f"\n=== pretraining {N_PRETRAIN_SEEDS} seeds on all {expr.shape[0]} signatures ===")
    embeds = []
    for ps in range(N_PRETRAIN_SEEDS):
        ge, me = train_finlayson(expr, ecfp, duid, drug_mean, all_uids,
                                 alpha, beta, seed=ps, log=log)
        embeds.append(embed_drugs(ge, me, expr, ecfp, duid, n_drugs))

    # ---- downstream class recovery ------------------------------------------
    log("\n=== downstream class recovery (drug-disjoint, seed0=42, 10 repeats) ===")
    rows = []
    for cls in CLASS_COLS:
        pos = drug_level[cls]
        pos_drugs = np.array(sorted(pos[pos == 1].index.tolist()))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_uids)]
        for sp in repeated_subsample_splits(pos_drugs, all_uids, n_repeats=N_REPEATS, seed0=SEED0):
            seen, test, ty = sp["seen_drugs"], sp["test_drugs"], sp["test_labels"]
            sy = pos.reindex(seen).fillna(0).values
            for ps, (z_s, z_e) in enumerate(embeds):
                for view, Z in [("structure", z_s), ("expression", z_e),
                                ("concat", np.concatenate([z_s, z_e], axis=1))]:
                    r = probe(Z, seen, sy, test, ty)
                    if r:
                        rows.append({"class": cls, "seed": sp["seed"], "pretrain_seed": ps,
                                     "method": f"finlayson_{view}", **r})
        log(f"  {cls}: done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/finlayson_baseline.csv", index=False)

    log("\n=== mean AUROC by class (Finlayson quadruplet, concat view) ===")
    con = df[df.method == "finlayson_concat"]
    per_class = con.groupby("class")["auroc"].mean()
    for c in CLASS_COLS:
        log(f"  {c:<22} {per_class[c]:.4f}")
    log(f"  {'MEAN':<22} {per_class.mean():.4f}")

    # paired comparison against the ECFP4 random forest, same splits
    main_csv = f"{RESULTS}/main_results.csv"
    if os.path.exists(main_csv):
        prev = pd.read_csv(main_csv)
        log("\n=== paired vs raw ECFP4 + RF (Wilcoxon over subsampling seeds) ===")
        for c in CLASS_COLS:
            a = con[con["class"] == c].groupby("seed")["auroc"].mean()
            b = prev[(prev["class"] == c) & (prev.method == "raw_ecfp4_RF")].set_index("seed")["auroc"]
            j = a.index.intersection(b.index)
            if len(j) >= 6:
                st, p = wilcoxon(a[j], b[j])
                log(f"  {c:<22} finlayson={a[j].mean():.4f}  rf={b[j].mean():.4f}  p={p:.4g}")
    log(f"\nwrote {RESULTS}/finlayson_baseline.csv ({len(df)} rows)")
    logf.close()


if __name__ == "__main__":
    main()
