"""
Independent replication on LINCS Phase 1 (GSE92742).

Two questions, both raised as limitations of the Phase 2 work:
  1. Does the aggregation x model-class conjunction reproduce on an independent
     LINCS release? (Paper A: single-data-source limitation)
  2. With the Phase 1 QC fields that Wang et al. 2016 used (distil_ss,
     is_exemplar), does expression beat structure for SIDER endpoints?
     (Paper B: unresolved non-replication)

Phase 1 is 473,647 signatures x 12,328 genes. We slice landmark rows and the
compound columns we need rather than loading the full matrix.
"""
import os, sys, gzip, shutil
import numpy as np, pandas as pd, h5py

ROOT = "/mnt/c/Users/rishik/epilepsy_repurpose"
P1 = f"{ROOT}/lincs_phase1"
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RES = f"{ROOT}/ICLR2027_repurpose/results"


def ensure_decompressed():
    gz = f"{P1}/L5_phase1.gctx.gz"
    out = f"{P1}/L5_phase1.gctx"
    if os.path.exists(out) and os.path.getsize(out) > 1e10:
        print(f"  already decompressed: {os.path.getsize(out)/1e9:.1f} GB", flush=True)
        return out
    print("  decompressing (this takes a while)...", flush=True)
    with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=64 * 1024 * 1024)
    print(f"  done: {os.path.getsize(out)/1e9:.1f} GB", flush=True)
    return out


def load_slice(gctx, want_sigs, landmark_ids):
    """Read only landmark gene rows and the requested signature columns."""
    with h5py.File(gctx, "r") as f:
        rid = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["0/META/ROW/id"][:]]
        cid = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["0/META/COL/id"][:]]
        rid_pos = {g: i for i, g in enumerate(rid)}
        cid_pos = {c: i for i, c in enumerate(cid)}
        rows = np.array(sorted(rid_pos[g] for g in landmark_ids if g in rid_pos))
        keep_sigs = [s for s in want_sigs if s in cid_pos]
        cols = np.array(sorted(cid_pos[s] for s in keep_sigs))
        print(f"  slicing {len(rows)} genes x {len(cols)} signatures", flush=True)
        dset = f["0/DATA/0/matrix"]
        # gctx stores as (col, row); read in column chunks to bound memory
        out = np.empty((len(cols), len(rows)), dtype=np.float32)
        CH = 5000
        for s in range(0, len(cols), CH):
            block = cols[s:s + CH]
            out[s:s + len(block)] = dset[block[0]:block[-1] + 1][
                np.searchsorted(np.arange(block[0], block[-1] + 1), block)][:, rows]
            if s % 50000 == 0:
                print(f"    {s}/{len(cols)}", flush=True)
        ordered_sigs = [cid[i] for i in cols]
        ordered_genes = [rid[i] for i in rows]
    return out, ordered_sigs, ordered_genes


def main():
    gctx = ensure_decompressed()

    print("Loading Phase 1 metadata...", flush=True)
    si = pd.read_csv(f"{P1}/GSE92742_Broad_LINCS_sig_info.txt.gz", sep="\t", low_memory=False)
    sm = pd.read_csv(f"{P1}/GSE92742_Broad_LINCS_sig_metrics.txt.gz", sep="\t", low_memory=False)
    gi = pd.read_csv(f"{P1}/GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t", low_memory=False)
    landmark = gi[gi["pr_is_lm"] == 1]["pr_gene_id"].astype(str).tolist()
    print(f"  landmark genes: {len(landmark)}", flush=True)

    si = si.merge(sm[["sig_id", "distil_cc_q75", "distil_ss", "tas",
                       "pct_self_rank_q25", "is_exemplar"]], on="sig_id", how="left")
    cp = si[si["pert_type"] == "trt_cp"].copy()

    # restrict to compounds we already have structures for, matched by name
    drug = pd.read_parquet(f"{DATA}/drug_features.parquet")
    names = set(drug["pert_iname"])
    cp = cp[cp["pert_iname"].isin(names)]
    print(f"  Phase1 signatures for our compounds: {len(cp)}, "
          f"compounds: {cp['pert_iname'].nunique()}", flush=True)

    X, sigs, genes = load_slice(gctx, cp["sig_id"].tolist(), landmark)
    np.save(f"{DATA}/p1_expr.npy", X)
    pd.DataFrame({"sig_id": sigs}).merge(cp, on="sig_id", how="left").to_parquet(
        f"{DATA}/p1_meta.parquet")
    pd.Series(genes).to_csv(f"{DATA}/p1_genes.csv", index=False, header=["gene_id"])
    print(f"Saved p1_expr.npy {X.shape}", flush=True)


if __name__ == "__main__":
    main()
