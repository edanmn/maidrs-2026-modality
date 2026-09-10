"""Fetch real canonical SMILES for LINCS Phase 2 compound names via PubChem PUG-REST.
Resumable checkpoint file; parallelized with a small thread pool (PubChem allows ~5 req/s)."""
import urllib.request, urllib.parse, json, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/SMILES,ConnectivitySMILES,MolecularFormula/JSON"
ROOT = "/mnt/c/Users/rishik/epilepsy_repurpose"
N_WORKERS = 4
DELAY = 0.6  # per-thread delay -> ~4*(1/0.6) ~= 6.6 req/s combined ceiling, throttled by response time in practice


def fetch_one(name, retries=3):
    url = BASE.format(urllib.parse.quote(name))
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
                props = data["PropertyTable"]["Properties"][0]
                smi = props.get("SMILES") or props.get("ConnectivitySMILES")
                time.sleep(DELAY)
                return name, smi
        except urllib.error.HTTPError as e:
            if e.code == 404:
                time.sleep(DELAY)
                return name, None
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return name, None


def main():
    sig = pd.read_csv(f'{ROOT}/lincs_phase2/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt', sep='\t', low_memory=False)
    cp = sig[sig['pert_type'] == 'trt_cp']
    names = sorted(cp['pert_iname'].dropna().unique().tolist())
    print(f"Total unique compounds to resolve: {len(names)}", flush=True)

    out_path = f'{ROOT}/ICLR2027_repurpose/data/lincs_smiles.csv'
    done = {}
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = dict(zip(prev['pert_iname'], prev['smiles']))
        print(f"Resuming: {len(done)} already resolved", flush=True)

    todo = [n for n in names if n not in done]
    print(f"Remaining to fetch: {len(todo)}", flush=True)

    results = list(done.items())
    n_processed = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(fetch_one, n) for n in todo]
        for fut in as_completed(futures):
            name, smi = fut.result()
            results.append((name, smi))
            n_processed += 1
            if n_processed % 50 == 0:
                pd.DataFrame(results, columns=['pert_iname', 'smiles']).to_csv(out_path, index=False)
                n_ok = sum(1 for _, s in results if s)
                print(f"[{n_processed}/{len(todo)} new; {len(results)} total] resolved so far: {n_ok}/{len(results)}", flush=True)

    pd.DataFrame(results, columns=['pert_iname', 'smiles']).to_csv(out_path, index=False)
    n_ok = sum(1 for _, s in results if s)
    print(f"DONE. Resolved {n_ok}/{len(results)} compounds -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
