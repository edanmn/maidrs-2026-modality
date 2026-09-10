"""
Leak-free evaluation protocol.

Positive-drug counts per therapeutic class are small (5-16 real drugs), so we
use repeated random drug-disjoint subsampling (Monte Carlo CV) rather than
strict K-fold: for each of N_REPEATS seeds, hold out a fraction of positive
drugs (min 2) plus a fixed-size random negative-drug pool as the test set;
everything else ("seen pool") is available for contrastive pretraining and
probe fitting. No drug's signatures ever appear in both seen and test pools
(drug-disjoint -> leak-free by construction, mirroring the fold-wise-GIP fix
from the SE-TTA revision).

Two pretraining regimes per repeat:
  - "standard": pretraining sees the seen pool for THIS class (other classes'
    members that overlap the seen pool are also fine, they're not excluded).
  - "class-holdout" (cold-start): pretraining excludes ALL members of the
    target class (not just the test-fold ones), so the encoder has never seen
    ANY drug from that therapeutic area. Directly measures generalization to
    an unseen disease area.
"""
import numpy as np


def repeated_subsample_splits(drug_uid_pos, drug_uid_all, n_repeats=10,
                                test_pos_frac=0.35, test_neg_pool=150, seed0=0):
    """Yields (seen_drugs, test_drugs, test_labels) for n_repeats seeds.
    drug_uid_pos: array of drug_uids that are positive for this class.
    drug_uid_all: array of ALL drug_uids with valid features.
    """
    all_set = set(drug_uid_all.tolist())
    pos_set = set(drug_uid_pos.tolist())
    neg_set = all_set - pos_set

    splits = []
    for r in range(n_repeats):
        rng = np.random.RandomState(seed0 + r)
        pos_arr = np.array(sorted(pos_set))
        n_test_pos = max(2, int(round(len(pos_arr) * test_pos_frac)))
        n_test_pos = min(n_test_pos, len(pos_arr) - 1) if len(pos_arr) > 1 else 1
        test_pos = rng.choice(pos_arr, size=n_test_pos, replace=False)

        neg_arr = np.array(sorted(neg_set))
        n_test_neg = min(test_neg_pool, len(neg_arr))
        test_neg = rng.choice(neg_arr, size=n_test_neg, replace=False)

        test_drugs = np.concatenate([test_pos, test_neg])
        test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
        seen_drugs = np.array(sorted(all_set - set(test_drugs.tolist())))

        splits.append({
            "seed": seed0 + r,
            "seen_drugs": seen_drugs,
            "test_drugs": test_drugs,
            "test_labels": test_labels,
        })
    return splits


def class_holdout_drugs(drug_uid_pos, drug_uid_all):
    """For cold-start pretraining: drugs to EXCLUDE from pretraining entirely
    (all positive drugs for this class)."""
    return set(drug_uid_pos.tolist())
