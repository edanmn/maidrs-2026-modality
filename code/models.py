"""
Model architectures for cross-modal contrastive drug-repurposing.

- StructureEncoder: ECFP4 (2048-d) -> joint embedding space
- ExpressionEncoder: landmark gene expression (978-d) -> joint embedding space
- CLIPStyleAligner: symmetric InfoNCE contrastive objective between the two
- Baselines: unimodal MLPs, concat-fusion MLP (no contrastive pretraining),
  parameter-free k-NN connectivity baseline (mirrors the SE-TTA lesson that a
  non-parametric baseline must be included for a fair evaluation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims, dropout=0.2, final_act=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    if final_act is not None:
        layers.append(final_act)
    return nn.Sequential(*layers)


class StructureEncoder(nn.Module):
    def __init__(self, in_dim=2048, hidden=512, out_dim=256, dropout=0.2):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, out_dim], dropout=dropout)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


class ExpressionEncoder(nn.Module):
    def __init__(self, in_dim=978, hidden=512, out_dim=256, dropout=0.2):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, out_dim], dropout=dropout)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


class CLIPStyleAligner(nn.Module):
    """Symmetric InfoNCE between structure and expression embeddings of the
    same drug (positives), with in-batch other-drug pairs as negatives, plus
    an optional hard-negative reweighting term using drug-class labels to
    avoid counting same-class off-diagonal pairs as hard negatives."""

    def __init__(self, structure_dim=2048, expr_dim=978, embed_dim=256,
                 dropout=0.2, init_temp=0.07, learnable_temp=True):
        super().__init__()
        self.structure_encoder = StructureEncoder(structure_dim, 512, embed_dim, dropout)
        self.expression_encoder = ExpressionEncoder(expr_dim, 512, embed_dim, dropout)
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_temp)))))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(init_temp)))

    def encode(self, structure, expression):
        return self.structure_encoder(structure), self.expression_encoder(expression)

    def forward(self, structure, expression, drug_uid=None, class_label=None, class_mode="exclude"):
        """class_label: optional LongTensor, one label id per sample, -1 for
        "no known class" (or "not visible to this pretraining run" -- callers
        are responsible for only labeling drugs that are safe to expose,
        e.g. excluding held-out evaluation drugs to stay leak-free).
        class_mode:
          - "exclude": same-class-different-drug pairs are masked out of the
            softmax denominator entirely (neither rewarded nor penalized) --
            a debiasing intervention against false-negative collisions
            (Chuang et al., 2020-style debiased contrastive learning).
          - "positive": same-class-different-drug pairs are added to the
            positive set alongside same-drug pairs -- explicit supervised-
            contrastive pull (Khosla et al., 2020-style SupCon), directly
            shaping the embedding to cluster the class rather than merely
            removing interference.
        """
        z_s, z_e = self.encode(structure, expression)
        temp = self.log_temp.exp().clamp(min=1e-3, max=1.0)
        logits = z_s @ z_e.t() / temp  # [B, B]

        if drug_uid is not None:
            # Multiple signatures can share the same drug_uid within a batch;
            # treat all same-drug pairs as positives (supervised-contrastive
            # style), not just the diagonal.
            same_drug = drug_uid.unsqueeze(0) == drug_uid.unsqueeze(1)
            pos_mask = same_drug.float()
            logits_masked, logits_masked_t = logits, logits.t()

            if class_label is not None:
                known = class_label >= 0
                same_class = (class_label.unsqueeze(0) == class_label.unsqueeze(1)) \
                    & known.unsqueeze(0) & known.unsqueeze(1)
                if class_mode == "exclude":
                    exclude_mask = same_class & (~same_drug)  # same class, different drug -> ignore
                    logits_masked = logits.masked_fill(exclude_mask, float("-inf"))
                    logits_masked_t = logits.t().masked_fill(exclude_mask.t(), float("-inf"))
                elif class_mode == "positive":
                    pos_mask = (same_drug | same_class).float()  # same class -> also positive
                else:
                    raise ValueError(f"unknown class_mode: {class_mode}")

            log_prob_s2e = F.log_softmax(logits_masked, dim=1)
            log_prob_e2s = F.log_softmax(logits_masked_t, dim=1)
            # Use `where`, not `log_prob * pos_mask`: masked (-inf) entries
            # times a zero pos_mask weight is 0*-inf = NaN in float arithmetic,
            # not 0. pos_mask is only 1 where log_prob is guaranteed finite
            # (same-drug pairs are never excluded), so this is exact.
            zero = torch.zeros_like(log_prob_s2e)
            loss_s2e = -torch.where(pos_mask.bool(), log_prob_s2e, zero).sum(1) / pos_mask.sum(1).clamp(min=1)
            zero_t = torch.zeros_like(log_prob_e2s)
            loss_e2s = -torch.where(pos_mask.t().bool(), log_prob_e2s, zero_t).sum(1) / pos_mask.t().sum(1).clamp(min=1)
            loss = (loss_s2e.mean() + loss_e2s.mean()) / 2
        else:
            labels = torch.arange(z_s.size(0), device=z_s.device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss, z_s, z_e


class ConcatFusionMLP(nn.Module):
    """Non-contrastive baseline: direct supervised classifier on concatenated
    [structure; expression] features, trained end-to-end per drug-class task."""

    def __init__(self, structure_dim=2048, expr_dim=978, hidden=512, dropout=0.3):
        super().__init__()
        self.net = mlp([structure_dim + expr_dim, hidden, hidden, 1], dropout=dropout)

    def forward(self, structure, expression):
        x = torch.cat([structure, expression], dim=-1)
        return self.net(x).squeeze(-1)


class UnimodalMLP(nn.Module):
    """Baseline using only one modality (structure OR expression)."""

    def __init__(self, in_dim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, 1], dropout=dropout)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def alignment_loss(z_s, z_e, drug_uid):
    """Wang & Isola (2020) alignment metric: expected squared distance between
    positive (same-drug) pairs. Lower = better aligned."""
    same_drug = drug_uid.unsqueeze(0) == drug_uid.unsqueeze(1)
    same_drug.fill_diagonal_(False)
    idx = same_drug.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return None
    d = (z_s[idx[:, 0]] - z_e[idx[:, 1]]).pow(2).sum(-1)
    return d.mean().item()


def uniformity_loss(z, t=2.0):
    """Wang & Isola (2020) uniformity metric: log of average pairwise
    Gaussian potential. More negative = more uniform on the hypersphere."""
    sq_dists = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_dists).mean()).item()
