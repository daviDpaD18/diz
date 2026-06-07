"""
DynamicViT — token pruning for ViT-B/16.

Rao et al. (2021) "DynamicViT: Efficient Vision Transformers with
Dynamic Token Sparsification" (NeurIPS 2021).

Two prediction MLPs are inserted after transformer blocks 4 and 8
(1-indexed, so 0-indexed positions 3 and 7). Each MLP scores every
patch token as keep/drop. During training the decision is made
differentiable via Gumbel-softmax. During inference a hard top-k
selection is used to give an exact keeping ratio.

Sequence length stays constant throughout (soft masking: dropped tokens
are zeroed, not removed). This keeps batching simple and lets us recover
the final binary token mask by multiplying the two stage decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictorMLP(nn.Module):
    """Lightweight 2-layer MLP that outputs keep/drop logits per token."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),   # index 0 = drop, index 1 = keep
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [B, N, 2]


class DynamicViT(nn.Module):
    """
    Wraps a pretrained timm ViT-B/16 with two learned token-pruning stages.

    Args:
        base_model:       A timm ViT loaded from a checkpoint (weights already loaded).
        final_keep_ratio: Fraction of the original 196 patch tokens to keep at the end.
                          E.g. 0.5 → 98 tokens survive.

    The per-stage keep ratio is sqrt(final_keep_ratio) so the two stages
    compound to the target. E.g. for 50%: each stage keeps ~70.7% of tokens.

    Only the two PredictorMLP modules are randomly initialised.
    Everything else inherits the pretrained weights.
    """

    PRUNE_AFTER = (3, 7)   # 0-indexed block positions (after block 4 and block 8)

    def __init__(self, base_model: nn.Module, final_keep_ratio: float):
        super().__init__()

        # Borrow all pretrained components
        self.patch_embed = base_model.patch_embed
        self.cls_token   = base_model.cls_token
        self.pos_embed   = base_model.pos_embed
        self.pos_drop    = getattr(base_model, 'pos_drop',  nn.Identity())
        self.patch_drop  = getattr(base_model, 'patch_drop', nn.Identity())
        self.norm_pre    = getattr(base_model, 'norm_pre',  nn.Identity())
        self.blocks      = base_model.blocks
        self.norm        = base_model.norm
        self.head        = base_model.head

        self.final_keep_ratio = final_keep_ratio
        self.stage_ratio      = final_keep_ratio ** 0.5   # per-stage target

        embed_dim = base_model.embed_dim
        self.predictors = nn.ModuleList([
            PredictorMLP(embed_dim),
            PredictorMLP(embed_dim),
        ])

    # ── forward ──────────────────────────────────────────────────────────────

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        return self.pos_drop(x + self.pos_embed)

    def forward(
        self,
        x: torch.Tensor,
        return_decisions: bool = False,
    ):
        x = self.patch_embed(x)
        x = self._embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        decisions = []
        pred_idx  = 0

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i in self.PRUNE_AFTER:
                patch        = x[:, 1:, :]            # exclude CLS: [B, N, C]
                B, N, C      = patch.shape
                logits       = self.predictors[pred_idx](patch)   # [B, N, 2]

                if self.training:
                    # Differentiable hard decision — gradients flow through
                    decision = F.gumbel_softmax(logits, tau=1.0, hard=True)[:, :, 1]
                else:
                    # Hard top-k — exact keeping ratio
                    n_keep   = max(1, int(N * self.stage_ratio))
                    scores   = logits[:, :, 1]
                    topk_idx = scores.topk(n_keep, dim=1).indices
                    decision = torch.zeros(B, N, device=x.device)
                    decision.scatter_(1, topk_idx, 1.0)

                decisions.append(decision)

                # Soft mask: zero dropped tokens, preserve sequence length
                cls   = x[:, :1, :]
                patch = patch * decision.unsqueeze(-1)
                x     = torch.cat([cls, patch], dim=1)
                pred_idx += 1

        x      = self.norm(x)
        logits = self.head(x[:, 0])   # CLS classification

        if return_decisions:
            return logits, decisions
        return logits

    # ── token mask extraction ─────────────────────────────────────────────────

    @torch.no_grad()
    def get_token_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference and return the binary surviving-token mask.

        Returns:
            [B, 14, 14] float tensor — 1.0 where a patch token survived
            both pruning stages, 0.0 where it was dropped.
        """
        was_training = self.training
        self.eval()
        _, decisions = self.forward(x, return_decisions=True)
        if was_training:
            self.train()

        # A token survives only if it passes BOTH stages
        mask = decisions[0] * decisions[1]   # element-wise AND: [B, 196]
        return mask.reshape(x.shape[0], 14, 14)


# ── loss ──────────────────────────────────────────────────────────────────────

def ratio_loss(
    decisions: list,
    final_keep_ratio: float,
    lam: float = 2.0,
) -> torch.Tensor:
    """
    Penalises deviation from the target keeping ratio at each stage.
    Both stages target sqrt(final_keep_ratio).
    lam = 2.0 follows the DynamicViT paper default.
    """
    stage_target = final_keep_ratio ** 0.5
    loss = torch.tensor(0.0, device=decisions[0].device, requires_grad=False)
    for decision in decisions:
        loss = loss + (decision.float().mean() - stage_target) ** 2
    return lam * loss
