"""Phonetic-prior decipherment model with DP objectives.

This module follows the paper algorithm structure:
1) ComputeCharDistr
2) EditDistDP
3) WordBoundaryDP
4) SGD step
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


Tensor = torch.Tensor


def _softmin(values: Sequence[Tensor]) -> Tensor:
    stacked = torch.stack([-v for v in values], dim=0)
    return -torch.logsumexp(stacked, dim=0)


@dataclass
class PhoneticPriorConfig:
    temperature: float = 0.2
    alpha: float = 3.5
    lambda_cov: float = 10.0
    lambda_loss: float = 100.0
    min_span: int = 4
    max_span: int = 10
    embedding_dim: int = 64
    lr: float = 0.2
    p_o: float = 0.2
    seed: int = 1234


@dataclass
class TrainStepOutput:
    objective: float
    quality: float
    omega_cov: float
    omega_loss: float
    num_sequences: int


class PhoneticPriorModel(nn.Module):
    """Joint segmentation and cognate-alignment model."""

    def __init__(
        self,
        lost_chars: Sequence[str],
        known_chars: Sequence[str],
        known_ipa_features: Optional[Tensor] = None,
        config: Optional[PhoneticPriorConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or PhoneticPriorConfig()
        self.lost_chars = sorted(set(lost_chars))
        self.known_chars = sorted(set(known_chars))
        if not self.lost_chars:
            self.lost_chars = ["?"]
        if not self.known_chars:
            self.known_chars = ["?"]

        self.lost2idx = {c: i for i, c in enumerate(self.lost_chars)}
        self.known2idx = {c: i for i, c in enumerate(self.known_chars)}

        d = self.config.embedding_dim
        if known_ipa_features is None:
            known_ipa_features = torch.eye(len(self.known_chars), dtype=torch.float32)
        if known_ipa_features.ndim != 2:
            raise ValueError("known_ipa_features must be rank-2")

        feature_dim = int(known_ipa_features.shape[1])
        self.register_buffer("known_ipa_features", known_ipa_features.float())
        self.ipa_projector = nn.Linear(feature_dim, d, bias=False)

        # Eq.3 logits over known/lost character pairs.
        self.mapping_logits = nn.Parameter(torch.zeros(len(self.known_chars), len(self.lost_chars)))
        nn.init.xavier_uniform_(self.mapping_logits)

    def known_embeddings(self) -> Tensor:
        return self.ipa_projector(self.known_ipa_features)

    def compute_char_distr(self) -> Tensor:
        """Eq.3: softmax(dot / T) over lost chars given known chars."""
        k_emb = self.known_embeddings()  # K x D

        # Eq.4: build lost embeddings as weighted sums of known IPA embeddings.
        mix = self.mapping_logits.softmax(dim=0)  # K x L, normalized over known chars.
        l_emb = mix.transpose(0, 1) @ k_emb  # L x D

        scores = (k_emb @ l_emb.transpose(0, 1)) / max(self.config.temperature, 1e-6)  # K x L
        return scores.log_softmax(dim=1).exp()

    def edit_distance_dp(self, lost_token: str, known_token: str, char_distr: Tensor) -> Tensor:
        """Section 3.2.2 monotonic alignment with substitution/deletion/insertion."""
        m = len(lost_token)
        n = len(known_token)
        alpha = torch.tensor(float(self.config.alpha), dtype=torch.float32, device=char_distr.device)

        dp: List[List[Tensor]] = [[torch.tensor(0.0, device=char_distr.device) for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + alpha
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + alpha

        for i in range(1, m + 1):
            lch = lost_token[i - 1]
            l_idx = self.lost2idx.get(lch, 0)
            for j in range(1, n + 1):
                kch = known_token[j - 1]
                k_idx = self.known2idx.get(kch, 0)

                sub_cost = -torch.log(char_distr[k_idx, l_idx] + 1e-9)
                candidates = [
                    dp[i - 1][j] + alpha,      # deletion
                    dp[i][j - 1] + alpha,      # insertion
                    dp[i - 1][j - 1] + sub_cost,  # substitution
                ]

                # Two-adjacent-index insertion transition.
                if j >= 2:
                    candidates.append(dp[i][j - 2] + alpha)
                dp[i][j] = _softmin(candidates)

        final_cost = dp[m][n]
        return -final_cost

    def word_boundary_dp(
        self,
        inscription: str,
        known_vocab: Sequence[str],
        char_distr: Tensor,
    ) -> Tuple[Tensor, float]:
        """Section 3.3 objective over latent Z with O and E_l tags."""
        seq = inscription.replace(" ", "")
        n = len(seq)
        if n == 0:
            return torch.tensor(0.0, device=char_distr.device), 0.0

        min_span = max(1, self.config.min_span)
        max_span = max(min_span, self.config.max_span)

        p_o = max(min(self.config.p_o, 0.99), 0.01)
        log_p_o = math.log(p_o)
        remaining = max(1, max_span - min_span + 1)
        log_p_el = {l: math.log((1.0 - p_o) / remaining) for l in range(min_span, max_span + 1)}

        neg_inf = torch.tensor(-1e9, dtype=torch.float32, device=char_distr.device)
        dp = [neg_inf.clone() for _ in range(n + 1)]
        best_cov = [0.0 for _ in range(n + 1)]
        dp[0] = torch.tensor(0.0, dtype=torch.float32, device=char_distr.device)

        vocab = [w for w in known_vocab if w]
        if not vocab:
            vocab = ["?"]

        for i in range(1, n + 1):
            candidates: List[Tensor] = []

            # O: unmatched single char.
            c_o = dp[i - 1] + log_p_o
            candidates.append(c_o)
            best_cov_i = best_cov[i - 1]

            for l in range(min_span, max_span + 1):
                if i - l < 0:
                    continue
                span = seq[i - l: i]
                token_scores = torch.stack([self.edit_distance_dp(span, y, char_distr) for y in vocab], dim=0)
                p_span = torch.logsumexp(token_scores, dim=0) - math.log(len(vocab))
                c_el = dp[i - l] + log_p_el[l] + p_span
                candidates.append(c_el)

                # Track best-match coverage for reporting.
                local_value = float(c_el.detach().cpu().item())
                current_value = float(candidates[0].detach().cpu().item())
                if local_value >= current_value:
                    best_cov_i = best_cov[i - l] + l

            dp[i] = torch.logsumexp(torch.stack(candidates, dim=0), dim=0)
            best_cov[i] = min(float(n), best_cov_i)

        quality = dp[n]
        coverage_ratio = best_cov[n] / max(1.0, float(n))
        return quality, coverage_ratio

    def omega_loss(self, char_distr: Tensor) -> Tensor:
        """Eq.10 sound-loss regularizer discouraging collapsed inventories."""
        # Sum of probability mass each lost sound receives from known sounds.
        mass = char_distr.sum(dim=0)
        target = torch.full_like(mass, fill_value=mass.mean().detach())
        return ((mass - target) ** 2).mean()

    def objective(self, inscriptions: Sequence[str], known_vocab: Sequence[str]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        char_distr = self.compute_char_distr()
        qualities: List[Tensor] = []
        coverages: List[float] = []
        for x in inscriptions:
            q_x, cov_x = self.word_boundary_dp(x, known_vocab, char_distr)
            qualities.append(q_x)
            coverages.append(cov_x)

        quality = torch.stack(qualities, dim=0).mean() if qualities else torch.tensor(0.0, device=char_distr.device)
        omega_cov = torch.tensor(sum(coverages) / max(1, len(coverages)), device=char_distr.device)
        omega_loss = self.omega_loss(char_distr)

        s = quality + self.config.lambda_cov * omega_cov - self.config.lambda_loss * omega_loss
        return s, quality, omega_cov, omega_loss


def ComputeCharDistr(model: PhoneticPriorModel) -> Tensor:
    return model.compute_char_distr()


def EditDistDP(model: PhoneticPriorModel, lost_token: str, known_token: str, char_distr: Tensor) -> Tensor:
    return model.edit_distance_dp(lost_token, known_token, char_distr)


def WordBoundaryDP(
    model: PhoneticPriorModel,
    inscription: str,
    known_vocab: Sequence[str],
    char_distr: Tensor,
) -> Tuple[Tensor, float]:
    return model.word_boundary_dp(inscription, known_vocab, char_distr)


def train_one_step(
    model: PhoneticPriorModel,
    optimizer: torch.optim.Optimizer,
    inscriptions: Sequence[str],
    known_vocab: Sequence[str],
) -> TrainStepOutput:
    """Algorithm 1 training step."""
    optimizer.zero_grad()

    # 1) ComputeCharDistr
    _ = ComputeCharDistr(model)

    # 2/3) EditDistDP + WordBoundaryDP folded into objective()
    s, quality, omega_cov, omega_loss = model.objective(inscriptions, known_vocab)

    # 4) SGD update
    loss = -s
    loss.backward()
    optimizer.step()

    return TrainStepOutput(
        objective=float(s.detach().cpu().item()),
        quality=float(quality.detach().cpu().item()),
        omega_cov=float(omega_cov.detach().cpu().item()),
        omega_loss=float(omega_loss.detach().cpu().item()),
        num_sequences=len(inscriptions),
    )
