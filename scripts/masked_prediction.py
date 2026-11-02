"""Pure-Python mask and ranking helpers for the V-JEPA synthetic fixture."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def token_partition(
    *,
    temporal_tokens: int,
    height_tokens: int,
    width_tokens: int,
    target_temporal: Iterable[int],
    target_rows: Iterable[int],
    target_cols: Iterable[int],
) -> tuple[list[int], list[int]]:
    """Return sorted (context_ids, target_ids) for a rectangular T/H/W mask.

    IDs use the official flattening order: t * H * W + row * W + col.
    Bounds are checked before any model call. Context contains every grid token
    outside the target set; target IDs appear only as mask-token positions.
    """
    dims = (temporal_tokens, height_tokens, width_tokens)
    if any(not isinstance(n, int) or n <= 0 for n in dims):
        raise ValueError("token grid dimensions must be positive integers")
    times = sorted(set(target_temporal))
    rows = sorted(set(target_rows))
    cols = sorted(set(target_cols))
    if not times or not rows or not cols:
        raise ValueError("target ranges must be non-empty")
    if times[0] < 0 or times[-1] >= temporal_tokens:
        raise ValueError("target temporal index outside token grid")
    if rows[0] < 0 or rows[-1] >= height_tokens:
        raise ValueError("target row outside token grid")
    if cols[0] < 0 or cols[-1] >= width_tokens:
        raise ValueError("target column outside token grid")
    total = temporal_tokens * height_tokens * width_tokens
    target = sorted(t * height_tokens * width_tokens + y * width_tokens + x
                    for t in times for y in rows for x in cols)
    target_set = set(target)
    context = [i for i in range(total) if i not in target_set]
    if len(context) + len(target) != total or target_set.intersection(context):
        raise AssertionError("context and target masks do not partition the token grid")
    return context, target


def mean_absolute_error(predicted: Sequence[Sequence[float]],
                        target: Sequence[Sequence[float]]) -> float:
    if not predicted or len(predicted) != len(target):
        raise ValueError("predicted and target token counts must match and be non-empty")
    width = len(predicted[0])
    if width == 0 or any(len(row) != width for row in predicted) or any(len(row) != width for row in target):
        raise ValueError("feature dimensions must be rectangular and non-empty")
    total = 0.0
    count = 0
    for pred_row, target_row in zip(predicted, target):
        for pred_value, target_value in zip(pred_row, target_row):
            if not math.isfinite(float(pred_value)) or not math.isfinite(float(target_value)):
                raise ValueError("features must be finite")
            total += abs(float(pred_value) - float(target_value))
            count += 1
    return total / count


def rank_candidates(predicted: Sequence[Sequence[float]],
                    candidate_targets: Mapping[str, Sequence[Sequence[float]]]
                    ) -> list[tuple[str, float]]:
    """Rank by masked target-space mean L1; lower score is better."""
    if not candidate_targets:
        raise ValueError("at least one candidate is required")
    # Python's sort is stable, so exact ties retain randomized presentation order.
    scored = [(name, mean_absolute_error(predicted, features), index)
              for index, (name, features) in enumerate(candidate_targets.items())]
    return [(name, score) for name, score, _ in sorted(scored, key=lambda item: item[1])]


def evaluate_ranking(
    ranking: Sequence[tuple[str, float]],
    correct_candidate_id: str,
    *,
    tie_tolerance: float = 1e-6,
    visible_context_ambiguous: bool = True,
) -> dict:
    """Join evaluator-only truth after scoring and explicitly preserve ties."""
    if not ranking or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("ranking must be non-empty and tie tolerance finite/non-negative")
    ids = [candidate_id for candidate_id, _ in ranking]
    if len(set(ids)) != len(ids) or correct_candidate_id not in ids:
        raise ValueError("candidate IDs must be unique and include the evaluator label")
    scores = [float(score) for _, score in ranking]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("ranking scores must be finite")
    best_score = scores[0]
    tied_best = [candidate_id for candidate_id, score in ranking
                 if float(score) - best_score <= tie_tolerance]
    correct_rank = ids.index(correct_candidate_id) + 1
    return {
        "correct_candidate_id": correct_candidate_id,
        "correct_rank": correct_rank,
        "tied_best_candidate_ids": tied_best,
        "best_is_tied": len(tied_best) > 1,
        "visible_context_ambiguous": bool(visible_context_ambiguous),
        "rule_answer_is_unique_in_fixture": not visible_context_ambiguous,
        "top1_success": len(tied_best) == 1 and tied_best[0] == correct_candidate_id,
    }


def evaluate_ranking_metrics(
    scores: Mapping[str, float],
    correct_candidate_id: str,
    *,
    tie_tolerance: float = 1e-6,
) -> dict:
    """Report tie-aware expected rank, reciprocal rank, and loss margin.

    Ties are treated as exchangeable rather than broken by candidate ID or input
    order. A correct candidate tied at the top receives expected top-1 credit
    1/k, where k is the size of the tied group.
    """
    if (not scores or not math.isfinite(tie_tolerance) or tie_tolerance < 0
            or len(scores) < 2 or len(set(scores)) != len(scores)):
        raise ValueError("scores need unique candidate IDs and a finite non-negative tie tolerance")
    normalized = {str(candidate_id): float(score) for candidate_id, score in scores.items()}
    if any(not candidate_id or not math.isfinite(score)
           for candidate_id, score in normalized.items()):
        raise ValueError("candidate IDs and scores must be non-empty and finite")
    if correct_candidate_id not in normalized:
        raise ValueError("correct candidate must be present in scores")

    ordered = sorted(normalized.items(), key=lambda item: item[1])
    correct_score = normalized[correct_candidate_id]
    group_start = group_end = None
    index = 0
    while index < len(ordered):
        group_begin = index
        group_end_index = group_begin
        while (group_end_index + 1 < len(ordered)
               and ordered[group_end_index + 1][1] - ordered[group_begin][1] <= tie_tolerance):
            group_end_index += 1
        group_ids = [candidate_id for candidate_id, _ in ordered[group_begin:group_end_index + 1]]
        if correct_candidate_id in group_ids:
            group_start, group_end = group_begin + 1, group_end_index + 1
            break
        index = group_end_index + 1
    assert group_start is not None and group_end is not None
    tied_count = group_end - group_start + 1
    expected_rank = (group_start + group_end) / 2.0
    expected_mrr = sum(1.0 / rank for rank in range(group_start, group_end + 1)) / tied_count
    wrong_scores = [score for candidate_id, score in normalized.items()
                    if candidate_id != correct_candidate_id]
    margin = min(wrong_scores) - correct_score
    best_group = [candidate_id for candidate_id, score in ordered
                  if score - ordered[0][1] <= tie_tolerance]
    expected_top1 = (1.0 / len(best_group)
                     if group_start == 1 and correct_candidate_id in best_group else 0.0)
    return {
        "correct_candidate_id": correct_candidate_id,
        "correct_rank_min": group_start,
        "correct_rank_max": group_end,
        "correct_rank_expected": expected_rank,
        "top1_strict": len(best_group) == 1 and best_group[0] == correct_candidate_id,
        "top1_expected": expected_top1,
        "mrr_expected": expected_mrr,
        "tied_best_candidate_ids": best_group,
        "tie_count": len(best_group),
        "margin_min_wrong_minus_correct": margin,
    }


def require_supported_loss_contract(loss_exp: float, target_normalization: str) -> None:
    """Fail closed unless the scorer matches the official target+L1 loss form."""
    if not math.isfinite(float(loss_exp)) or float(loss_exp) != 1.0:
        raise ValueError("only official loss_exp=1.0 is supported by the masked-target MAE scorer")
    if target_normalization != "layer_norm_last_dim":
        raise ValueError("scoring requires the official per-token target layer normalization")


def token_ids_for_frame_range(start_frame: int, end_frame: int, *, tubelet_size: int,
                              height_tokens: int, width_tokens: int) -> list[int]:
    """Return all temporal token IDs intersecting a closed frame range.

    The caller must ensure the answer region does not split a tubelet; otherwise
    visible answer pixels would share a target token and the run must fail closed.
    """
    if start_frame < 0 or end_frame < start_frame or tubelet_size <= 0:
        raise ValueError("invalid frame range or tubelet size")
    if start_frame % tubelet_size or (end_frame + 1) % tubelet_size:
        raise ValueError("target frame range must align exactly to tubelet boundaries")
    first = start_frame // tubelet_size
    last = end_frame // tubelet_size
    return [t * height_tokens * width_tokens + y * width_tokens + x
            for t in range(first, last + 1)
            for y in range(height_tokens) for x in range(width_tokens)]
