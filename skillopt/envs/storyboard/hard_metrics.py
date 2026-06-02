"""Storyboard hard metrics — automated structural evaluation.

Three metrics:
1. shot_count_deviation: |AI shots - GT shots| / GT shots
2. cut_alignment_rate: proportion of AI cut-points matching GT cut-points (±1 sentence tolerance)
3. shot_scale_edit_distance: normalized Levenshtein distance of shot-scale sequences
"""
from __future__ import annotations

import csv
import io
import re


# ---------------------------------------------------------------------------
# Shot-scale vocabulary normalization
# ---------------------------------------------------------------------------

_SCALE_CANONICAL = {
    "ws": "WS", "wide": "WS", "全景": "WS",
    "fs": "FS", "full": "FS", "远景": "FS",
    "mfs": "MFS", "medium full": "MFS", "中全景": "MFS",
    "ms": "MS", "medium": "MS", "中景": "MS",
    "mcu": "MCU", "medium close": "MCU", "中近景": "MCU",
    "cu": "CU", "close": "CU", "closeup": "CU", "close-up": "CU",
    "近景": "CU", "特写": "CU",
    "ecu": "ECU", "extreme close": "ECU", "大特写": "ECU",
    "pov": "POV",
    "ots": "OTS", "over the shoulder": "OTS", "过肩": "OTS",
}


def _normalize_scale(raw: str) -> str:
    """Normalize a shot-scale string to canonical form."""
    cleaned = raw.strip().lower()
    cleaned = re.sub(r"[（(].*?[)）]", "", cleaned)
    cleaned = re.sub(r"(双人|三人|多人|关系)", "", cleaned)
    cleaned = cleaned.strip()
    if cleaned in _SCALE_CANONICAL:
        return _SCALE_CANONICAL[cleaned]
    for key, val in _SCALE_CANONICAL.items():
        if key in cleaned:
            return val
    return cleaned.upper() or "UNK"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_csv_rows(csv_text: str) -> list[dict] | None:
    """Parse CSV text into list of row dicts."""
    cleaned = csv_text.strip()
    fence_match = re.search(r"```(?:csv)?\s*\n(.*?)\n\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    if not cleaned:
        return None
    try:
        reader = csv.DictReader(io.StringIO(cleaned))
        rows = list(reader)
        return rows if rows else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

def _segment_sentences(script_text: str) -> list[str]:
    """Split script into sentence-level units for cut-point alignment."""
    lines = script_text.strip().split("\n")
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[。！？.!?])\s*|(?<=\n)", line)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def _find_sentence_index(sentences: list[str], text: str) -> int:
    """Find which sentence best matches a text snippet. Returns -1 if no match."""
    if not text.strip():
        return -1
    text_lower = text.strip().lower()
    best_idx = -1
    best_overlap = 0
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        overlap = len(set(text_lower) & set(sent_lower))
        if text_lower in sent_lower or sent_lower in text_lower:
            overlap = max(overlap, len(text_lower))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
    if best_overlap < 3:
        return -1
    return best_idx


def _extract_cut_points(rows: list[dict], sentences: list[str]) -> list[int]:
    """Map each shot to its starting sentence index to get cut-point sequence."""
    cut_points = []
    content_key = None
    dialogue_key = None
    for key in rows[0]:
        if "画面" in key or "content" in key.lower():
            content_key = key
        if "台词" in key or "dialogue" in key.lower() or "台詞" in key:
            dialogue_key = key

    last_idx = -1
    for row in rows:
        dialogue = (row.get(dialogue_key, "") or "").strip() if dialogue_key else ""
        content = (row.get(content_key, "") or "").strip() if content_key else ""

        idx = -1
        if dialogue:
            idx = _find_sentence_index(sentences, dialogue)
        if idx == -1 and content:
            idx = _find_sentence_index(sentences, content)
        if idx == -1:
            idx = last_idx + 1 if last_idx >= 0 else 0

        cut_points.append(idx)
        last_idx = idx

    return cut_points


# ---------------------------------------------------------------------------
# Levenshtein distance (pure Python, no deps)
# ---------------------------------------------------------------------------

def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hard_metrics(
    script_text: str,
    generated_csv: str,
    ground_truth_csv: str,
) -> dict:
    """Compute three automated hard metrics.

    Returns dict with:
        shot_count_deviation: float (0 = perfect, higher = worse)
        cut_alignment_rate: float (0-1, higher = better)
        shot_scale_edit_dist: float (0-1 normalized, 0 = identical)
        hard_score: float (0-1 composite, higher = better)
    """
    gen_rows = _parse_csv_rows(generated_csv)
    gt_rows = _parse_csv_rows(ground_truth_csv)

    if gen_rows is None:
        return {
            "shot_count_deviation": 1.0,
            "cut_alignment_rate": 0.0,
            "shot_scale_edit_dist": 1.0,
            "hard_score": 0.0,
            "parse_ok": False,
        }

    if gt_rows is None:
        return {
            "shot_count_deviation": 0.0,
            "cut_alignment_rate": 0.0,
            "shot_scale_edit_dist": 0.0,
            "hard_score": 0.0,
            "parse_ok": False,
        }

    # --- Metric 1: Shot count deviation ---
    n_gen = len(gen_rows)
    n_gt = len(gt_rows)
    shot_count_dev = abs(n_gen - n_gt) / max(n_gt, 1)

    # --- Metric 2: Cut-point alignment rate ---
    sentences = _segment_sentences(script_text)
    if sentences:
        gen_cuts = _extract_cut_points(gen_rows, sentences)
        gt_cuts = _extract_cut_points(gt_rows, sentences)

        # Count how many GT cut-points have a matching gen cut-point within ±1
        matched = 0
        gen_cut_set = set(gen_cuts)
        for gt_cut in gt_cuts:
            if any(c in gen_cut_set for c in range(gt_cut - 1, gt_cut + 2)):
                matched += 1
        cut_alignment = matched / max(len(gt_cuts), 1)
    else:
        cut_alignment = 0.0

    # --- Metric 3: Shot-scale edit distance ---
    scale_key = None
    for key in gen_rows[0]:
        if "景别" in key or "scale" in key.lower() or "shot" in key.lower():
            scale_key = key
            break
    gt_scale_key = None
    for key in gt_rows[0]:
        if "景别" in key or "scale" in key.lower() or "shot" in key.lower():
            gt_scale_key = key
            break

    if scale_key and gt_scale_key:
        gen_scales = [_normalize_scale(row.get(scale_key, "")) for row in gen_rows]
        gt_scales = [_normalize_scale(row.get(gt_scale_key, "")) for row in gt_rows]
        edit_dist = _levenshtein(gen_scales, gt_scales)
        max_len = max(len(gen_scales), len(gt_scales), 1)
        norm_edit_dist = edit_dist / max_len
    else:
        norm_edit_dist = 1.0

    # --- Composite hard score ---
    # shot_count_dev: lower is better, cap at 1.0
    count_score = max(0.0, 1.0 - min(shot_count_dev, 1.0))
    # cut_alignment: already 0-1, higher is better
    # norm_edit_dist: lower is better
    scale_score = max(0.0, 1.0 - min(norm_edit_dist, 1.0))

    # Weighted composite: count 30%, alignment 40%, scale 30%
    hard_score = 0.3 * count_score + 0.4 * cut_alignment + 0.3 * scale_score

    return {
        "shot_count_deviation": round(shot_count_dev, 4),
        "cut_alignment_rate": round(cut_alignment, 4),
        "shot_scale_edit_dist": round(norm_edit_dist, 4),
        "hard_score": round(hard_score, 4),
        "parse_ok": True,
    }
