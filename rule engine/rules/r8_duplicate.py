"""
R8 — Duplicate / Overlapping Project
----------------------------------------
Detects: projects that appear to describe the same or highly similar
work (potential duplicate allocation / repeated funding).

Method:
    1. Build a text field per project = work_title + work_description.
    2. TF-IDF vectorize all project texts, compute pairwise cosine
       similarity (this is the "embeddings + cosine similarity" idea
       from the design doc — TF-IDF is a lightweight, dependency-light
       stand-in; swap in sentence-transformer embeddings later if you
       want semantic rather than lexical similarity).
    3. For each project, find its single most-similar OTHER project.
    4. If that other project is also in the same constituency/area,
       add a same-area bonus (duplicates in the same place are more
       suspicious than similar work in two different villages).

Scoring:
    best_similarity (0-1) -> 0-100, boosted by same-area bonus, capped at 100.
    Below R8_SIMILARITY_HARD_MIN, don't even bother — score 0, not triggered.

Output explicitly says "Potential duplicate/overlapping project",
never "confirmed fraud" — see evidence/reason fields.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import COLUMNS, THRESHOLDS
from utils.scoring import make_result

RULE_ID = "R8"

# PATCH — the ORIGINAL version below built ONE dense NxN cosine-similarity
# matrix for every project against every other project. On this dataset
# (41,086 projects) that matrix is 41,086 x 41,086 float64 = ~13.5 GB,
# which will OOM-crash almost any machine (this sandbox only has 3.9 GB
# total RAM). It also compares projects that share nothing in common
# (different state, different work type) which is wasted compute AND a
# source of false positives.
#
# Fix: BLOCK first. Only run TF-IDF/cosine similarity WITHIN small
# candidate groups that already share (constituency, work_title) — i.e.
# only compare projects that could plausibly be duplicates in the first
# place. This cuts the ~1.7 billion pairwise comparisons down to a few
# million, and keeps every similarity matrix small enough to hold in
# memory. Groups larger than MAX_BLOCK_SIZE are skipped rather than
# risking a large matrix again (very large blocks are usually generic
# high-volume work types like "Purchase of books" that need a smarter
# strategy, not brute-force comparison).

MAX_BLOCK_SIZE = 400


def run(projects: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS

    work_ids = projects[c["work_id"]].tolist()
    texts_series = (
        projects.get(c["work_title"], "").fillna("").astype(str) + " " +
        projects.get(c["work_description"], "").fillna("").astype(str)
    )
    areas_series = projects.get(c["constituency"], pd.Series([None] * len(projects)))

    if len(texts_series) < 2:
        for wid in work_ids:
            results.append(make_result(RULE_ID, wid, None, False,
                                        reason="Not enough text data to compare"))
        return results

    hard_min = THRESHOLDS["R8_SIMILARITY_HARD_MIN"]
    area_bonus = THRESHOLDS["R8_SAME_AREA_BONUS"]

    best_sim_map = {}      # work_id -> best similarity found so far
    best_partner_map = {}  # work_id -> most-similar other work_id
    best_text_map = {}     # work_id -> that partner's text snippet

    block_key_title = c["work_title"]
    block_key_area = c["constituency"]
    grouped = projects.groupby([block_key_area, block_key_title]).indices

    for (_area, _title), idx in grouped.items():
        if len(idx) < 2 or len(idx) > MAX_BLOCK_SIZE:
            continue  # nothing to compare, or block too large to risk

        block_work_ids = projects[c["work_id"]].values[idx]
        block_texts = texts_series.values[idx]

        if all(t.strip() == "" for t in block_texts):
            continue

        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf_matrix = vectorizer.fit_transform(block_texts)
        sim_matrix = cosine_similarity(tfidf_matrix)  # small: len(idx) x len(idx)

        n = len(idx)
        for i in range(n):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            best_j = sims.argmax()
            best_sim = sims[best_j]
            wid = block_work_ids[i]
            if best_sim > best_sim_map.get(wid, -1):
                best_sim_map[wid] = best_sim
                best_partner_map[wid] = block_work_ids[best_j]
                best_text_map[wid] = block_texts[best_j][:80]

    areas_by_id = dict(zip(work_ids, areas_series.tolist()))

    for work_id in work_ids:
        best_sim = best_sim_map.get(work_id)

        if best_sim is None or best_sim < hard_min:
            results.append(make_result(
                RULE_ID, work_id, 0.0, False,
                indicator={"best_similarity": round(float(best_sim), 3) if best_sim is not None else None},
                reason="No sufficiently similar project found (within same constituency + work_title)",
            ))
            continue

        partner = best_partner_map[work_id]
        # blocking already guarantees same constituency + work_title, so the
        # area bonus always applies here (kept for compatibility/clarity)
        same_area = True
        adjusted_sim = min(1.0, best_sim + (area_bonus if same_area else 0))
        score = round(adjusted_sim * 100, 2)
        triggered = score >= 70

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"best_similarity": round(float(best_sim), 3), "same_area": same_area},
            evidence={"most_similar_work_id": partner,
                      "most_similar_title": best_text_map[work_id]},
            reason=(f"Potential duplicate/overlapping project — {best_sim:.0%} text "
                    f"similarity with work_id {partner} in the same area"
                    if triggered else "Similar text found but below duplicate threshold"),
        ))

    return results
