"""
R9 — Completed Without Evidence
----------------------------------
Detects: projects marked COMPLETED with no completion evidence
(e.g. a project image) on file.

Logic:
    status == COMPLETED AND evidence missing -> flag

Scoring (binary, as specified):
    Evidence available -> 0
    Evidence missing    -> 100
    Not COMPLETED       -> None (rule not applicable)

Important: a missing image does NOT mean the work doesn't exist.
It means "completion evidence is unavailable — verify."
"""

import pandas as pd
from config import COLUMNS
from utils.scoring import make_result

RULE_ID = "R9"


def _is_evidence_present(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() not in ("", "0", "false", "False", "NaN", "nan")
    return bool(value)


def run(projects: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        status = str(row.get(c["status"], "")).upper()

        if status != "COMPLETED":
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Not applicable — project not COMPLETED"))
            continue

        has_evidence = _is_evidence_present(row.get(c["has_image"]))
        score = 0.0 if has_evidence else 100.0
        triggered = not has_evidence

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"evidence_present": has_evidence},
            reason=("Completion evidence is unavailable; verify" if triggered
                    else "Completion evidence present"),
        ))

    return results
