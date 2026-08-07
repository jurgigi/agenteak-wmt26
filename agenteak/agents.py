"""Terminology agent: parsing and reconciliation.

Kept separate from pipeline.py so that it imports neither torch nor langgraph
and can be unit-tested on its own — these two functions decide which
terminology reaches the translator and the editor, so a silent regression here
would corrupt a whole submission without raising.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .textproc import norm

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_term_agent_reply(answer: str) -> Optional[list]:
    """-> [{"es":..., "eu":...}] or None if the reply is unusable.

    None is meaningful: the caller keeps the deterministic candidate list, so a
    malformed reply degrades the run to the matcher-only baseline rather than
    losing terminology altogether.
    """
    if not (answer or "").strip():
        return None
    text = re.sub(r"^```(?:json)?|```$", "", answer.strip(), flags=re.M).strip()
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    terms = data.get("terms") if isinstance(data, dict) else None
    if not isinstance(terms, list):
        return None
    out = []
    for item in terms:
        if isinstance(item, dict) and item.get("es") and item.get("eu"):
            out.append({"es": str(item["es"]).strip(),
                        "eu": str(item["eu"]).strip()})
    return out


def reconcile(selected: list, candidates: dict, possible: dict) -> tuple:
    """Keep only entries the agent was actually offered, and only targets that
    are approved. The agent may prune, recover and disambiguate; it may not
    invent terminology, and it may not invent a Basque target.

    -> (terms, notes)
    """
    offered = {}
    offered.update(candidates or {})
    offered.update(possible or {})
    by_norm = {norm(k): k for k in offered}

    terms, added, dropped, rejected = {}, [], [], []
    for item in selected:
        key = by_norm.get(norm(item["es"]))
        if key is None:
            rejected.append(item["es"])              # not in the glossary
            continue
        approved = offered[key]
        chosen = next((t for t in approved if norm(t) == norm(item["eu"])), None)
        if chosen is None:
            chosen = approved[0]                     # unapproved target
            rejected.append(f'{key} -> {item["eu"]}')
        terms[key] = [chosen]
        if key not in (candidates or {}):
            added.append(key)
    for key in (candidates or {}):
        if key not in terms:
            dropped.append(key)
    return terms, {"added": added, "dropped": dropped, "rejected": rejected}
