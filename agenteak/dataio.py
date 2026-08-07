"""Reading source documents and glossaries, and writing results.

Two input shapes are supported.

1. SHARED-TASK JSON (what the WMT terminology task ships, and what a submission
   must be written back as):

       text.<domain>.<direction>.json   ->  ["doc one...", "doc two...", ...]
       terms.<domain>.<direction>.json  ->  {"proper": {es: [eu, ...]},
                                             "random": {es: [eu, ...]}}

   Output is a JSON list of translations, one per input document, in the same
   order and with the same paragraph structure.

2. PLAIN TEXT, for using the pipeline on your own material:

       --input article.txt        one file  = one document
       --input texts/             a directory; every *.txt = one document,
                                  sorted by filename
       --glossary terms.tsv       optional, "es<TAB>eu" per line; several
                                  targets separated by "|" or ";"
       --glossary terms.json      optional, {es: eu} or {es: [eu, ...]}

   Output is written as .txt files (one per input document) alongside the JSON.

Plain-text input goes through exactly the same graph: the pipeline works on
paragraphs and never sees the container format.

No model dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Source documents
# ---------------------------------------------------------------------------


def load_documents(path: str, verbose: int = 1) -> tuple:
    """-> (documents, names, kind).

    names is used to build per-document .txt filenames in text mode; it is None
    for shared-task JSON, where output order is all that matters.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input not found: {p}")

    if p.is_dir():
        files = sorted(f for f in p.iterdir()
                       if f.suffix.lower() in (".txt", ".md"))
        if not files:
            raise ValueError(f"no .txt files in {p}")
        docs = [f.read_text(encoding="utf-8").strip("\n") for f in files]
        if verbose:
            print(f"Loaded {len(docs)} text document(s) from {p}/")
        return docs, [f.stem for f in files], "text"

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("shared-task text file must be a JSON list of strings")
        if verbose:
            print(f"Loaded {len(data)} source document(s) from {p}")
        return data, None, "json"

    text = p.read_text(encoding="utf-8").strip("\n")
    if verbose:
        print(f"Loaded 1 text document from {p}")
    return [text], [p.stem], "text"


# ---------------------------------------------------------------------------
# Terminology
# ---------------------------------------------------------------------------


def _as_targets(value) -> list:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    parts = [s.strip() for s in str(value).replace(";", "|").split("|")]
    return [s for s in parts if s]


def load_terminology(path: str, verbose: int = 1) -> dict:
    """-> {"proper": {...}, "random": {...}}.

    Accepts the shared-task JSON (which carries both conditions), a flat JSON
    dict, or a TSV. A flat glossary is treated as the "proper" condition, and
    "random" is left empty — running mode=random then has nothing to inject,
    which the runner reports rather than silently ignoring.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"terminology not found: {p}")

    if p.suffix.lower() in (".tsv", ".txt", ".csv"):
        sep = "," if p.suffix.lower() == ".csv" else "\t"
        flat = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if sep not in line:
                continue
            src, tgt = line.split(sep, 1)
            if src.strip():
                flat[src.strip()] = _as_targets(tgt)
        out = {"proper": flat, "random": {}}
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and ("proper" in data or "random" in data):
            out = {k: {s: _as_targets(t) for s, t in (data.get(k) or {}).items()}
                   for k in ("proper", "random")}
        elif isinstance(data, dict):
            out = {"proper": {s: _as_targets(t) for s, t in data.items()},
                   "random": {}}
        else:
            raise ValueError("glossary JSON must be an object")

    if verbose:
        print(f"Loaded terminology from {p}: "
              f"proper={len(out['proper'])}, random={len(out['random'])}")
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def output_stem(mode: str, system_name: str, domain: str, direction: str,
                tag: str = "") -> str:
    """base_noterm is a contrastive run of our own, not a task condition: it is
    submitted (if at all) as a separate system under the noterm condition, so
    the condition in the filename stays one of the three the task defines."""
    if mode == "base_noterm":
        return f"{system_name}-base.noterm.{domain}.{direction}{tag}"
    return f"{system_name}.{mode}.{domain}.{direction}{tag}"


def write_outputs(out_dir, stem: str, outputs: list, names=None,
                  kind: str = "json") -> Path:
    """Always writes the JSON list (this is the submission artefact). In text
    mode, additionally writes one .txt per document."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{stem}.json"
    out_file.write_text(json.dumps(outputs, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    if kind == "text" and names:
        txt_dir = out_dir / f"{stem}.txt"
        txt_dir.mkdir(parents=True, exist_ok=True)
        for name, text in zip(names, outputs):
            (txt_dir / f"{name}.eu.txt").write_text(text, encoding="utf-8")
    return out_file


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")
