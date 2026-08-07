"""Drive a corpus through the pipeline, one document at a time.

The output and stats files are rewritten after every document, so an interrupted
run costs at most the document in progress.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from .config import DOMAIN_TAG, MODE_SPECS
from .dataio import (load_documents, load_terminology, output_stem, write_json,
                     write_outputs)
from .terminology import TerminologyDatabase
from .textproc import newline_signature, split_sentences


def run_mode(cfg, mode: str, documents, names, kind, terminology, ctx, app):
    """One condition over the whole corpus."""
    from transformers import set_seed

    out_dir = Path(cfg.out_dir)
    term_key = MODE_SPECS[mode]["terms"]
    if term_key:
        terms = (terminology or {}).get(term_key, {})
        if not terms:
            raise ValueError(
                f"mode '{mode}' needs the '{term_key}' glossary but it is empty. "
                f"Supply --terms with a '{term_key}' section, or run mode 'noterm'.")
        ctx["term_db"] = TerminologyDatabase(terms, cfg.term_max_gap)
        print(f"mode='{mode}': terminology '{term_key}' ({len(terms)} entries), "
              f"weights='{MODE_SPECS[mode]['weights']}', "
              f"terminology agent={'on' if cfg.term_agent else 'off'}")
    else:
        ctx["term_db"] = None
        print(f"mode='{mode}': no terminology anywhere in the pipeline, "
              f"weights='{MODE_SPECS[mode]['weights']}'")

    tag = f".trial{len(documents)}" if cfg.limit_docs is not None else ""
    stem = output_stem(mode, cfg.system_name, cfg.domain, cfg.direction, tag)
    stats_file = out_dir / f"{stem}.stats.json"
    memory_file = out_dir / f"{stem}.memory.json"

    outputs, stats, memories = [], [], []
    t_start = time.perf_counter()

    for di, text in enumerate(documents):
        set_seed(cfg.seed + di)
        print(f"\n{'='*66}\nDOCUMENT {di+1}/{len(documents)}  ({len(text)} chars, "
              f"{len(newline_signature(text))+1} paragraphs)  |  mode={mode}  "
              f"domain={cfg.domain}\n{'='*66}")
        t0 = time.perf_counter()

        from .pipeline import translate_document
        mem, timings = translate_document(app, di, text, mode, cfg.domain)
        outputs.append(mem.translation)

        issues = Counter(i["type"] for v in mem.diagnostics.values() for i in v)
        agent_added = sum(len(v.get("added", [])) for v in mem.agent_notes.values())
        agent_dropped = sum(len(v.get("dropped", [])) for v in mem.agent_notes.values())
        stats.append({
            "doc": di,
            "name": names[di] if names else None,
            "chunks": len(mem.chunks),
            "sentences": sum(len(split_sentences(c["src"])) for c in mem.chunks),
            "terms_candidates": len({k for d in mem.candidates.values() for k in d}),
            "terms_confirmed": len(mem.all_terms()),
            "terms_recovered": agent_added,
            "terms_pruned": agent_dropped,
            "short_paragraphs_merged": sum(len(c["leads"]) for c in mem.chunks),
            "chunks_edited": len(mem.edited),
            "chunks_reformatted": len(mem.formatted),
            "diagnostics": dict(issues),
            "consistent": mem.consistency["ok"],
            "seconds": round(time.perf_counter() - t0, 1),
            "timings": timings,
        })
        if cfg.dump_memory:
            memories.append(mem.to_dict())
            write_json(memory_file, memories)

        out_file = write_outputs(out_dir, stem, outputs, names, kind)
        write_json(stats_file, stats)

        s = stats[-1]
        print(f"  document {di+1} done in {s['seconds']}s | {s['chunks']} chunks, "
              f"{s['terms_confirmed']} terms "
              f"(+{s['terms_recovered']}/-{s['terms_pruned']}), "
              f"{s['chunks_edited']} edited | "
              f"diagnostics: {s['diagnostics'] or 'clean'} | "
              f"saved {len(outputs)}/{len(documents)} -> {out_file.name}")
        mem.reset()

    assert len(outputs) == len(documents)
    bad = [i for i, (a, b) in enumerate(zip(documents, outputs))
           if newline_signature(a) != newline_signature(b)]
    if bad:
        print(f"*** WARNING: line-break structure differs in documents {bad}")

    totals = Counter()
    for s in stats:
        totals.update(s["diagnostics"])
    print(f"\nMODE {mode} complete in {time.perf_counter()-t_start:.0f}s -> {out_file}")
    print(f"  chunks: {sum(s['chunks'] for s in stats)}, "
          f"edited: {sum(s['chunks_edited'] for s in stats)}, "
          f"terms: {sum(s['terms_confirmed'] for s in stats)}, "
          f"diagnostics: {dict(totals) or 'clean'}")
    return {"mode": mode, "output": str(out_file), "stats": stats}


def run(cfg):
    """Load everything once, then run each requested mode."""
    from .models import ModelPool, resolve_budgets
    from .pipeline import build_graph

    profile = resolve_budgets(cfg)
    print(f"profile={profile} | 4-bit={cfg.load_in_4bit} | "
          f"resident 8B models={cfg.max_resident_large}")
    if cfg.domain not in DOMAIN_TAG:
        print(f"*** WARNING: domain '{cfg.domain}' is not one of {sorted(DOMAIN_TAG)}; "
              f"the fine-tuned domain conditioning will not fire.")

    documents, names, kind = load_documents(cfg.text_path, cfg.verbose)
    if cfg.limit_docs is not None:
        documents = documents[:cfg.limit_docs]
        names = names[:cfg.limit_docs] if names else None
        print(f"*** TRIAL: first {len(documents)} document(s) — NOT a valid submission.")

    terminology = load_terminology(cfg.terms_path, cfg.verbose) if cfg.terms_path else None
    needs_terms = any(MODE_SPECS[m]["terms"] for m in cfg.modes)
    if needs_terms and terminology is None:
        raise ValueError("modes 'proper'/'random' need --terms; "
                         "use mode 'noterm' to run without terminology.")

    ctx = {"term_db": None}
    ctx["pool"] = ModelPool(cfg)
    app = build_graph(cfg, ctx)

    # Fine-tuned weights first, so the pool reloads as little as possible.
    order = sorted(cfg.modes, key=lambda m: (MODE_SPECS[m]["weights"] != "ft", m))
    results = []
    try:
        for m in order:
            results.append(run_mode(cfg, m, documents, names, kind, terminology,
                                    ctx, app))
    finally:
        ctx["pool"].clear()
    return results
