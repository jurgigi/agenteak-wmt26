"""The Agenteak graph.

    document
      |
      +-- prepare    chunk into <=3-sentence units
      +-- match      DETERMINISTIC candidate terminology for each chunk
      +-- select     TERMINOLOGY AGENT (Qwen3-4B): prune, recover, disambiguate
      +-- translate  TRANSLATOR (fine-tuned Latxa-8B), chunk by chunk
      +-- verify     EDITOR (Latxa-8B): check terminology, repair declension
      +-- format     deterministic formatting repair
      +-- finalize   reconstruct the document, run report-only diagnostics
      v

Each stage writes its layer into a per-document store that the next stage reads.
The store is created empty for each source text and discarded when that text is
finished, so nothing survives from one document to the next.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from .agents import parse_term_agent_reply, reconcile
from .config import MODE_SPECS
from .diagnostics import consistency_report, diagnose_chunk
from .prompts import (FMT_SYSTEM, editor_system_prompt, editor_user_prompt,
                      term_agent_messages, translator_system_prompt)
from .textproc import (FMT_CLOSE, FMT_OPEN, TAG_RE, absolutive, between,
                       clean_output, enforce_structure, format_diff, is_symbolic,
                       looks_like_leak, match_case, norm, rebuild, repair_format,
                       same_words, source_carryover, split_document,
                       split_sentences, untranslated, words)


# ===========================================================================
# Per-document store
# ===========================================================================


class DocMemory:
    """chunks -> candidates -> terms -> drafts -> finals, keyed by chunk id."""

    def __init__(self, idx: int, source: str):
        self.idx = idx
        self.source = source
        self.reset()

    def reset(self):
        self.parts = []
        self.chunks = []          # [{cid, part, seq, src, leads}]
        self.candidates = {}      # cid -> {es -> [eu, ...]}   (match)
        self.terms = {}           # cid -> {es -> [eu]}        (select)
        self.spans = {}           # cid -> {es -> span}        (match, debug)
        self.agent_notes = {}     # cid -> {"added": [], "dropped": [], "raw": ""}
        self.drafts = {}          # cid -> str                 (translate)
        self.finals = {}          # cid -> str                 (verify/format)
        self.edited = set()
        self.formatted = set()
        self.short = {}           # cid -> which tier resolved a heading
        self.diagnostics = {}
        self.consistency = {"ok": True, "inconsistent": {}}
        self.translation = ""

    # --- writers ----------------------------------------------------------

    def set_structure(self, parts, chunks):
        self.parts, self.chunks = parts, chunks

    def set_candidates(self, cid, terms, spans=None):
        self.candidates[cid] = {k: list(v) for k, v in (terms or {}).items()}
        self.terms.setdefault(cid, dict(self.candidates[cid]))
        if spans:
            self.spans[cid] = dict(spans)

    def set_terms(self, cid, terms, notes=None):
        self.terms[cid] = {k: (list(v) if isinstance(v, list) else [v])
                           for k, v in (terms or {}).items()}
        if notes:
            self.agent_notes[cid] = notes

    def set_draft(self, cid, text):
        self.drafts[cid] = text
        self.finals.setdefault(cid, text)

    def set_final(self, cid, text, edited=False):
        self.finals[cid] = text
        if edited:
            self.edited.add(cid)

    # --- readers ----------------------------------------------------------

    def terms_of(self, cid):
        return self.terms.get(cid, {})

    def all_terms(self):
        out = {}
        for d in self.terms.values():
            for k, v in d.items():
                out.setdefault(k, list(v))
        return out

    def to_dict(self):
        return {"doc": self.idx,
                "chunks": [{"cid": c["cid"], "part": c["part"], "seq": c["seq"],
                            "leads": [l["text"] for l in c["leads"]],
                            "src": c["src"],
                            "candidates": self.candidates.get(c["cid"], {}),
                            "terms": self.terms.get(c["cid"], {}),
                            "agent": self.agent_notes.get(c["cid"], {}),
                            "spans": self.spans.get(c["cid"], {}),
                            "draft": self.drafts.get(c["cid"], ""),
                            "final": self.finals.get(c["cid"], ""),
                            "edited": c["cid"] in self.edited,
                            "short": self.short.get(c["cid"], ""),
                            "issues": self.diagnostics.get(c["cid"], [])}
                           for c in self.chunks],
                "consistency": self.consistency}


class DocState(TypedDict):
    memory: object
    mode: str
    domain: str
    timings: Optional[list]


# ===========================================================================
# Stage 1 — prepare
# ===========================================================================


def make_prepare(cfg):
    def prepare_node(state: DocState) -> DocState:
        mem = state["memory"]
        parts, chunks = split_document(
            mem.source, cfg.max_sentences, cfg.max_chunk_chars,
            cfg.merge_short_paragraphs, cfg.short_paragraph_words)
        mem.set_structure(parts, chunks)
        return state
    return prepare_node


# ===========================================================================
# Stage 2 — match (deterministic)
# ===========================================================================


def make_match(cfg, ctx):
    def match_node(state: DocState) -> DocState:
        mem = state["memory"]
        db = ctx.get("term_db")
        if db is None:
            for c in mem.chunks:
                mem.set_candidates(c["cid"], {})
            return state
        for c in mem.chunks:
            terms = db.match(c["src"])
            spans = (db.match_spans(c["src"])
                     if cfg.dump_memory or cfg.verbose >= 2 else None)
            mem.set_candidates(c["cid"], terms, spans)
            if cfg.verbose >= 2 and terms:
                print(f"          [{c['cid']}] " + ", ".join(
                    f"{k} = {(spans or {}).get(k, '?')}" for k in terms))
        return state
    return match_node


# ===========================================================================
# Stage 3 — select (TERMINOLOGY AGENT)
# ===========================================================================

def make_select(cfg, ctx):
    def select_node(state: DocState) -> DocState:
        """Prune the matcher's false positives, recover entries it missed, and
        choose one target where the glossary offers several.

        Skipped entirely when there is no terminology (noterm / base_noterm) or
        when cfg.term_agent is False, in which case the matcher's output stands.
        """
        mem = state["memory"]
        db = ctx.get("term_db")
        if db is None or not cfg.term_agent:
            return state

        agent = ctx["pool"].agent_for("term_agent", state["mode"])
        n_calls = 0
        for n, c in enumerate(mem.chunks, 1):
            cid = c["cid"]
            candidates = mem.candidates.get(cid, {})
            possible = db.near_misses(c["src"], candidates,
                                      cfg.term_agent_recall_candidates)
            if not candidates and not possible:
                mem.set_terms(cid, {})
                continue
            if cfg.verbose:
                print(f"\r        selecting terminology, chunk {n}/{len(mem.chunks)} ...",
                      end="", flush=True)
            messages = term_agent_messages(c["src"], candidates, possible,
                                           few_shot=True)
            _, answer = agent.ask(messages, thinking=cfg.term_agent_thinking,
                                  think_budget=cfg.think_budget,
                                  answer_budget=min(cfg.answer_budget, 512))
            n_calls += 1
            selected = parse_term_agent_reply(answer)
            if selected is None:
                # Unusable reply: keep the deterministic candidates.
                mem.set_terms(cid, candidates,
                              {"added": [], "dropped": [], "rejected": [],
                               "fallback": "unparsed"})
                continue
            terms, notes = reconcile(selected, candidates, possible)
            mem.set_terms(cid, terms, notes)
            if cfg.verbose >= 2 and (notes["added"] or notes["dropped"]):
                print(f"\n          [{cid}] +{notes['added']} -{notes['dropped']}")
        if cfg.verbose:
            print(f"\r        terminology agent: {n_calls} call(s)" + " " * 20)
        return state
    return select_node


# ===========================================================================
# Stage 4 — translate
# ===========================================================================


def make_translate(cfg, ctx):
    heading_gloss_n = {norm(k): v for k, v in cfg.heading_gloss.items()}

    def resolve_short_paragraph(trans, system, source, terms, next_source):
        """A heading, caption or label -> (translation, tier).

        The translator was fine-tuned on sentence pairs; handed "Uso" on its own
        it returns "Uso". Four tiers, cheapest first: an approved term, a known
        section heading, a translation that sees the next paragraph as CONTEXT
        but keeps only the first line, and finally a standalone attempt.
        """
        n = norm(source).strip(" .:;")
        for src_term, targets in (terms or {}).items():
            if norm(src_term) == n and targets:
                target = absolutive(targets[0]) if cfg.heading_absolutive else targets[0]
                return match_case(source, target), "glossary"
        gloss = heading_gloss_n.get(n)
        if gloss:
            return match_case(source, gloss), "gloss"
        if next_source:
            ctxt = source + "\n" + next_source[:cfg.heading_context_chars]
            budget = min(cfg.answer_budget,
                         max(96, int(trans.n_tokens(ctxt) * 2.4) + 64))
            out = clean_output(trans.generate(
                [{"role": "system", "content": system},
                 {"role": "user", "content": ctxt}],
                max_new_tokens=budget, temperature=0.2), keep_lines=True)
            first = (out.split("\n")[0].strip() if "\n" in out
                     else (split_sentences(out) or [""])[0])
            if first and len(words(first)) <= len(words(source)) + 2 \
                    and not untranslated(source, first):
                return match_case(source, first), "context"
        for temp in (0.2, 0.7):
            budget = min(cfg.answer_budget,
                         max(64, int(trans.n_tokens(source) * 3) + 32))
            out = clean_output(trans.generate(
                [{"role": "system", "content": system},
                 {"role": "user", "content": source}],
                max_new_tokens=budget, temperature=temp))
            if out and not untranslated(source, out):
                return match_case(source, out), "standalone"
        return source, "unresolved"

    def translate_node(state: DocState) -> DocState:
        """The fine-tuned translator sees one source chunk and nothing else.

        No term block, no brief, no output contract: the fine-tune already
        encodes the task, and scaffolding it never saw at training time can only
        push it off-distribution.
        """
        mem = state["memory"]
        trans = ctx["pool"].agent_for("translator", state["mode"])
        system = translator_system_prompt(state["domain"])

        for n, c in enumerate(mem.chunks, 1):
            if cfg.verbose:
                print(f"\r        translating chunk {n}/{len(mem.chunks)} ...",
                      end="", flush=True)
            src, keep = c["src"], bool(c["leads"])

            if is_symbolic(src):
                mem.short[c["cid"]] = "symbolic"
                mem.set_draft(c["cid"], src)
                continue

            if len(words(src)) <= cfg.short_paragraph_words and not c["leads"]:
                nxt = mem.chunks[n]["src"] if n < len(mem.chunks) else ""
                draft, how = resolve_short_paragraph(
                    trans, system, src, mem.terms_of(c["cid"]), nxt)
                mem.short[c["cid"]] = how
                mem.set_draft(c["cid"], draft)
                continue

            budget = min(cfg.answer_budget, max(96, int(trans.n_tokens(src) * 2.4) + 64))
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": src}]
            draft = clean_output(trans.generate(messages, max_new_tokens=budget,
                                                temperature=0.2), keep_lines=keep)
            if not draft or looks_like_leak(draft, src):
                retry = clean_output(trans.generate(messages, max_new_tokens=budget,
                                                    temperature=0.5), keep_lines=keep)
                if retry and not looks_like_leak(retry, src):
                    draft = retry
                elif not draft:
                    draft = retry
            mem.set_draft(c["cid"], draft or src)
        if cfg.verbose:
            print()
        return state
    return translate_node


# ===========================================================================
# Stage 5 — verify
# ===========================================================================


def make_verify(cfg, ctx):
    def verify_node(state: DocState) -> DocState:
        """The editor is given the source chunk, the draft and the terminology
        selected for that chunk, in the prompt shape it was trained on. What it
        returns is the translation.

        The single exception is a hard failure — nothing generated, prompt
        scaffolding echoed back, or untranslated Spanish spliced in — where the
        draft stands, because those are not edits at all.
        """
        mem = state["memory"]
        mode = state["mode"]
        with_terms = MODE_SPECS[mode]["terms"] is not None
        editor = ctx["pool"].agent_for("editor", mode)
        system = editor_system_prompt(state["domain"], with_terms)

        for n, c in enumerate(mem.chunks, 1):
            cid = c["cid"]
            if cfg.verbose:
                print(f"\r        verifying chunk {n}/{len(mem.chunks)} ...",
                      end="", flush=True)
            src, draft = c["src"], mem.drafts.get(cid, "")
            if not cfg.verify_short_paragraphs and cid in mem.short:
                mem.set_final(cid, draft)
                continue
            terms = mem.terms_of(cid) if with_terms else {}

            user = editor_user_prompt(src, draft, terms)
            budget = min(cfg.answer_budget,
                         max(96, int(editor.n_tokens(draft or src) * 1.8) + 96))
            edited = clean_output(
                editor.generate([{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                                max_new_tokens=budget, temperature=0.2),
                keep_lines=bool(c["leads"]))

            carried = source_carryover(src, edited, draft) if edited else None
            if edited and not looks_like_leak(edited, src) and not carried:
                mem.set_final(cid, edited, edited=norm(edited) != norm(draft))
            else:
                if carried and cfg.verbose:
                    print(f"\r        chunk {cid}: edit rejected, untranslated "
                          f"Spanish spliced in ({carried!r})")
                mem.set_final(cid, draft)
        if cfg.verbose:
            print()
        return state
    return verify_node


# ===========================================================================
# Stage 6 — format
# ===========================================================================


def make_format(cfg, ctx):
    def format_node(state: DocState) -> DocState:
        mem = state["memory"]
        if not cfg.format_repair:
            return state
        for c in mem.chunks:
            cid = c["cid"]
            text = mem.finals.get(cid, "")
            if not text:
                continue
            src_lines = c["src"].split("\n")
            out_lines = text.split("\n")
            if len(src_lines) == len(out_lines):
                fixed = "\n".join(repair_format(s, t)
                                  for s, t in zip(src_lines, out_lines))
            else:
                fixed = repair_format(c["src"].replace("\n", " "), text)
            if fixed and fixed != text:
                mem.formatted.add(cid)
                mem.finals[cid] = fixed
        return state
    return format_node


# ===========================================================================
# Stage 7 — finalize
# ===========================================================================


def make_finalize(cfg):
    def finalize_node(state: DocState) -> DocState:
        mem = state["memory"]
        mem.translation = enforce_structure(
            mem.source, rebuild(mem.parts, mem.chunks, mem.finals), cfg.verbose)
        if cfg.diagnostics:
            forms_by_chunk = {}
            for c in mem.chunks:
                cid = c["cid"]
                d = diagnose_chunk(c["src"], mem.finals.get(cid, ""),
                                   mem.terms_of(cid))
                mem.diagnostics[cid] = d["issues"]
                forms_by_chunk[cid] = d["forms"]
            mem.consistency = consistency_report(forms_by_chunk)
        return state
    return finalize_node


# ===========================================================================
# Instrumentation and graph
# ===========================================================================

_INNER_PROGRESS = {"translate", "verify", "select"}


def _summary(name, state):
    mem = state["memory"]
    if name == "prepare":
        sents = sum(len(split_sentences(c["src"])) for c in mem.chunks)
        merged = sum(len(c["leads"]) for c in mem.chunks)
        return (f"{len(mem.chunks)} chunks / {sents} sentences"
                + (f" ({merged} short paragraph(s) merged forward)" if merged else ""))
    if name == "match":
        n = sum(1 for t in mem.candidates.values() if t)
        distinct = {k for d in mem.candidates.values() for k in d}
        return f"{len(distinct)} candidate terms over {n} chunks"
    if name == "select":
        added = sum(len(v.get("added", [])) for v in mem.agent_notes.values())
        dropped = sum(len(v.get("dropped", [])) for v in mem.agent_notes.values())
        return (f"{len(mem.all_terms())} terms confirmed "
                f"(+{added} recovered, -{dropped} pruned)")
    if name == "translate":
        return f"{sum(1 for v in mem.drafts.values() if v)} chunks translated"
    if name == "verify":
        return f"{len(mem.edited)}/{len(mem.chunks)} chunks edited"
    if name == "format":
        return f"{len(mem.formatted)} chunks reformatted"
    if name == "finalize":
        issues = Counter(i["type"] for v in mem.diagnostics.values() for i in v)
        return (f"{len(mem.translation)} characters | diagnostics: "
                f"{dict(issues) or 'clean'}"
                + ("" if mem.consistency["ok"]
                   else f" | inconsistent stems: {list(mem.consistency['inconsistent'])}"))
    return ""


def timed(name, fn, cfg):
    def wrapper(state):
        inner = name in _INNER_PROGRESS
        if cfg.verbose:
            print(f"      -> {name:<9} ...", end="\n" if inner else "", flush=True)
        t0 = time.perf_counter()
        state = fn(state)
        dt = round(time.perf_counter() - t0, 2)
        state["timings"] = (state.get("timings") or []) + [(name, dt)]
        if cfg.verbose:
            print(f"{'' if inner else chr(13)}      v {name:<9} [{dt:7.2f}s]  "
                  f"{_summary(name, state)}")
        return state
    return wrapper


def build_graph(cfg, ctx):
    """Strictly linear, one pass, one source text per invocation."""
    g = StateGraph(DocState)
    stages = [("prepare", make_prepare(cfg)),
              ("match", make_match(cfg, ctx)),
              ("select", make_select(cfg, ctx)),
              ("translate", make_translate(cfg, ctx)),
              ("verify", make_verify(cfg, ctx)),
              ("format", make_format(cfg, ctx)),
              ("finalize", make_finalize(cfg))]
    for name, fn in stages:
        g.add_node(name, timed(name, fn, cfg))
    g.set_entry_point(stages[0][0])
    for (a, _), (b, _) in zip(stages, stages[1:]):
        g.add_edge(a, b)
    g.add_edge(stages[-1][0], END)
    return g.compile()


def translate_document(app, idx: int, text: str, mode: str, domain: str):
    """ONE source text, start to finish, on a store built fresh for it."""
    init: DocState = {"memory": DocMemory(idx, text), "mode": mode,
                      "domain": domain, "timings": None}
    result = app.invoke(init)
    return result["memory"], result["timings"]
