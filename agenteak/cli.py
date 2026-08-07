"""Command line entry point.

    python -m agenteak --domain automotive --mode proper \
        --input data/text.automotion.eseu.json \
        --terms data/terms.automotion.eseu.json \
        --out output/

    python -m agenteak --domain energy --mode noterm --input my_article.txt
"""

from __future__ import annotations

import argparse
import sys

from .config import Config, MODE_SPECS, canonical_domain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agenteak",
        description="Agenteak: multi-agent Spanish-to-Basque terminology-aware translation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io = p.add_argument_group("input / output")
    io.add_argument("--input", "-i", required=True, dest="text_path",
                    help="shared-task JSON, a .txt file, or a directory of .txt files")
    io.add_argument("--terms", "-t", dest="terms_path", default=None,
                    help="shared-task terms JSON, a flat JSON glossary, or a TSV")
    io.add_argument("--out", "-o", dest="out_dir", default="output",
                    help="output directory")
    io.add_argument("--system-name", default="agenteak",
                    help="prefix for output filenames")

    run = p.add_argument_group("what to run")
    run.add_argument("--domain", "-d", default="automotive",
                     help="automotive | energy (automotion is accepted)")
    run.add_argument("--mode", "-m", action="append", dest="modes", default=None,
                     choices=sorted(MODE_SPECS), help="repeatable")
    run.add_argument("--direction", default="eseu")
    run.add_argument("--limit", type=int, default=None, dest="limit_docs",
                     help="trial run on the first N documents")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--verbose", "-v", type=int, default=1, choices=(0, 1, 2))

    ag = p.add_argument_group("agents")
    ag.add_argument("--no-term-agent", action="store_true",
                    help="skip the Qwen terminology agent; use the deterministic "
                         "matcher alone")
    ag.add_argument("--no-thinking", action="store_true",
                    help="run the terminology agent without its reasoning phase")
    ag.add_argument("--no-editor-fewshot", action="store_true",
                    help="prompt the editor without its few-shot examples")
    ag.add_argument("--recall-candidates", type=int, default=10,
                    dest="term_agent_recall_candidates",
                    help="unmatched glossary entries offered to the agent per chunk")
    ag.add_argument("--translator-model", default=None)
    ag.add_argument("--editor-model", default=None)
    ag.add_argument("--term-agent-model", default=None)
    ag.add_argument("--base-model", default=None)
    ag.add_argument("--translator-adapter", default=None)
    ag.add_argument("--editor-adapter", default=None)

    adv = p.add_argument_group("advanced")
    adv.add_argument("--max-sentences", type=int, default=3)
    adv.add_argument("--term-max-gap", type=int, default=0,
                     help="content words allowed inside a term match")
    adv.add_argument("--no-format-repair", action="store_true")
    adv.add_argument("--no-diagnostics", action="store_true")
    adv.add_argument("--dump-memory", action="store_true",
                     help="write the per-document store next to the output")
    adv.add_argument("--4bit", dest="load_in_4bit", action="store_true", default=None)
    adv.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    return p


def config_from_args(args) -> Config:
    cfg = Config(
        domain=canonical_domain(args.domain),
        direction=args.direction,
        modes=tuple(args.modes or ["proper"]),
        seed=args.seed,
        limit_docs=args.limit_docs,
        verbose=args.verbose,
        text_path=args.text_path,
        terms_path=args.terms_path,
        out_dir=args.out_dir,
        system_name=args.system_name,
        term_agent=not args.no_term_agent,
        term_agent_thinking=not args.no_thinking,
        term_agent_recall_candidates=args.term_agent_recall_candidates,
        editor_few_shot=not args.no_editor_fewshot,
        max_sentences=args.max_sentences,
        term_max_gap=args.term_max_gap,
        format_repair=not args.no_format_repair,
        diagnostics=not args.no_diagnostics,
        dump_memory=args.dump_memory,
        load_in_4bit=args.load_in_4bit,
    )
    for attr, value in (("translator_model", args.translator_model),
                        ("editor_model", args.editor_model),
                        ("term_agent_model", args.term_agent_model),
                        ("base_model", args.base_model),
                        ("translator_adapter", args.translator_adapter),
                        ("editor_adapter", args.editor_adapter)):
        if value:
            setattr(cfg, attr, value)
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    from .runner import run
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
