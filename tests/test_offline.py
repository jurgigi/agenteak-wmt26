"""Tests for everything that does not need a GPU or a model.

    python -m tests.test_offline        (from the repo root)

These cover chunking, reconstruction, terminology matching, near-miss recall,
glossary/document loading and the terminology agent's reply parser — i.e. the
parts where a silent regression would corrupt a submission without raising.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteak.dataio import load_documents, load_terminology, output_stem
from agenteak.terminology import TerminologyDatabase, match_target, nominal_key
from agenteak.textproc import (chunk_paragraph, enforce_structure,
                               newline_signature, rebuild, repair_format,
                               split_document, split_sentences)

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}\n     got:  {got!r}\n     want: {want!r}")
        print(f"  FAIL {name}")
    else:
        print(f"  ok   {name}")


def section(t):
    print(f"\n{t}\n{'-' * len(t)}")


# ---------------------------------------------------------------------------
section("sentence splitting and chunking")

check("abbreviation not a boundary",
      len(split_sentences("La presión máx. es de 3 bar. El motor arranca.")), 2)
check("three-sentence chunk",
      len(chunk_paragraph("Uno. Dos. Tres. Cuatro. Cinco.", 3)), 2)
check("empty paragraph", chunk_paragraph(""), [])

# ---------------------------------------------------------------------------
section("document structure round-trip")

DOC = "Título\n\nPrimer párrafo. Segunda frase.\n\nSegundo párrafo."
parts, chunks = split_document(DOC)
finals = {c["cid"]: c["src"] for c in chunks}          # identity "translation"
check("round-trip preserves the document", rebuild(parts, chunks, finals), DOC)
check("newline signature", newline_signature(DOC), ["\n\n", "\n\n"])
check("structure repair restores runs",
      newline_signature(enforce_structure(DOC, "A B C", verbose=0)),
      ["\n\n", "\n\n"])

# ---------------------------------------------------------------------------
section("Spanish nominal folding")

check("plural -es", nominal_key("motores", False), "motor")
check("plural -ces", nominal_key("luces", False), "luz")
check("gender fold", nominal_key("frenos", True), nominal_key("frena", True))
check("vowel-final plural folds with its singular",
      nominal_key("aceites", False), nominal_key("aceite", False))
check("consonant-final plural folds with its singular",
      nominal_key("motores", False), nominal_key("motor", False))

# ---------------------------------------------------------------------------
section("terminology matching")

GLOSS = {
    "motor de combustión interna": ["barne-errekuntzako motor"],
    "aceite de motor": ["motor-olio"],
    "bomba de aceite": ["olio-ponpa"],
    "sistema de frenado": ["balaztatze-sistema"],
    "sistema de refrigeración": ["hozte-sistema"],
    "batería": ["bateria", "pila"],
}
db = TerminologyDatabase(GLOSS, max_gap=0)

check("plural inflection matches",
      "motor de combustión interna" in db.match("Los motores de combustión interna giran."),
      True)
check("intervening function words are free",
      "sistema de frenado" in db.match("El sistema del frenado es hidráulico."),
      True)
check("gap>0 not allowed by default",
      "aceite de motor" in db.match("La bomba de aceite movida por el motor."),
      False)
check("but the real term IS found there",
      "bomba de aceite" in db.match("La bomba de aceite movida por el motor."),
      True)
check("single-word entry",
      "batería" in db.match("La batería alimenta el arranque."), True)

near = db.near_misses("El vehículo incorpora sistemas de frenado y de refrigeración.",
                      exclude={}, limit=10)
check("near-miss recovers the coordinated term",
      "sistema de refrigeración" in near, True)

# ---------------------------------------------------------------------------
section("Basque target matching")

check("declined form counts as present",
      match_target("barne-errekuntzako motor",
                   "barne-errekuntzako motorrak energia sortzen du")["status"],
      "present")
check("absent target",
      match_target("hozte-sistema", "bateriak ondo funtzionatzen du")["status"],
      "absent")
check("short target guarded",
      match_target("auto", "Automotive Society aipatzen da")["status"], "absent")

# ---------------------------------------------------------------------------
section("formatting repair")

check("terminal punctuation copied",
      repair_format("El motor arranca.", "Motorra abiarazten da"),
      "Motorra abiarazten da.")
check("initial case copied",
      repair_format("El motor.", "motorra."), "Motorra.")
check("space before percent removed",
      repair_format("30 % del total.", "guztiaren 30 %."), "guztiaren 30%.")

# ---------------------------------------------------------------------------
section("terminology agent reply parsing")

from agenteak.agents import parse_term_agent_reply, reconcile

check("plain JSON",
      parse_term_agent_reply('{"terms": [{"es": "batería", "eu": "bateria"}]}'),
      [{"es": "batería", "eu": "bateria"}])
check("fenced JSON",
      parse_term_agent_reply('```json\n{"terms": []}\n```'), [])
check("prose around JSON",
      parse_term_agent_reply('Sure! {"terms": [{"es": "a", "eu": "b"}]} done'),
      [{"es": "a", "eu": "b"}])
check("unparseable -> None", parse_term_agent_reply("no idea"), None)
check("empty -> None", parse_term_agent_reply(""), None)

terms, notes = reconcile(
    [{"es": "batería", "eu": "pila"},
     {"es": "sistema de frenado", "eu": "balaztatze-sistema"},
     {"es": "invented term", "eu": "asmatua"}],
    candidates={"batería": ["bateria", "pila"], "aceite de motor": ["motor-olio"]},
    possible={"sistema de frenado": ["balaztatze-sistema"]})
check("agent disambiguates", terms["batería"], ["pila"])
check("agent recovers a near miss", notes["added"], ["sistema de frenado"])
check("agent prunes a false positive", notes["dropped"], ["aceite de motor"])
check("invented terms rejected", "invented term" in terms, False)

unapproved, _ = reconcile([{"es": "batería", "eu": "asmatua"}],
                           candidates={"batería": ["bateria", "pila"]}, possible={})
check("unapproved target falls back to the first approved one",
      unapproved["batería"], ["bateria"])

# ---------------------------------------------------------------------------
section("I/O")

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "a.json").write_text(json.dumps(["doc one", "doc two"]), encoding="utf-8")
    docs, names, kind = load_documents(str(d / "a.json"), verbose=0)
    check("shared-task JSON", (len(docs), kind), (2, "json"))

    (d / "one.txt").write_text("Hola.\n\nAdiós.", encoding="utf-8")
    (d / "two.txt").write_text("Otro.", encoding="utf-8")
    docs, names, kind = load_documents(str(d), verbose=0)
    check("directory of txt", (len(docs), names, kind), (2, ["one", "two"], "text"))

    docs, names, kind = load_documents(str(d / "one.txt"), verbose=0)
    check("single txt", (len(docs), names, kind), (1, ["one"], "text"))

    (d / "g.tsv").write_text("energía renovable\tenergia berriztagarri\n"
                             "red eléctrica\tsare elektriko|elektrizitate-sare\n",
                             encoding="utf-8")
    t = load_terminology(str(d / "g.tsv"), verbose=0)
    check("TSV glossary -> proper", t["proper"]["red eléctrica"],
          ["sare elektriko", "elektrizitate-sare"])
    check("TSV glossary -> random empty", t["random"], {})

    (d / "t.json").write_text(json.dumps(
        {"proper": {"a": ["b"]}, "random": {"a": ["c"]}}), encoding="utf-8")
    t = load_terminology(str(d / "t.json"), verbose=0)
    check("shared-task terms", (t["proper"]["a"], t["random"]["a"]), (["b"], ["c"]))

check("output stem", output_stem("proper", "agenteak", "automotive", "eseu"),
      "agenteak.proper.automotive.eseu")
check("base_noterm stem", output_stem("base_noterm", "agenteak", "energy", "eseu"),
      "agenteak-base.noterm.energy.eseu")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAILED:
    print(f"{len(FAILED)} FAILURE(S):\n")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
