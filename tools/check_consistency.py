"""Consistency audit of the manuscript and supplement against the repository.

Usage:
    python -m tools.check_consistency MANUSCRIPT.docx SUPPLEMENT.docx [RESPONSE.docx]

Checks
  1. Figure and table captions: each numbered once, each cited in the text.
  2. Supplementary tables, figures and appendices cited in the manuscript exist.
  3. Section cross references point to existing headings.
  4. Every author-year citation has a reference entry and every entry is cited.
  5. Every decimal number, percentage and "k of 30" count in the results text
     (Abstract, Section 4, Section 6.1.1, Supplementary Appendix B) occurs in
     results/manifest.json or is a fixed design constant. This is a presence
     check: it cannot detect a wrong attribution, sign or interval, so numbers
     were also traced by hand.
  6. Strings that belonged to earlier, superseded versions are absent.
Exit status 1 if any check fails.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
FAIL = []


def say(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAIL.append(msg)


def blocks(path):
    """Yield (kind, style, text) for paragraphs and tables in body order."""
    doc = Document(path)
    for el in doc.element.body.iterchildren():
        if el.tag == qn("w:p"):
            sty = el.find(qn("w:pPr"))
            s = ""
            if sty is not None and sty.find(qn("w:pStyle")) is not None:
                s = sty.find(qn("w:pStyle")).get(qn("w:val"))
            yield "p", s, "".join(t.text or "" for t in el.iter(qn("w:t")))
        elif el.tag == qn("w:tbl"):
            cells = []
            for tc in el.iter(qn("w:tc")):
                cells.append("".join(t.text or "" for t in tc.iter(qn("w:t"))))
            yield "t", "", " | ".join(cells)


def split_ms(path):
    body, refs, headings, captions = [], [], [], []
    in_refs = False
    for kind, sty, text in blocks(path):
        if kind == "p" and text.strip() == "References":
            in_refs = True
            continue
        if in_refs:
            if text.strip():
                refs.append(text.strip())
            continue
        if kind == "p" and sty.startswith("Heading"):
            headings.append(text.strip())
        elif kind == "p" and re.match(r"^\d+(\.\d+)+ \S", text.strip()) and len(text) < 120:
            headings.append(text.strip())
        if kind == "p" and re.match(r"^(Figure|Table) S?\d+\.", text.strip()):
            captions.append(text.strip())
        body.append(text)
    return body, refs, headings, captions


def main(ms_path, sm_path, resp_path=None):
    body, refs, headings, captions = split_ms(ms_path)
    sm_body, _, sm_headings, sm_captions = split_ms(sm_path)
    text = "\n".join(body)
    sm_text = "\n".join(sm_body)

    print("1. Captions and in-text references (manuscript)")
    for kind, n in (("Figure", 8), ("Table", 6)):
        for i in range(1, n + 1):
            caps = [c for c in captions if c.startswith(f"{kind} {i}.")]
            say(len(caps) == 1, f"{kind} {i}: one caption ({len(caps)} found)")
            n_all = len(re.findall(rf"\b{kind}s? {i}(?:[a-z](?:, [a-z])?)?\b", text))
            n_all += len(re.findall(rf"\b{kind}s \d+(?:, \d+)* and {i}\b", text))
            n_all += len(re.findall(rf"\b{kind} \d+ and {kind} {i}\b", text))
            cites = n_all - len(caps)
            say(cites >= 1, f"{kind} {i}: cited in text ({cites})")
        extra = [c for c in captions if re.match(rf"^{kind} (\d+)\.", c) and int(re.match(rf"^{kind} (\d+)", c).group(1)) > n]
        say(not extra, f"no {kind.lower()} numbered beyond {n}")

    print("2. Supplementary items cited in the manuscript")
    for m in sorted(set(re.findall(r"Supplementary (Table|Figure) (S\d+)", text))):
        ok = any(c.startswith(f"{m[0]} {m[1]}.") for c in sm_captions)
        say(ok, f"{m[0]} {m[1]} exists in the supplement")
    for kind in ("Table", "Figure"):
        for c in sm_captions:
            mm = re.match(rf"^{kind} (S\d+)\.", c)
            if mm:
                tag = mm.group(1)
                n_cites = len(re.findall(rf"\b{kind}s? (?:S\d+(?:, | and | to ))*{tag}\b", text))
                say(n_cites >= 1, f"{kind} {tag} cited in the manuscript ({n_cites})")
    for m in sorted(set(re.findall(r"Appendix ([A-C](?:\.\d)?)", text))):
        ok = any(h.startswith(f"Appendix {m[0]}") for h in sm_headings) if "." not in m else any(h.startswith(m) for h in sm_headings)
        say(ok, f"Supplementary Appendix {m} exists")

    print("3. Section cross references")
    sec_ids = set()
    for h in headings:
        mm = re.match(r"^(\d+(?:\.\d+)*)\.? ", h)
        if mm:
            sec_ids.add(mm.group(1))
    for ref in sorted(set(re.findall(r"Sections? (\d+(?:\.\d+)*)", text)) | set(re.findall(r"Sections \d+ and (\d+)", text))):
        say(ref in sec_ids, f"Section {ref} exists")

    print("4. Citations and reference list")
    entries = []
    for r in refs:
        m = re.match(r"^(.+?)\((\d{4})\)", r)
        if not m:
            say(False, f"unparsed reference: {r[:60]}")
            continue
        authors = m.group(1)
        surnames = re.findall(r"(?:^|, |& )((?:van |de |van den )?[A-ZÀ-Ž][\w'\-À-ž]+(?: [A-Z][\w\-]+)?),", authors)
        entries.append({"first": surnames[0] if surnames else authors.split(",")[0],
                        "second": surnames[1] if len(surnames) > 1 else None,
                        "n": len(surnames), "year": m.group(2), "text": r, "cited": 0})
    cit_pat = re.compile(
        r"((?:van |de )?[A-Z][\w'\-À-ž]+)(?:,? (?:and|&) ((?:van )?[A-Z][\w\-À-ž]+)|, ([A-Z][\w\-]+), et al\.| et al\.)?,? \(?(\d{4})\)?")
    unmatched = []
    for m in cit_pat.finditer(text):
        first, second2, second_etal, year = m.group(1), m.group(2), m.group(3), m.group(4)
        if first in ("Section", "Figure", "Table", "Step", "Gap", "Gaps", "Problem", "January", "March", "June",
                     "Appendix", "Python", "Level", "Before", "Since", "In", "By", "Of", "The", "We", "From", "Zenodo",
                     "October"):
            continue
        if not (1900 < int(year) < 2030):
            continue
        cands = [e for e in entries if e["first"].split()[-1] == first.split()[-1] and e["year"] == year]
        sec = second2 or second_etal
        if sec:
            cands = [e for e in cands if e["second"] and e["second"].split()[-1] == sec.split()[-1]] or cands
        if len(cands) == 1:
            cands[0]["cited"] += 1
        elif len(cands) > 1:
            unmatched.append(f"ambiguous: {m.group(0)}")
        else:
            unmatched.append(m.group(0))
    say(not unmatched, f"all in-text citations matched ({len(unmatched)} unmatched)")
    for u in sorted(set(unmatched)):
        print("        ", u)
    uncited = [e["text"][:70] for e in entries if e["cited"] == 0]
    say(not uncited, f"all {len(entries)} reference entries cited ({len(uncited)} uncited)")
    for u in uncited:
        print("        ", u)

    print("5. Numbers traceable to the results manifest")
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    allowed = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            for nd in (0, 1, 2, 3):
                allowed.add(f"{abs(x):.{nd}f}")
                allowed.add(f"{abs(x) * 100:.{nd}f}")
    walk(manifest)
    for v in manifest["C"]["standard"]["learned"].values():
        for p_ in v["params_median"]:
            allowed.add(f"{abs(p_):.2f}")
    # design constants and literature figures that are not results
    allowed |= {"0.95", "0.05", "0.1", "0.5", "1.0", "1.5", "0.01", "0.02", "0.2", "0.6", "0.9", "0.4", "0.75", "0.72",
                "0.03", "2.5", "4.5", "0.025", "0.975", "0.999", "0.998", "0.9995", "0.99", "0.30", "0.50", "0.25",
                "1.00", "0.00", "0.0", "2.0", "3.0", "5.0", "0.8", "1.2", "1.25", "1.15", "11.5", "8.5", "1.5", "0.12",
                "0.17"}
    allowed |= {f"{x}" for x in ("3.13", "2.5.3", "1.18.1", "3.11.2", "6.0.0", "1.1", "1.2", "1.3")}

    def results_text(paragraphs, start_pat, end_pat):
        out, on = [], False
        for p in paragraphs:
            if re.match(start_pat, p.strip()):
                on = True
            elif re.match(end_pat, p.strip()):
                on = False
            if on:
                out.append(p)
        return "\n".join(out)

    sec4 = results_text(body, r"^4\. Evidence", r"^5\. Six")
    abstract = results_text(body, r"^Abstract$", r"^Keywords")
    s611 = results_text(body, r"^6\.1\.1", r"^6\.2")
    appB = results_text(sm_body, r"^Appendix B", r"^Appendix C")
    bad = []
    for name, chunk in (("abstract", abstract), ("Section 4", sec4), ("Section 6.1.1", s611), ("Appendix B", appB)):
        chunk = chunk.replace("−", "-")
        chunk = re.sub(r"(?<=\d),(?=\d{3}\b)", "", chunk)        # thousands separators
        chunk = "\n".join(re.sub(r"^\d+(\.\d+)+ ", "", line) for line in chunk.split("\n"))
        for tok in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", chunk):
            if tok not in allowed:
                bad.append(f"{name}: {tok}")
        for tok in re.findall(r"(?<![\w.])(\d+)%", chunk):          # percentages
            if tok not in allowed and tok != "95":
                bad.append(f"{name}: {tok}%")
        for tok in re.findall(r"(?<![\w.])(\d+) of 30\b", chunk):     # seed and field counts
            if tok not in allowed:
                bad.append(f"{name}: {tok} of 30")
    say(not bad, f"decimal numbers traceable ({len(bad)} not found)")
    for b in sorted(set(bad))[:60]:
        print("        ", b)

    print("6. Superseded strings")
    banned = ["Hafner & Boettiger", "Hafner& Boettiger", "Marrec", "Lin et al.", "statistically identical",
              "n = 78", "98 references", "Seventeen", "seventeen", "RMSE from 0.256", "0.100 at", "69%", "69 percent", "2024b",
              "Phankomolsil", "Shalev et al.", "Liao", "Tolan", "Gu et al., 2020", "Gu et al. (2020)",
              "Murray et al., 2024", "Karmoude et al., 2023", "small conv", "pre specified", "pre-registered",
              "CHIRO", "reuseable", "Figure 7 and use", "Section 6.3.1", "v4.0.1", "Cohen, D", "Wu et al., 2024b",
              "Appendix B of Supplementary material", "limitation in Section 8"]
    for b in banned:
        n = text.count(b) + sm_text.count(b)
        say(n == 0, f"absent: '{b}'" + (f" ({n} found)" if n else ""))
    dashes = text.count("—") + sm_text.count("—")
    say(dashes == 0, f"no em-dash sentence breaks ({dashes} found)")
    if resp_path:
        rtext = "\n".join(t for _, _, t in blocks(resp_path))
        dashes = rtext.count("—")
        say(dashes == 0, f"response letter: no em dashes ({dashes} found)")

    print(f"\n{len(FAIL)} check(s) failed.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
