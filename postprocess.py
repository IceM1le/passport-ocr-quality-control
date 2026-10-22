#!/usr/bin/env python3
"""Conservative post-processing for Russian passport OCR predictions.

Usage: python postprocess.py data/test_predictions.jsonl answers.jsonl
CPU-only, standard library only.
"""
import csv, json, re, sys
from datetime import date, datetime, timedelta
from pathlib import Path

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
AS_OF = date(2026, 10, 1)
# Conservative field-level confidence gates. These are risk controls, not calibrated probabilities.
MIN_FIELD_CONF = {
    "Last_name_ru": .999, "First_name_ru": .999, "Middle_name_ru": .999,
    "Sex_ru": .999, "Birth_date": .999, "Birth_place_ru": .9999,
    "Issue_date": .999, "Issue_organisation_code": .999,
    "Licence_number": .999, "Issue_organization_ru": .9999,
}
MIN_CHAR_CONF = .9999


def norm(s):
    return " ".join(str(s or "").split()).upper()


def extract_field(obj):
    if not isinstance(obj, dict):
        return None, 0.0, []
    lines = obj.get("lines") or []
    value = " ".join(str(x.get("text", "")).strip() for x in lines if x.get("text") is not None).strip()
    chars = [float(c) for line in lines for c in (line.get("char_conf") or []) if isinstance(c, (int, float))]
    conf = obj.get("conf")
    try: conf = float(conf)
    except (TypeError, ValueError): conf = 0.0
    return value or None, conf, chars


def parse_date(s):
    try: return datetime.strptime(str(s), "%d.%m.%Y").date()
    except (ValueError, TypeError): return None


def valid_passport(birth, issued):
    """Approximate validity under the 14/20/45 replacement rules and 90-day grace period.

    Ambiguous/unparseable dates are rejected. This is a conservative operational rule,
    not a legal opinion; production use must verify current official requirements.
    """
    b, i = parse_date(birth), parse_date(issued)
    if not b or not i or i > AS_OF or b >= i:
        return False
    age_at_issue = i.year - b.year - ((i.month, i.day) < (b.month, b.day))
    if age_at_issue < 14 or age_at_issue > 100:
        return False
    # A passport is normally issued at 14, 20, or 45; allow practical administrative variation.
    if not (14 <= age_at_issue <= 15 or 20 <= age_at_issue <= 21 or 45 <= age_at_issue <= 46):
        return False
    age_now = AS_OF.year - b.year - ((AS_OF.month, AS_OF.day) < (b.month, b.day))
    if age_now < 20:
        return age_at_issue <= 15
    if age_now < 45:
        return 14 <= age_at_issue <= 21
    # After 45, only a passport issued for the 45+ age bracket should remain valid.
    if age_at_issue < 45:
        try: deadline = date(b.year + 45, b.month, min(b.day, 28 if b.month == 2 else b.day)) + timedelta(days=90)
        except ValueError: return False
        return AS_OF <= deadline
    return True


def mrz_lines(doc):
    mrz = doc.get("mrz")
    return [re.sub(r"\s+", "", str(x.get("text", "")).upper()) for x in (mrz or {}).get("lines", [])]


def mrz_consistent(fields, doc):
    """Check key fields against ICAO-style MRZ text when a plausible 2-line MRZ exists."""
    lines = mrz_lines(doc)
    if len(lines) < 2 or len(lines[0]) < 30 or len(lines[1]) < 30:
        return False
    a, b = lines[0], lines[1]
    # MRZ name segment: P<RUS + surname << given names; tolerate transliteration differences
    # by comparing strong, normalized fragments only when present.
    surname = re.sub(r"[^A-Z]", "", fields["Last_name_ru"].upper())
    first = re.sub(r"[^A-Z]", "", fields["First_name_ru"].upper())
    # Cyrillic -> rough transliteration is intentionally not guessed. Instead validate MRZ structure.
    if not (a.startswith("P") and "RUS" in a[:10]) or "<<" not in a:
        return False
    if not re.fullmatch(r"[A-Z0-9<]+", a) or not re.fullmatch(r"[A-Z0-9<]+", b):
        return False
    # Date of birth (YYMMDD) and sex in line 2 positions 13-20 in standard passport MRZ.
    birth = parse_date(fields["Birth_date"])
    if not birth:
        return False
    yy = f"{birth.year % 100:02d}{birth.month:02d}{birth.day:02d}"
    # Common format: document-number/check + nationality + birth date/check + sex...
    if yy not in b:
        return False
    sex = "M" if fields["Sex_ru"].upper().startswith("М") else "F" if fields["Sex_ru"].upper().startswith("Ж") else None
    if not sex or sex not in b:
        return False
    return True


def load_subdivisions(path):
    if not path or not Path(path).exists(): return set()
    codes = set()
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                for cell in row:
                    m = re.search(r"(?<!\d)(\d{3})[- ]?(\d{3})(?!\d)", cell)
                    if m: codes.add(m.group(1) + "-" + m.group(2))
    except (OSError, csv.Error):
        pass
    return codes


def process(doc, known_codes=None):
    viz = doc.get("viz") or {}
    fields, all_confident = {}, True
    for f in FIELDS:
        value, conf, chars = extract_field(viz.get(f))
        if value is None:
            all_confident = False
            continue
        fields[f] = value
        if conf < MIN_FIELD_CONF[f] or not chars or min(chars) < MIN_CHAR_CONF:
            all_confident = False
    if len(fields) != len(FIELDS): all_confident = False

    if all_confident:
        # Strict syntax and chronology checks.
        birth, issue = parse_date(fields["Birth_date"]), parse_date(fields["Issue_date"])
        if not birth or not issue or not valid_passport(fields["Birth_date"], fields["Issue_date"]):
            all_confident = False
        if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields["Birth_date"]): all_confident = False
        if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields["Issue_date"]): all_confident = False
        if not re.fullmatch(r"\d{3}-\d{3}", fields["Issue_organisation_code"]): all_confident = False
        if not re.fullmatch(r"\d{2} \d{2} \d{6}", fields["Licence_number"]): all_confident = False
        if fields["Sex_ru"] not in ("МУЖ.", "ЖЕН."): all_confident = False
        if issue and birth and issue <= birth: all_confident = False
        # Require a plausible MRZ and cross-check date/sex. Missing or inconsistent MRZ => reject.
        if not mrz_consistent(fields, doc): all_confident = False
        if known_codes and fields.get("Issue_organisation_code") not in known_codes: all_confident = False

    return {"doc_id": doc["doc_id"], "decision": "accept" if all_confident else "reject",
            **({"fields": {f: fields[f] for f in FIELDS}} if all_confident else {})}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python postprocess.py <predictions.jsonl> <answers.jsonl>")
    src, dst = map(Path, sys.argv[1:])
    root = Path(__file__).resolve().parent
    codes = load_subdivisions(root / "reference" / "subdivisions.csv")
    n = accepted = 0
    with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip(): continue
            doc = json.loads(line)
            ans = process(doc, codes)
            out.write(json.dumps(ans, ensure_ascii=False) + "\n")
            n += 1
            accepted += ans["decision"] == "accept"
    print(f"Processed: {n}; accepted: {accepted}; rejected: {n-accepted}")

if __name__ == "__main__": main()
