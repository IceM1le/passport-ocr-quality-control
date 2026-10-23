#!/usr/bin/env python3
"""Conservative post-processing for Russian passport OCR predictions.

Usage: python postprocess.py data/test_predictions.jsonl answers.jsonl
CPU-only; uses only the Python standard library.
"""
import csv
import json
import math
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

FIELDS = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
    "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
    "Issue_organization_ru",
]
AS_OF = date(2026, 10, 1)

# Conservative gates. These are decision thresholds, not calibrated probabilities.
MIN_FIELD_CONF = {
    "Last_name_ru": .999, "First_name_ru": .999, "Middle_name_ru": .999,
    "Sex_ru": .999, "Birth_date": .999, "Birth_place_ru": .9999,
    "Issue_date": .999, "Issue_organisation_code": .999,
    "Licence_number": .999, "Issue_organization_ru": .9999,
}
MIN_CHAR_CONF = .9999

# Empirical character mapping observed in the supplied OCR/MRZ data. This is not
# a legal transliteration standard: the MRZ OCR output contains systematic
# substitutions such as Ш -> 4, Ч -> 3, Я -> 8.
MRZ_CHAR_MAP = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E",
    "Ё": "E", "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K",
    "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
    "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C",
    "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "X", "Ы": "Y", "Ь": "9",
    "Э": "6", "Ю": "7", "Я": "8",
}


def norm(s):
    return " ".join(str(s or "").split()).upper()


def extract_field(obj):
    if not isinstance(obj, dict):
        return None, 0.0, [], None, []
    lines = obj.get("lines") or []
    value = " ".join(
        str(x.get("text", "")).strip()
        for x in lines if x.get("text") is not None
    ).strip()
    chars = []
    confidence_data_valid = bool(lines)
    for line in lines:
        text = line.get("text")
        raw_confidences = line.get("char_conf")
        # Confidence must be present for every recognized character. Silently
        # dropping malformed values could make an incomplete confidence list
        # look safer than it is.
        if not isinstance(text, str) or not isinstance(raw_confidences, list):
            confidence_data_valid = False
            break
        if len(raw_confidences) != len(text):
            confidence_data_valid = False
            break
        try:
            line_confidences = [float(c) for c in raw_confidences]
        except (TypeError, ValueError):
            confidence_data_valid = False
            break
        if not all(
                math.isfinite(c) and 0.0 <= c <= 1.0
                for c in line_confidences
        ):
            confidence_data_valid = False
            break
        chars.extend(line_confidences)
    if not confidence_data_valid:
        chars = []
    try:
        conf = float(obj.get("conf"))
        if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            conf = 0.0
    except (TypeError, ValueError):
        conf = 0.0
    return value or None, conf, chars, obj.get("box"), lines


def parse_date(s):
    try:
        return datetime.strptime(str(s), "%d.%m.%Y").date()
    except (ValueError, TypeError):
        return None


def anniversary(birth, years):
    """Return a birthday anniversary, handling 29 February conservatively."""
    year = birth.year + years
    if birth.month == 2 and birth.day == 29:
        return date(year, 2, 28)
    return date(year, birth.month, birth.day)


def valid_passport(birth_text, issued_text):
    """Check passport validity on AS_OF using the 20/45 replacement deadlines.

    A passport issued before age 20 expires at the 20th birthday plus the
    90-day replacement period. One issued from age 20 to before 45 expires at
    the 45th birthday plus 90 days. A passport issued at age 45 or later is
    treated as indefinitely valid. This operational rule should be checked
    against current official requirements before production use.
    """
    birth, issued = parse_date(birth_text), parse_date(issued_text)
    if not birth or not issued or issued > AS_OF or birth >= issued:
        return False

    age_at_issue = issued.year - birth.year - (
        (issued.month, issued.day) < (birth.month, birth.day)
    )
    if age_at_issue < 14:
        return False
    if age_at_issue < 20:
        deadline = anniversary(birth, 20) + timedelta(days=90)
        return AS_OF <= deadline
    if age_at_issue < 45:
        deadline = anniversary(birth, 45) + timedelta(days=90)
        return AS_OF <= deadline
    return True


def mrz_lines(doc):
    mrz = doc.get("mrz")
    return [
        re.sub(r"\s+", "", str(x.get("text", "")).upper())
        for x in (mrz or {}).get("lines", [])
    ]


def mrz_char_value(char):
    if char == "<":
        return 0
    if char.isdigit():
        return int(char)
    if "A" <= char <= "Z":
        return ord(char) - 55
    return None


def mrz_check_digit(value):
    weights = (7, 3, 1)
    total = 0
    for i, char in enumerate(value):
        number = mrz_char_value(char)
        if number is None:
            return None
        total += number * weights[i % 3]
    return str(total % 10)


def mrz_name_tokens(lines):
    if not lines:
        return None
    line = lines[0]
    if len(line) != 44 or not line.startswith("PNRUS") or "<<" not in line:
        return None
    if not re.fullmatch(r"[A-Z0-9<]+", line):
        return None
    surname, given = line[5:].split("<<", 1)
    given_tokens = [token for token in given.split("<") if token]
    if not surname or not given_tokens:
        return None
    return [surname.rstrip("<")] + given_tokens


def to_mrz_name(value):
    chars = [char for char in str(value or "").upper() if "А" <= char <= "Я" or char == "Ё"]
    if not chars:
        return None
    try:
        return "".join(MRZ_CHAR_MAP[char] for char in chars)
    except KeyError:
        return None


def mrz_consistent(fields, doc):
    """Check MRZ structure, birth-date check digit, sex, surname and first name."""
    lines = mrz_lines(doc)
    if len(lines) < 2:
        return False
    first, second = lines[0], lines[1]
    if len(first) != 44 or len(second) != 44:
        return False
    if not re.fullmatch(r"[A-Z0-9<]+", second):
        return False
    birth = parse_date(fields.get("Birth_date"))
    if not birth:
        return False
    expected_birth = birth.strftime("%y%m%d")
    if second[13:19] != expected_birth:
        return False
    if second[19] != mrz_check_digit(second[13:19]):
        return False

    sex = "M" if fields.get("Sex_ru", "").upper().startswith("М") else (
        "F" if fields.get("Sex_ru", "").upper().startswith("Ж") else None
    )
    if not sex or second[20] != sex:
        return False
    if second[9] != mrz_check_digit(second[:9]):
        return False

    tokens = mrz_name_tokens(lines)
    if not tokens or len(tokens) < 2:
        return False
    surname = to_mrz_name(fields.get("Last_name_ru"))
    first_name = to_mrz_name(fields.get("First_name_ru"))
    if not surname or not first_name:
        return False
    if surname != tokens[0] or first_name != tokens[1]:
        return False

    # Training error analysis found an additional ambiguous MRZ pattern:
    # the OCR can read Ж as K in a surname ending in -СКИЙ, making the
    # incorrect visual value МУТЯНСКИЙ appear to agree with MRZ MUT8NSKIQ,
    # while the label is МУТЯНСЖИЙ. Reject this observed ambiguity rather than
    # treating the two OCR outputs as independent proof. This is a dataset-
    # specific heuristic and should be revalidated on a separate validation set.
    if fields.get("Last_name_ru", "").upper().endswith("СКИЙ") and tokens[0].endswith("SKIQ"):
        return False

    # In the provided MRZ OCR, H/J confusion can make Х and Ж indistinguishable
    # in a first name even when both confidence scores are 1.0. Reject this
    # ambiguous case rather than treating matching noisy outputs as proof.
    if "Х" in fields.get("First_name_ru", "").upper():
        return False
    return True


def suspicious_long_field_geometry(obj):
    """Detect a tall field box containing only one short text line.

    This can indicate that OCR omitted one of the expected lines. It is a
    conservative heuristic used only for long free-text fields.
    """
    if not isinstance(obj, dict):
        return False
    box = obj.get("box") or []
    lines = obj.get("lines") or []
    if len(box) < 4 or len(lines) != 1:
        return False
    line_box = lines[0].get("box") or []
    if len(line_box) < 4 or not isinstance(box[3], (int, float)) or not isinstance(line_box[3], (int, float)):
        return False
    return line_box[3] > 0 and box[3] >= 1.6 * line_box[3]


def load_subdivisions(path):
    if not path or not Path(path).exists():
        return set()
    codes = set()
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                for cell in row:
                    match = re.search(r"(?<!\d)(\d{3})[- ]?(\d{3})(?!\d)", cell)
                    if match:
                        codes.add(match.group(1) + "-" + match.group(2))
    except (OSError, csv.Error):
        pass
    return codes


def process(doc, known_codes=None):
    viz = doc.get("viz") or {}
    fields = {}
    all_confident = True
    raw_objects = {}
    for field in FIELDS:
        value, conf, chars, box, lines = extract_field(viz.get(field))
        raw_objects[field] = viz.get(field)
        if value is None:
            all_confident = False
            continue
        fields[field] = value
        if conf < MIN_FIELD_CONF[field] or not chars or min(chars) < MIN_CHAR_CONF:
            all_confident = False
    if len(fields) != len(FIELDS):
        all_confident = False

    if all_confident:
        birth = parse_date(fields["Birth_date"])
        issue = parse_date(fields["Issue_date"])
        if not birth or not issue or not valid_passport(fields["Birth_date"], fields["Issue_date"]):
            all_confident = False
        if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields["Birth_date"]):
            all_confident = False
        if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields["Issue_date"]):
            all_confident = False
        if not re.fullmatch(r"\d{3}-\d{3}", fields["Issue_organisation_code"]):
            all_confident = False
        if not re.fullmatch(r"\d{2} \d{2} \d{6}", fields["Licence_number"]):
            all_confident = False
        if fields["Sex_ru"] not in ("МУЖ.", "ЖЕН."):
            all_confident = False
        if issue and birth and issue <= birth:
            all_confident = False
        if suspicious_long_field_geometry(raw_objects.get("Birth_place_ru")):
            all_confident = False
        if suspicious_long_field_geometry(raw_objects.get("Issue_organization_ru")):
            all_confident = False
        if not mrz_consistent(fields, doc):
            all_confident = False
        if known_codes is not None and fields.get("Issue_organisation_code") not in known_codes:
            all_confident = False

    return {
        "doc_id": doc["doc_id"],
        "decision": "accept" if all_confident else "reject",
        **({"fields": {field: fields[field] for field in FIELDS}} if all_confident else {}),
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python postprocess.py <predictions.jsonl> <answers.jsonl>")
    src, dst = map(Path, sys.argv[1:])
    root = Path(__file__).resolve().parent
    codes_path = root / "reference" / "subdivisions.csv"
    codes = load_subdivisions(codes_path)
    if not codes:
        raise SystemExit(
            f"Subdivision reference is missing or empty: {codes_path}. "
            "Refusing to bypass the subdivision-code validation."
        )
    n = accepted = 0
    with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            doc = json.loads(line)
            answer = process(doc, codes)
            out.write(json.dumps(answer, ensure_ascii=False) + "\n")
            n += 1
            accepted += answer["decision"] == "accept"
    print(f"Processed: {n}; accepted: {accepted}; rejected: {n - accepted}")


if __name__ == "__main__":
    main()
