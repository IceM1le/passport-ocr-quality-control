#!/usr/bin/env python3
"""Консервативная постобработка OCR-результатов российских паспортов.

Запуск: python postprocess.py data/test_predictions.jsonl answers.jsonl
Используется только стандартная библиотека Python; достаточно CPU.
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

# Консервативные пороги принятия решения, а не откалиброванные вероятности.
MIN_FIELD_CONF = {
    "Last_name_ru": .999, "First_name_ru": .999, "Middle_name_ru": .999,
    "Sex_ru": .999, "Birth_date": .999, "Birth_place_ru": .9999,
    "Issue_date": .999, "Issue_organisation_code": .999,
    "Licence_number": .999, "Issue_organization_ru": .9999,
}
MIN_CHAR_CONF = .9999

# Эмпирическое соответствие символов, обнаруженное в предоставленных OCR/MRZ-данных.
# Это не официальный стандарт транслитерации: в OCR машиночитаемой зоны встречаются
# систематические замены, например Ш -> 4, Ч -> 3, Я -> 8.
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
        # Уверенность должна быть указана для каждого распознанного символа.
        # Если молча отбросить некорректные значения, неполный список уверенности
        # может ошибочно выглядеть надёжнее, чем он есть на самом деле.
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
    """Возвращает дату годовщины рождения; для 29 февраля используется консервативное правило."""
    year = birth.year + years
    if birth.month == 2 and birth.day == 29:
        return date(year, 2, 28)
    return date(year, birth.month, birth.day)


def valid_passport(birth_text, issued_text):
    """Проверяет действительность паспорта на дату AS_OF с учётом замены в 20/45 лет.

    Паспорт, выданный до 20-летия, считается действительным до окончания 90-дневного
    периода после 20-летия. Паспорт, выданный с 20 лет до 45 лет, — до окончания
    90-дневного периода после 45-летия. Паспорт, выданный в 45 лет или позднее,
    считается действительным бессрочно. Перед использованием в рабочей системе
    это правило необходимо сверить с актуальными официальными требованиями.
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
    """Проверяет структуру MRZ, контрольную цифру даты рождения, пол, фамилию и имя."""
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

    # При анализе ошибок train обнаружен дополнительный неоднозначный шаблон MRZ:
    # OCR может распознать Ж как K в фамилии, оканчивающейся на -СКИЙ. Тогда ошибочное
    # визуальное значение МУТЯНСКИЙ выглядит согласованным с MRZ MUT8NSKIQ,
    # хотя в разметке указано МУТЯНСЖИЙ. Отклоняем этот случай, а не считаем два
    # OCR-результата независимым подтверждением. Эвристика специфична для набора данных
    # и требует повторной проверки на отдельной валидационной выборке.
    if fields.get("Last_name_ru", "").upper().endswith("СКИЙ") and tokens[0].endswith("SKIQ"):
        return False

    # В предоставленных OCR-результатах MRZ путаница H/J может сделать Х и Ж
    # неразличимыми в имени, даже если оба показателя уверенности равны 1.0.
    # Отклоняем такой неоднозначный случай и не считаем совпадение шумных результатов доказательством.
    if "Х" in fields.get("First_name_ru", "").upper():
        return False
    return True


def suspicious_long_field_geometry(obj):
    """Находит высокую область поля, в которой распознана только одна короткая строка.

    Это может означать, что OCR пропустил одну из ожидаемых строк. Консервативная
    эвристика применяется только к длинным текстовым полям.
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
