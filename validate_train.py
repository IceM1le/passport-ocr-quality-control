#!/usr/bin/env python3
"""Evaluate train answers, including passport validity omitted by evaluate.py.

Usage: python validate_train.py train_answers.jsonl data/train_labels.jsonl
"""
import json
import sys
from pathlib import Path

from evaluate import FIELDS, norm
from postprocess import valid_passport


def load(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row["doc_id"]
            if doc_id in rows:
                raise ValueError(f"Duplicate doc_id {doc_id!r} in {path}:{line_no}")
            rows[doc_id] = row
    return rows


def main(answers_path, labels_path):
    answers, labels = load(answers_path), load(labels_path)
    missing = set(labels) - set(answers)
    extra = set(answers) - set(labels)
    if missing or extra:
        raise SystemExit(f"ID mismatch: missing={len(missing)}, extra={len(extra)}")

    accepted = correct = field_error = invalid_accepted = 0
    per_field = {field: 0 for field in FIELDS}
    for doc_id, answer in answers.items():
        if answer.get("decision") != "accept":
            continue
        accepted += 1
        truth = labels[doc_id]["fields"]
        predicted = answer.get("fields") or {}
        wrong = [field for field in FIELDS if norm(predicted.get(field)) != norm(truth.get(field))]
        for field in wrong:
            per_field[field] += 1
        if wrong:
            field_error += 1
        else:
            correct += 1
        if not valid_passport(truth.get("Birth_date"), truth.get("Issue_date")):
            invalid_accepted += 1

    print(f"documents:                  {len(labels)}")
    print(f"accepted:                   {accepted}")
    print(f"field-correct accepts:      {correct}")
    print(f"accepts with field errors:  {field_error}")
    print(f"accepted invalid passports: {invalid_accepted}")
    print("field errors: " + (", ".join(f"{k}={v}" for k, v in per_field.items() if v) or "none"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python validate_train.py <answers.jsonl> <labels.jsonl>")
    main(sys.argv[1], sys.argv[2])
