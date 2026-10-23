# Passport OCR Quality Control

Conservative post-processing for OCR results of Russian internal passports. The program does not train an OCR model; it checks already extracted fields and accepts a document only when all configured gates pass.

## What it does

- Checks that all 10 required visual fields are present and have high field/character confidence; missing, incomplete, or non-finite confidence values fail closed.
- Validates date and identifier formats.
- Checks passport validity on **2026-10-01** using the 20/45-year replacement rules and the 90-day grace period.
- Checks MRZ structure, birth-date check digit, sex, document-number check digit, and agreement between the MRZ name and the OCR surname/first name.
- Rejects suspicious geometry in long text fields when a tall field box contains only one text line.
- Checks the subdivision code against the supplied reference list.
- Emits one JSONL answer per input document. Rejected documents have no `fields` key.

The MRZ character mapping is empirical for the supplied OCR outputs and is **not** a general-purpose ICAO transliteration standard. Some conservative ambiguity rules are dataset-specific and must be validated on an untouched validation split before production use.

## Requirements

- Python 3.10+
- No third-party packages; CPU-only and no network access required at runtime.

Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Input files

Place the employer-provided files locally in `data/`:

- `data/train_predictions.jsonl`
- `data/train_labels.jsonl`
- `data/test_predictions.jsonl`

The reference code list belongs in `reference/subdivisions.csv`. Input data are excluded from Git by `.gitignore`; do not publish them unless the owner explicitly authorizes distribution.

## Run

Create the submission file for the unlabeled test set:

```powershell
python postprocess.py data/test_predictions.jsonl answers.jsonl
```

Create and evaluate a separate train output:

```powershell
python postprocess.py data/train_predictions.jsonl train_answers.jsonl
python evaluate.py train_answers.jsonl data/train_labels.jsonl
python validate_train.py train_answers.jsonl data/train_labels.jsonl
```

`evaluate.py` checks exact field agreement but does not validate passport validity. `validate_train.py` adds a validity check based on the labeled dates.

Run unit tests:

```powershell
python -m unittest discover -s tests -v
```

## Results from the supplied training set

The initial baseline accepted 127 of 3,000 documents and had 6 accepted documents with field errors. The conservative updated version accepted 109 of 3,000 and had no field errors among accepted documents in this training diagnostic. This is not an independent validation result: the ambiguity rules were refined after inspecting training errors, so the zero-error result is optimistic and must not be interpreted as a guarantee on unseen documents.

The updated code accepts 135 of 2,000 test documents. The true number of erroneous accepts on test is unknown because test labels were not supplied. A naive point estimate based on the training result is zero, but this is not a guarantee: the rules were refined after inspecting training errors. Even with zero errors among 109 accepted training examples, a simple one-sided 95% binomial upper bound is about 2.7% only under independent, representative sampling and a fixed-in-advance algorithm; those assumptions are not met here. See `REPORT.md` for the caveat.

## Privacy and submission

`answers.jsonl` is the required task artifact and contains recognized identity fields for accepted documents. Keep the repository private unless the dataset owner confirms that these values are synthetic or may be published. Do not commit source images, raw predictions, labels, credentials, or unrelated local files.
