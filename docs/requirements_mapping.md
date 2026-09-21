# Unit 1 Requirements Mapping — Python Version

| Requirement | Implementation |
|---|---|
| Preserve original analysis | `original/original_analysis.Rmd`; first Git commit |
| Separate input/code/intermediate/final outputs/tests | `data/`, `src/`, `artifacts/`, `results/`, `tests/` |
| Meaningful version control | Multiple focused commits with descriptive messages |
| Record computational environment | `.venv/` workflow + pinned `requirements.txt` |
| Exact fresh-system setup commands | `README.md` → **Computational environment** |
| One documented reproduction command | `make reproduce` |
| Non-interactive execution | Four Python stage scripts with fixed repository-relative paths |
| Fixed randomness | Seed 123 for split, CV, and stochastic model settings |
| Progress/failure messages | Each stage prints progress and raises clear missing-input errors |
| At least two meaningful tests | `tests/test_functions.py` |
| Data validation | Required columns + valid binary churn target |
| Function correctness | Deterministic split + cleaning + training-only median imputation |
| Project purpose and outputs documented | `README.md` |
| Data source/access restrictions documented | `README.md` |
| Exact commands for reproduction/tests | `README.md` |
| Repository structure documented | `README.md` |
| Expected outputs and locations | `README.md` |
| Reflection | `docs/reflection.md` |

## Commands the grader should use

```bash
make setup
make test
make reproduce
```

Expected final report:

```text
results/report/churn_report.html
```
