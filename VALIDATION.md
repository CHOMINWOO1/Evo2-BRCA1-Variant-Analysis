# Public-release validation

The following checks were performed while preparing this public copy:

- **6 tests passed** on Windows / Python 3.12, covering selected file hashes, Python syntax, primary-effect consistency, numerical-QC cohort boundaries, context-purge scope and the train-only research kernel.
- The kernel self-check exercises held-out feature/label isolation, direct-solve agreement, duplicate-kernel equivalence, constant features and zero-direction controls using synthetic inputs. It does not refit real cohorts or load model weights.
- All selected Python scripts compile. Historical GPU modules remain Linux-oriented; their runtime requirements were not emulated on Windows.
- Regenerated and visually inspected the four-panel result figure from checked-in aggregates. No synthetic numbers were substituted for experimental results.
- Selected original code and result file byte hashes are preserved in `docs/SOURCE_FILES.json`. The numerical-QC JSON is explicitly marked as a field projection.
- Secret scanning and private-path checks are applied to publication candidates. Runtimes, source Git history, keys, raw activations, logs and account/agent tooling are excluded.

The GitHub workflow runs the six public checks plus the two original region-coverage/candidate-selection tests on Linux with Python 3.11 and 3.12. The CI result is recorded separately after the first push.

Not rerun for this release: GPU setup, Evo2 inference, the 13,455-SNV screen, the corrected 193-SNV extraction, real-label probe fitting and bootstrap. These are historical results backed by the included aggregates and source-audit hashes. Full activation and phenotype artifacts are required for complete computational reproduction.

```bash
python -m pip install -r requirements-analysis.txt
python -m pytest -q
python scripts/analysis/brca1_signed_kernel.py
python scripts/reporting/render_portfolio_results.py --output runs/figures
python -m compileall -q scripts
```
