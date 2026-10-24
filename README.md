# Bearing Remaining Useful Life from Vibration Data

**MMA3001 Numerical Methods and Machine Learning, individual project**
Author: Ryan · Data: NASA/IMS bearing run-to-failure dataset · Language: Python 3 (Google Colab)

This repository predicts **how many operating hours a rolling element bearing has left** (its remaining useful life, RUL) from the vibration it produces. It compares that prediction against the manufacturer's catalogue rating life (L10). Every result in the project report can be regenerated from the code and notebooks here.

**Headline result.** On the test bearing, which failed after 164 h, the catalogue L10 (1571 h) was wrong by about 1400 h. Vibration-based models were wrong by 15 to 80 h. On bearings the models had never seen, support vector regression (SVR) trained on two bearings was the most accurate (61.7 h), beating a "predict the average" baseline by 61%. Adding a fourth bearing (test 3), SVR was the only model whose error fell (186 h to 144 h), and it predicted the new bearing best (111 h against 191 h for the baseline). All models are still late too often to be used without a safety margin (see [Limitations](#limitations)).

---

## Contents

1. [The engineering problem](#the-engineering-problem)
2. [Data](#data)
3. [Inputs and outputs](#inputs-and-outputs)
4. [Method](#method)
5. [Repository structure](#repository-structure)
6. [Quick start (Google Colab)](#quick-start-google-colab)
7. [Reproducing the results, step by step](#reproducing-the-results-step-by-step)
8. [Tests and documentation](#tests-and-documentation)
9. [Results](#results)
10. [Limitations](#limitations)
11. [Common errors and fixes](#common-errors-and-fixes)
12. [Committing from Colab](#committing-from-colab)
13. [Use of AI](#use-of-ai)
14. [References and licence](#references-and-licence)

---

## The engineering problem

Rolling element bearings are in almost every rotating machine, from pumps and gearboxes to aircraft transmissions, and they are a leading cause of unplanned downtime. Maintenance is usually planned from the catalogue rating life **L10**: the hours that 90% of a large population of identical bearings survive (ISO 281). L10 describes the population, not the bearing in front of you.

**Question:** given the vibration measured on a bearing so far, how many hours does it have left?

**Intended use:** decision support for a maintenance engineer, to schedule an inspection or replacement shortly before failure. It is not an autonomous safety system.

**Scope:** fixed speed and load (as in the dataset), time-domain features only (RMS and kurtosis), and simple, explainable methods taught in the unit.

## Data

**NASA/IMS bearing dataset** (Center for Intelligent Maintenance Systems, University of Cincinnati; NASA Prognostics Data Repository).

- Four Rexnord ZA-2115 double-row roller bearings on one shaft, 2000 rpm, 6000 lb radial load.
- One accelerometer per bearing (two per bearing in test 1).
- Each file is a 1 s snapshot of **20,480 samples** (20,480 Hz, not the 20 kHz in the dataset readme), tab-separated, acceleration in g. File names are timestamps such as `2004.02.12.10.32.39`.

| Test | Snapshots | Duration | Channels | Failures used here |
| --- | --- | --- | --- | --- |
| 2nd_test | 984, every 10 min | 164 h | 4 | Bearing 1, outer race (main development data) |
| 1st_test | 2156, with gaps | about 830 h | 8 | Bearing 3, inner race (channel 5); bearing 4, roller (channel 7) |
| 3rd_test | 6324, every 10 min (the IMS readme says 4448) | about 1073 h | 4 | Bearing 3, outer race (channel 3): fourth bearing, Data Analysis cells 20 to 22 |

**The raw data (about 6 GB) is not in this repository.** Download it from the NASA Prognostics Data Repository and put it on Google Drive. The code expects:

```
/content/drive/MyDrive/MMA3001/MMA3001_Project/IMS/IMS/2nd_test/2nd_test/
/content/drive/MyDrive/MMA3001/MMA3001_Project/IMS/IMS/1st_test/1st_test/
```

The **feature tables** computed from the raw data *are* in `data/` (a few hundred kB), so all modelling notebooks run without the raw files.

## Inputs and outputs

| Item | Format | Units | Valid range | Meaning |
| --- | --- | --- | --- | --- |
| Raw snapshot (input) | Tab-separated text, 20,480 rows × 4 or 8 channels | g | One file per snapshot | Raw vibration |
| Feature table | `data/features_<test>.csv`, one row per snapshot, index `sample`, column `file`, then `rms_k`, `kurtosis_k`, `peak_k` per channel k | g; kurtosis dimensionless | RMS > 0; kurtosis ≈ 3 when healthy | Condition indicators |
| Model features | Trailing mean of RMS and kurtosis over the last 1 to 96 snapshots | as above | Past data only | Noise-filtered indicators |
| Predicted RUL (output) | Array, one value per snapshot | hours | Clipped to [0, longest RUL seen in training] | Hours to failure |
| Scores (output) | `data/rul_results_*.csv` | hours, % | MAE, RMSE ≥ 0; late % 0 to 100 | Method comparison |

**Invalid or missing input handling**

- Missing folder: `FileNotFoundError` with the path.
- Empty, truncated or non-numeric file: `ValueError`. During feature extraction the file is skipped with a message, so its `sample` number is missing from the table.
- Google Drive duplicates (`name (1)`): ignored when listing files.
- Constant signal (dead sensor): kurtosis returns NaN, and NaN rows are dropped before modelling.
- Recording gaps: found by `data_quality.time_gaps`. Time axes are built from file timestamps, not file counts.

Function-level inputs, outputs and exceptions are in the HTML documentation in `docs/`.

## Method

| Step | What it does | Course link |
| --- | --- | --- |
| Feature extraction (`bearing_features.py`) | RMS and kurtosis per snapshot and channel | Week 1 (testing), Week 2 (vectorisation) |
| Data quality (`data_quality.py`) | Gaps, wrong lengths, NaNs, flat or clipped channels | Week 1.2 |
| Spectrum (`spectrum.py`) | FFT with bearing defect frequencies (BPFO 236.4 Hz) to confirm the failure mode | Engineering justification |
| Noise filtering (`rul_models.trailing_mean`) | Backward-looking moving average, window 1 to 96 snapshots (no future data leaks in) | Week 3.2, Week 5.6 |
| Baseline (`rul_models.catalogue_life_hours`) | L10 = (C/P)^(10/3) × 10⁶ / (60 n) = 1571 h; L50 via a Weibull distribution | ISO 281 |
| Trend extrapolation (`extrapolate_rul_fast`) | Least-squares line through recent RMS, extrapolated to a failure threshold | Week 3.1, Week 5.1, Week 2.2 |
| Regression models | Linear, decision tree, SVR (scikit-learn), plus a mean-value Dummy baseline | Week 5.1 to 5.5 |
| Validation (`splits.py`, `tune_and_test`) | Time, interleaved and random-block splits (60/20/20); tune on validation, score test once; leave one bearing out | Week 5.5, 5.6 |
| Performance (`perf_cells`) | `line_profiler`, `timeit`, FLOPs and arithmetic intensity, O(n²) to O(n) rewrite | Week 2.1 |

## Repository structure

```
MMA3001-Project/
├── README.md                 this file
├── LICENSE
├── requirements.txt          package versions used on Colab
├── pytest.ini                lets plain `pytest` find src/
├── src/                      importable modules (NumPy-style docstrings)
│   ├── bearing_features.py   load snapshots, compute RMS / kurtosis / peak
│   ├── data_quality.py       timestamps, gaps, per-snapshot quality checks
│   ├── spectrum.py           FFT and bearing defect frequencies
│   ├── trends.py             smoothing (moving average, spline, KNN) and health stages
│   ├── rul_models.py         baseline, trend extrapolation, model tuning and scoring
│   └── splits.py             train / validation / test split methods
├── tests/                    pytest unit tests, one file per module
├── test_report.txt           output of the latest pytest run
├── docs/                     HTML documentation generated by pdoc (open docs/index.html)
├── notebooks/                Main.ipynb (data preparation, performance), Data Analysis.ipynb (RUL modelling)
└── data/                     feature tables and result tables (CSV)
```

## Quick start (Google Colab)

1. Open a notebook from `notebooks/` in Colab (File > Open notebook > GitHub, then paste the repository URL).
2. Run the **setup cell**. It mounts Google Drive, clones this repository to `/content/repo` and adds `src/` to the Python path:

```python
from google.colab import drive, userdata
drive.mount("/content/drive")
try:
    token = userdata.get("GH_PAT") + "@"              # only needed to push changes
except Exception:
    token = ""                                        # no secret: read-only clone works
!git clone -q https://{token}github.com/Ryan-T-Monash/MMA3001-Project.git /content/repo
%cd /content/repo
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, "/content/repo/src")
```

Without a `GH_PAT` secret the setup cell still clones the public repository, so you can run everything; you just cannot push. The modelling notebook runs from the CSV files in `data/`, so the raw IMS files are only needed to rebuild features. In `Main.ipynb`, the data quality, spectrum and performance (P2 to P8) sections read the raw IMS files, so download the NASA IMS data and set `PROJECT` in the setup cell first. Everything in `Data Analysis.ipynb` runs from `data/`.

3. Try the modules:

```python
import pandas as pd
from rul_models import catalogue_life_hours
feat = pd.read_csv("data/features_2nd_test.csv", index_col=0)
print(feat[["rms_1", "kurtosis_1"]].describe())
print(f"Catalogue L10: {catalogue_life_hours():.0f} h")       # 1571 h
```

## Reproducing the results, step by step

Run the cells **in order, top to bottom**. Most cells use variables made by earlier cells, so after a restart use **Runtime > Run before**.

| Notebook | Cells | What it produces |
| --- | --- | --- |
| Data preparation (`Main.ipynb`) | Setup, features, data quality, FFT | `data/features_2nd_test.csv`, spectra confirming the outer race fault |
| | P1 to P8 (performance) | Profile, timing, FLOPs, scaling plot, `data/performance_2nd_test.csv` |
| RUL modelling (`Data Analysis.ipynb`) | 1 to 3 | Settings and test 2 features |
| | 4 to 10 | Catalogue baseline, splits, tuning on validation, test scores, overfitting plots, `data/rul_results_2nd_test.csv` |
| | 11 to 12 | Sensitivity plots and the final comparison table, `data/final_comparison_2nd_test.csv` |
| | 13 to 16 | Test 1 features (read from `data/`, rebuilt from raw files only if missing), unseen-bearing test, `data/rul_results_unseen_bearing.csv` |
| | 17 to 18 | Train on two bearings, test on the third, `data/rul_results_leave_one_bearing_out.csv` |
| | 19 | Safety margin against late predictions, `data/rul_safety_margin.csv` |
| | 20 to 22 | Test 3 as a fourth bearing (needs Main section 2b once): train on three, test on the fourth, `data/rul_results_four_bearings.csv`, `data/rul_training_bearings_summary_4.csv` |

All tunable settings (smoothing windows, tree depths, SVR C, extrapolation windows, failure thresholds, split ratio, seeds) are in **cell 2** of `Data Analysis.ipynb`.

## Tests and documentation

```python
%cd /content/repo
!pytest tests -v                      # 48 tests (pytest.ini adds src/ to the path)
!pytest tests -v > test_report.txt    # save the report
!pdoc --math ./src/*.py -o ./docs                     # rebuild the HTML docs
```

The tests use inputs with known exact answers, not real data. For example: the RMS of a unit sine is 1/√2, Gaussian kurtosis is 3, L10 matches a hand calculation, extrapolation is exact for a straight line, and every optimised function matches its original version.

## Results

| Method | Test 2, time split (h) | Unseen bearings, 1 training bearing (h) | Unseen bearings, 2 training bearings (h) |
| --- | --- | --- | --- |
| **SVR (selected)** | 79.8 | 37.0 | **61.7** |
| Decision tree | 49.1 | 43.1 | 90.2 |
| Linear regression | **14.8** | 38.3 | 134.5 |
| Mean (Dummy) | 98.3 | **34.9** | 159.1 |
| Trend extrapolation | 51.0 | 68.3 | 388.4 |
| Catalogue L10 | 1407 | 743 | 743 |

Values are mean absolute error in hours. Unseen-bearing columns cover the final 164 h of test 1 bearings 3 and 4. SVR is selected because the intended use is bearings the model has not seen. Linear regression is best only for forecasting the same bearing it was trained on.

**Fourth bearing (test 3 bearing 3, outer race, 1073 h).** Trained on the other three bearings, MAE over the final 164 h: SVR 110.7 h, Dummy 191.4 h, linear 201.7 h, tree 216.8 h, trend extrapolation 366.7 h, catalogue L10 497.6 h. Averaged over the three original bearings, going from two to three training bearings changed SVR from 185.8 h to 144.2 h, linear from 201.0 to 204.3 h, the tree from 207.4 to 317.9 h and the Dummy from 184.7 to 359.2 h.

**Performance:** rewriting the trend extrapolation with running sums made it **440× faster** (85.8 ms to 0.195 ms per setting) with identical results. Vectorising the feature maths gained only 1.05× on Colab, because reading files from Drive dominates the run time.

## Limitations

- Only four failure histories, with three fault types. With one training bearing, no model beat the Dummy on new bearings.
- 75 to 90% of predictions on new bearings are late. A safety margin is needed before any real use.
- Bearings under identical conditions lived 164 h and about 830 h. A model trained only on long lives over-predicts short ones.
- Fixed speed and load; failure time is taken as the end of each test.

## Common errors and fixes

**Running cells**

| Error | Cause | Fix |
| --- | --- | --- |
| `NameError: name 'splits' is not defined` (or `time`, `extrap`, `models`, `fixed_validation_curve`) | A cell was run before the cells that create its variables, often after a restart | Select the cell, then **Runtime > Run before**, then run it |
| `ImportError: cannot import name 'snapshot_features_fast'` (or another new function) | The repository's `src/` still has an old copy of the module, or Python cached the old one | Update the `.py` file in `src/` (or on Drive and rerun the copy cell), then **Runtime > Restart session** and rerun |
| `ModuleNotFoundError: No module named 'rul_models'` | `src/` is not on the Python path | Rerun the setup cell, or `import sys; sys.path.insert(0, "/content/repo/src")` |
| `SyntaxError` on a line of plain words | Part of a comment was lost when pasting (the `#` was cut off) | Paste each cell starting from its `# ---- Colab cell` line |
| `FileNotFoundError: Folder not found` | Wrong data path, or Drive not mounted | Run `drive.mount("/content/drive")` and check the path in the setup cell |
| Folder named `2nd_test (1)` | Google Drive renamed a re-uploaded folder | Rename it back to `2nd_test` in Drive |
| `Skipped <file>: Empty snapshot file` | A raw file was truncated during upload | Harmless for one or two files. For many, re-upload the folder |
| Feature extraction is very slow | Reading 1 to 2 GB of text from Drive (about 20 ms per file) | Use the cached `data/features_*.csv`; only rebuild when the raw data changes |
| `ModuleNotFoundError: line_profiler` / `pdoc` | Not installed on a fresh runtime | `!pip install -q line_profiler pdoc` (or `-r requirements.txt`) |

**Git and GitHub**

| Error | Cause | Fix |
| --- | --- | --- |
| `untracked working tree files would be overwritten by merge` | A file was copied into the repo locally that was also committed from another notebook | Commit first, then pull: `!git add -A && git commit -m "msg"` then `!git pull --rebase -X theirs origin main` |
| `! [rejected] main -> main (non-fast-forward)` | GitHub has commits that this copy lacks (for example pushed from the other notebook) | `!git pull --rebase -X theirs origin main` then `!git push origin main` |
| `CONFLICT (content)` during a rebase | The same lines were edited in two places | Fix the marked file, `!git add <file>`, `!git rebase --continue`, or `!git rebase --abort` to undo |
| `Authentication failed` / `403` on push | The GitHub token (`GH_PAT`) is missing, expired, or not enabled for this notebook | Create a new fine-grained token with *Contents: read and write* on this repo, save it in Colab **Secrets** as `GH_PAT`, turn on notebook access, and re-clone |
| `fatal: destination path '/content/repo' already exists` | The setup cell was run twice | Harmless; or `!rm -rf /content/repo` and rerun the setup cell |
| `nothing to commit, working tree clean` | No files changed since the last commit | Nothing to fix |
| Colab "Save a copy in GitHub" shows `Unexpected response from GitHub` | Colab's GitHub sign-in failed or expired | Revoke *Colaboratory* at github.com/settings/applications and sign in again (tick "Include private repos" if the repo is private), or use the publish cell below |
| Author shows as `root` in commits | Git identity not set in the runtime | `!git config --global user.name "Your Name"` and `user.email` |

## Committing from Colab

All commits are made from Colab with this **publish cell** (run it at the end of a session). It commits first and then pulls, so local files never block the pull.

```python
MSG = "Describe what changed"
NB = "Main.ipynb"                                   # or "Data Analysis.ipynb"
%cd /content/repo
!mkdir -p notebooks && cp "/content/drive/MyDrive/Colab Notebooks/{NB}" notebooks/
!pdoc --math ./src/*.py -o ./docs
!pytest tests -v > test_report.txt
!git add -A
!git commit -m "{MSG}"
!git pull --rebase -X theirs origin main
!git push origin main
```

Press **Ctrl+S** first so Drive has the latest copy of the notebook.

## Use of AI

Anthropic's Claude was used to break down the brief, draft modules, tests and Colab cells from the author's specifications, debug Colab and Git problems, and draft the report and this README. All code was verified by the unit tests and by checking results against physical constraints, and performance claims were re-measured on Colab. The engineering problem, baseline, validation design, simplifications and final model choice were the author's decisions. The full AI reflection is in the project report.

## References and licence

- Lee, J., Qiu, H., Yu, G. and Lin, J. (2007). *IMS Bearing Data Set*. NASA Prognostics Data Repository, NASA Ames Research Center.
- Qiu, H., Lee, J., Lin, J. and Yu, G. (2006). Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics. *Journal of Sound and Vibration*, 289(4–5), 1066–1090.
- ISO 281:2007, *Rolling bearings: Dynamic load ratings and rating life*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

Code is released under the licence in [LICENSE](LICENSE). The IMS data belongs to its original authors and is not redistributed here.
