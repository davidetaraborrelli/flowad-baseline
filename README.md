# FlowAD-Baseline

A tiny, clean **baseline for anomaly / attack detection on network flows** with Python and scikit-learn.
It trains classic models (Logistic Regression, Random Forest, One-Class SVM, Isolation Forest) and
produces **ROC/PR curves**, **FPR @ fixed TPR**, and a reproducible `metrics.json`.

## TL;DR
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Quick demo with synthetic data (no dataset needed yet)
python src/train.py --fast

# "Real" run: place a CSV under data/raw and set configs/default.yaml
python src/train.py --config configs/default.yaml
```

## Data format
- Expect a CSV with numeric/categorical columns and one label column `Label` with values like `BENIGN`/`ATTACK` (you can change names in `configs/default.yaml`).
- If you don't have a dataset yet, the `--fast` flag creates **synthetic** flows, just to showcase the pipeline and the plots.

## Outputs
- `results/metrics.json` key metrics per model (ROC-AUC, PR-AUC)
- `results/thresholds.json` threshold chosen at **TPR=0.9** with the corresponding FPR
- `results/plots/roc_curve.png`, `results/plots/pr_curve.png`

## Acceptance (MVP)
- One command that runs end-to-end on a subset in <10 min
- README
- ROC/PR plots and a metrics/thresholds JSON

### Results (demo)
| Model                | ROC-AUC | PR-AUC | FPR @ TPR=0.9 |
|----------------------|--------:|-------:|--------------:|
| Logistic Regression  |  0.98   |  0.95  |      0.04     |
| Random Forest        |  0.98   |  0.93  |      0.06     |
| One-Class SVM        |  0.51   |  0.22  |      0.98     |
| Isolation Forest     |  0.88   |  0.63  |      0.26     |


## Project structure
```
repo/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ data.py
│  ├─ model.py
│  ├─ metrics.py
│  ├─ plot.py
│  └─ train.py
├─ data/
│  ├─ raw/
│  └─ processed/
└─ results/
   └─ plots/
```

## License
MIT


