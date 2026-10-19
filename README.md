# Otomoto Marketing Segmentation: ANN Optimization


## Overview
This project optimizes an artificial neural network (ANN) that segments telecom customers into likely-to-churn and likely-to-stay groups, so marketing can target retention campaigns. It compares several optimization algorithms on the same data split and reports accuracy, precision, recall, F1, ROC-AUC and loss.

## Files
| File | Description |
|---|---|
| `otomoto_ann_optimization.py` | Data preparation, baseline model, all optimizer experiments, and evaluation |
| `make_figures.py` | Creates the charts used in the report (run after the main script) |
| `results_all_runs.csv` | Test-set metrics for every model and random seed |
| `Otomoto_ANN_Optimization_Report.docx` | Full report: process, results, recommendations, references |
| `teleconnect.csv` | Customer dataset provided with the assignment (7,043 customers) |

## Models compared
- **Baseline:** 30-16-8-1 ANN trained with plain SGD (learning rate 0.01). This is a reconstruction of the existing model.
- **SGD + Momentum**, **RMSprop**, **Adagrad**, **Adam**
- **Adam (tuned):** Adam plus dropout, L2 regularization, early stopping, learning-rate decay, class weighting and a validation-chosen threshold

All models use the same split, loss (binary cross-entropy), batch size (64) and a 60-epoch budget. Each is trained with 3 random seeds.

## Results (test set, mean of 3 seeds)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline SGD | 0.794 | 0.633 | 0.534 | 0.579 | 0.836 |
| SGD + Momentum | 0.784 | 0.616 | 0.501 | 0.551 | 0.834 |
| RMSprop | 0.786 | 0.620 | 0.505 | 0.556 | 0.836 |
| Adagrad | 0.793 | 0.630 | 0.535 | 0.578 | 0.838 |
| Adam | 0.789 | 0.630 | 0.497 | 0.555 | 0.833 |
| Adam (tuned) | 0.788 | 0.591 | 0.654 | 0.621 | 0.841 |

**Key finding:** changing the optimizer alone made little difference. The tuned Adam model found far more churners (recall 65% vs 53%) with similar accuracy, at the cost of some precision.

## How to run
1. Install Python 3.9 to 3.12 and the libraries:
```
   pip install tensorflow scikit-learn pandas matplotlib
```
2. Keep `teleconnect.csv` in the same folder as the scripts.
3. Run the experiments (this takes several minutes on a CPU):
```
   python otomoto_ann_optimization.py
```
4. Create the charts:
```
   python make_figures.py
```

Results may differ slightly on other machines because of random initialization.

## Limitations
- The original Otomoto model was not provided, so the baseline is a reconstruction.
- Three seeds and a single train/test split give indicative rather than definitive differences.
- Training was capped at 60 epochs.

## Tools
Python, TensorFlow/Keras, scikit-learn, pandas, matplotlib.
