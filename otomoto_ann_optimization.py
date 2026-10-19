"""
Otomoto Marketing Segmentation - ANN Optimization (Module 6)
Predicts customer churn (segment: likely-to-churn vs likely-to-stay) from
the TeleConnect customer data, then compares optimization algorithms.
Run:  python otomoto_ann_optimization.py [path/to/teleconnect.csv]
"""
import os, sys, json, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

CSV = sys.argv[1] if len(sys.argv) > 1 else "teleconnect.csv"
OUT = "results"; os.makedirs(OUT, exist_ok=True)
SEEDS = [1, 2, 3]
EPOCHS, BATCH = 60, 64

# ---------------- 1. Data preparation ----------------
df = pd.read_csv(CSV)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)          # 11 new customers (tenure=0)
y = (df["Churn"] == "Yes").astype(int).values
X = df.drop(columns=["customerID", "Churn"])
X = pd.get_dummies(X, drop_first=True).astype("float32")
feature_names = list(X.columns)

X_tr, X_te, y_tr, y_te = train_test_split(X.values, y, test_size=0.2,
                                          stratify=y, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15,
                                            stratify=y_tr, random_state=42)
sc = StandardScaler().fit(X_tr)
X_tr, X_val, X_te = sc.transform(X_tr), sc.transform(X_val), sc.transform(X_te)
cw = dict(enumerate(compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)))
print(f"features={X.shape[1]} train={len(y_tr)} val={len(y_val)} test={len(y_te)} churn rate={y.mean():.3f}")

# ---------------- 2. Models ----------------
def build(n_in, hidden=(16, 8), dropout=0.0, l2=0.0, opt=None):
    reg = keras.regularizers.l2(l2) if l2 else None
    layers = [keras.layers.Input((n_in,))]
    for h in hidden:
        layers.append(keras.layers.Dense(h, activation="relu", kernel_regularizer=reg))
        if dropout: layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    m = keras.Sequential(layers)
    m.compile(optimizer=opt, loss="binary_crossentropy")
    return m

def metrics(model, X, y, thr=0.5):
    p = model.predict(X, verbose=0).ravel()
    pred = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), auc=roc_auc_score(y, p),
                loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), p

# Experiments. Baseline = "existing" model: 16-8-1 ANN, plain SGD (lr 0.01), no regularisation.
EXPERIMENTS = {
    "Baseline SGD":        dict(opt=lambda: keras.optimizers.SGD(0.01)),
    "SGD + Momentum":      dict(opt=lambda: keras.optimizers.SGD(0.01, momentum=0.9)),
    "RMSprop":             dict(opt=lambda: keras.optimizers.RMSprop(0.001)),
    "Adagrad":             dict(opt=lambda: keras.optimizers.Adagrad(0.01)),
    "Adam":                dict(opt=lambda: keras.optimizers.Adam(0.001)),
    "Adam (tuned)":        dict(opt=lambda: keras.optimizers.Adam(0.001), dropout=0.3, l2=1e-3,
                                early=True, lr_sched=True, class_weight=True, hidden=(32, 16)),
}

rows, histories, probs, cms = [], {}, {}, {}
for name, cfg in EXPERIMENTS.items():
    for seed in SEEDS:
        keras.utils.set_random_seed(seed)
        m = build(X_tr.shape[1], cfg.get("hidden", (16, 8)), cfg.get("dropout", 0), cfg.get("l2", 0), cfg["opt"]())
        cbs = []
        if cfg.get("early"):   cbs.append(keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss"))
        if cfg.get("lr_sched"):cbs.append(keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5))
        t0 = time.time()
        h = m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH,
                  verbose=0, callbacks=cbs, class_weight=cw if cfg.get("class_weight") else None)
        dt = time.time() - t0
        # decision threshold chosen on validation set (max F1) only for the tuned model; others use 0.5
        thr = 0.5
        if cfg.get("class_weight"):
            pv = m.predict(X_val, verbose=0).ravel()
            grid = np.arange(0.2, 0.8, 0.02)
            thr = grid[np.argmax([f1_score(y_val, (pv >= t).astype(int)) for t in grid])]
        res, p = metrics(m, X_te, y_te, thr)
        res.update(model=name, seed=seed, epochs_run=len(h.history["loss"]), train_time=dt, threshold=float(thr))
        rows.append(res)
        if seed == SEEDS[0]:
            histories[name] = {k: [float(v) for v in vals] for k, vals in h.history.items() if k in ("loss", "val_loss")}
            probs[name] = p.tolist()
            cms[name] = confusion_matrix(y_te, (p >= thr).astype(int)).tolist()
    r = pd.DataFrame([x for x in rows if x["model"] == name])
    print(f"{name:16s} acc={r.accuracy.mean():.4f} prec={r.precision.mean():.3f} rec={r.recall.mean():.3f} "
          f"f1={r.f1.mean():.3f} auc={r.auc.mean():.4f} loss={r.loss.mean():.4f} epochs={r.epochs_run.mean():.0f}")

pd.DataFrame(rows).to_csv(f"{OUT}/all_runs.csv", index=False)
json.dump(dict(histories=histories, probs=probs, cms=cms, y_te=y_te.tolist(), features=feature_names),
          open(f"{OUT}/artifacts.json", "w"))
