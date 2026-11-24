#!/usr/bin/env python3
"""
AI-Powered Phishing URL Detection and Classification System
Training and evaluation pipeline (steps 1-8 of the implementation plan).

Base paper : Aljofey et al., "An Effective Phishing Detection Model Based on Character
             Level Convolutional Neural Network from URL", Electronics 2020, 9, 1514.
Primary data: PhiUSIIL Phishing URL dataset (only the URL string and the label are used).

Usage (see README.md):
    python phishing_pipeline.py prepare --data PhiUSIIL_Phishing_URL_Dataset.csv --out results
    python phishing_pipeline.py exp1 --out results
    python phishing_pipeline.py exp3 --out results
    python phishing_pipeline.py exp4 --out results
    python phishing_pipeline.py exp5 errors latency tables --out results
"""
import argparse
import gc
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
SEED = 42
MAXLEN = 200            # maximum input length L (base paper)
EMB_DIM = 95            # embedding dimension K (base paper)

# --------------------------------------------------------------------------------------
# Character dictionary (Table 1 of the paper): a-z 1-26, A-Z 27-52, 0-9 53-62, 32 symbols
# 63-94, UNK = 95, padding = 0  ->  96 indices in total.
# --------------------------------------------------------------------------------------
SPECIALS = ",;.!?:\"'/\\|_@#$%^&*~`+-=<>()[]{}"
assert len(SPECIALS) == 32
CHAR2IDX = {}
for _i, _c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    CHAR2IDX[_c] = 1 + _i
for _i, _c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    CHAR2IDX[_c] = 27 + _i
for _i, _c in enumerate("0123456789"):
    CHAR2IDX[_c] = 53 + _i
for _i, _c in enumerate(SPECIALS):
    CHAR2IDX[_c] = 63 + _i
UNK_IDX = 95
VOCAB_SIZE = 96

MODELS_CLASSIC = ["MNB", "LR", "GNB", "RF", "XGB"]
MODELS_DEEP_FG = ["DNN", "CNN"]                      # Experiment 1 deep models (all FGs)
MODELS_EXP3 = ["RNN", "RCNN", "DNN", "VDCNN", "CNN"]  # Experiment 3 (FG2)
FGS = ["FG1", "FG2", "FG3", "FG4"]
MODEL_LABEL = {"MNB": "Multinomial Naive Bayes", "LR": "Logistic Regression", "GNB": "Gaussian Naive Bayes",
               "RF": "Random Forest", "XGB": "eXtreme Gradient Boosting", "DNN": "Deep Neural Networks",
               "CNN": "Convolutional Neural Networks", "RNN": "Recurrent Neural Networks",
               "RCNN": "Recurrent Convolutional NN", "VDCNN": "Very Deep CNN"}
SENSITIVE = ["login", "signin", "log-in", "register", "verify", "verification", "account", "secure",
             "update", "bank", "password", "confirm", "webscr", "paypal", "wallet", "support"]



# ======================================================================================
# 1. DATA: loading, cleaning, leakage audit, splits
# ======================================================================================
_PREFIX = re.compile(r"^(?:(?:https?|ftp)://)?(?:www\.)?", re.I)


def normalise_url(u):
    """Remove 'http://', 'https://' and 'www.' (as in the base paper)."""
    return _PREFIX.sub("", str(u).strip())


try:
    import tldextract
    _TLDX = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)   # bundled snapshot, offline
except Exception:  # pragma: no cover
    _TLDX = None


def host_of(u):
    return u.split("/")[0].split("?")[0].split("#")[0].split("@")[-1].split(":")[0]


def registered_domain(u):
    h = host_of(u)
    if _TLDX is not None:
        try:
            return _TLDX(h).registered_domain or h
        except Exception:
            pass
    p = h.split(".")
    return ".".join(p[-2:]) if len(p) >= 2 else h


def tld_of(u):
    h = host_of(u)
    if _TLDX is not None:
        try:
            return _TLDX(h).suffix or "none"
        except Exception:
            pass
    return h.rsplit(".", 1)[-1] if "." in h else "none"


def load_phiusiil(path, label_is_legit=True):
    df = pd.read_csv(path, usecols=lambda c: c.lower() in ("url", "label"))
    df.columns = [c.lower() for c in df.columns]
    y = df["label"].astype(int).values
    if label_is_legit:          # PhiUSIIL documentation: 1 = legitimate, 0 = phishing -> flip
        y = 1 - y
    return pd.DataFrame({"url_raw": df["url"].astype(str), "y": y})


def _format_audit(urls, y, raw):
    """Check whether the classes differ in trivial formatting; if a model on these features
    alone separates the classes, the dataset has formatting artefacts."""
    s = pd.Series(urls)
    feats = pd.DataFrame({
        "trailing_slash": s.str.endswith("/").astype(int),
        "has_upper": s.str.contains(r"[A-Z]").astype(int),
        "length": s.str.len()})
    if raw:
        feats["https"] = s.str.lower().str.startswith("https://").astype(int)
        feats["http"] = s.str.lower().str.startswith("http://").astype(int)
        feats["www"] = s.str.lower().str.contains(r"^(?:https?://)?www\.").astype(int)
    yv = np.asarray(y)
    rates = {c: {"legit": float(feats.loc[yv == 0, c].mean()), "phish": float(feats.loc[yv == 1, c].mean())} for c in feats.columns}
    auc = None
    if len(set(yv)) == 2 and len(feats) > 50:
        Xf = MinMaxScaler().fit_transform(feats.values.astype(float))
        a, b, ya, yb = train_test_split(Xf, yv, test_size=0.3, random_state=SEED, stratify=yv)
        auc = float(roc_auc_score(yb, LogisticRegression(max_iter=300).fit(a, ya).predict_proba(b)[:, 1]))
    return {"group_means": rates, "format_only_auc": auc}


def clean_dataset(df, name):
    """De-duplicate, normalise, audit. Returns the clean frame and the statistics dict."""
    st = {"dataset": name, "raw_rows": int(len(df)),
          "raw_class_counts": {"benign": int((df.y == 0).sum()), "phishing": int((df.y == 1).sum())}}
    df = df.dropna(subset=["url_raw"]).copy()
    df["url_raw"] = df["url_raw"].astype(str).str.strip()
    st["leakage_audit_raw"] = _format_audit(df.url_raw.values, df.y.values, raw=True)
    n0 = len(df)
    df = df.drop_duplicates(subset=["url_raw"]).reset_index(drop=True)
    st["raw_duplicate_urls_removed"] = int(n0 - len(df))            # expected 425 for PhiUSIIL
    df["url"] = df["url_raw"].map(normalise_url)
    conflict = df.groupby("url")["y"].nunique()
    bad = set(conflict[conflict > 1].index)
    st["conflicting_label_urls_removed"] = int(df.url.isin(bad).sum())
    df = df[~df.url.isin(bad)]
    n1 = len(df)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    st["duplicates_after_normalisation_removed"] = int(n1 - len(df))
    df = df[df.url.str.len() > 0].reset_index(drop=True)
    st["final_rows"] = int(len(df))
    st["final_class_counts"] = {"benign": int((df.y == 0).sum()), "phishing": int((df.y == 1).sum())}
    st["leakage_audit_normalised"] = _format_audit(df.url.values, df.y.values, raw=False)
    if st["final_class_counts"]["phishing"] > st["final_class_counts"]["benign"]:
        print("WARNING: phishing is the majority class; PhiUSIIL has more legitimate URLs. "
              "Check the label direction (--label-is-legit / --label-is-phish).")
    df["domain"] = df.url.map(registered_domain)
    return df[["url", "y", "domain", "url_raw"]], st


def make_splits(df, domain_split=True):
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=df.y.values)
    tr, va = train_test_split(tr, test_size=0.10, random_state=SEED, stratify=df.y.values[tr])
    out = {"tr": tr, "va": va, "te": te}
    if domain_split:
        g = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        dtr, dte = next(g.split(idx, df.y.values, groups=df.domain.values))
        dtr, dva = train_test_split(dtr, test_size=0.10, random_state=SEED)
        out.update({"dtr": dtr, "dva": dva, "dte": dte})
    return out


def cmd_prepare(a):
    """Prepare the single PhiUSIIL dataset and its train/validation/test splits."""
    if not a.data:
        raise ValueError("Pass --data with the PhiUSIIL CSV path.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_phiusiil(a.data, label_is_legit=not a.label_is_phish)
    df, stats = clean_dataset(raw, "PhiUSIIL")
    splits = make_splits(df, domain_split=True)
    for key, idx in splits.items():
        labels = df.y.values[idx]
        stats[f"split_{key}"] = {
            "rows": int(len(idx)),
            "benign": int((labels == 0).sum()),
            "phishing": int((labels == 1).sum())
        }
    stats["domain_overlap_stratified_split"] = int(
        len(set(df.domain.values[splits["tr"]]) & set(df.domain.values[splits["te"]]))
    )
    stats["domain_overlap_domain_split"] = int(
        len(set(df.domain.values[splits["dtr"]]) & set(df.domain.values[splits["dte"]]))
    )
    df.to_csv(out / "clean_PhiUSIIL.csv", index=False)
    np.savez(out / "splits_PhiUSIIL.npz", **splits)
    report = {"PhiUSIIL": stats, "environment": environment_info()}
    with open(out / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"PhiUSIIL": {k: v for k, v in stats.items() if not k.startswith("leak")}}, indent=2))


def environment_info():
    info = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    import sklearn
    info["scikit-learn"] = sklearn.__version__
    for mod in ("xgboost", "tensorflow"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    return info


def load_prepared(out, name="PhiUSIIL"):
    out = Path(out)
    df = pd.read_csv(out / f"clean_{name}.csv", keep_default_na=False)
    z = np.load(out / f"splits_{name}.npz")
    return df, {k: z[k] for k in z.files}


# ======================================================================================
# 2. FEATURES: FG1 char TF-IDF, FG2 char embedding ids, FG3 char counts, FG4 hand-crafted
# ======================================================================================
SYM17 = [".", "-", "_", "/", "?", "=", "@", "&", "!", " ", "+", "~", ",", "*", "#", "$", "%"]
SYM_HOST = [s for s in SYM17 if s != "+"]                       # 16 symbols for the host
TLD_RE = re.compile(r"\.(?:com|org|net|edu|gov|mil|int|info|biz|io|co|us|uk|de|cn|ru|in|br|jp|fr|au|ca|it|es|nl|xyz|top|site|online|club|app|dev|ly|me|tv|cc|ws|tk|ml|ga|cf|gq)(?![a-z0-9])", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$|^0x[0-9a-f]+$", re.I)


def _names():
    n = []
    for part, syms in (("U", SYM17), ("H", SYM_HOST), ("P", SYM17), ("F", SYM17), ("Q", SYM17)):
        n += [f"{part}_count[{s}]" for s in syms] + [f"{part}_length"]
        if part == "U":
            n += ["U_email_exist", "U_count_tld"]
        if part == "H":
            n += ["H_vowels", "H_ip_exist"]
        if part == "Q":
            n += ["Q_number_params", "Q_count_tld"]
    return n


FG4_NAMES = _names()       # 95 names: U 20 + H 19 + P 18 + F 18 + Q 20


def handcrafted_one(u):
    rest, query = (u.split("?", 1) + [""])[:2] if "?" in u else (u, "")
    rest = rest.split("#", 1)[0]
    if "/" in rest:
        host, path = rest.split("/", 1); path = "/" + path
    else:
        host, path = rest, ""
    file = path.rsplit("/", 1)[-1] if path else ""
    pathonly = path[: len(path) - len(file)]
    f = [u.count(s) for s in SYM17] + [len(u), 1 if EMAIL_RE.search(u) else 0, len(TLD_RE.findall(u))]
    h = host.split(":")[0]
    f += [host.count(s) for s in SYM_HOST] + [len(host), sum(host.lower().count(v) for v in "aeiou"), 1 if IPV4_RE.match(h) else 0]
    f += [pathonly.count(s) for s in SYM17] + [len(pathonly)]
    f += [file.count(s) for s in SYM17] + [len(file)]
    f += [query.count(s) for s in SYM17] + [len(query), (query.count("&") + 1) if query else 0, len(TLD_RE.findall(query))]
    return f


def handcrafted(urls):
    arr = np.array([handcrafted_one(u) for u in urls], dtype=np.float32)
    assert arr.shape[1] == 95, arr.shape
    return arr


def encode_urls(urls, maxlen=MAXLEN):
    arr = np.zeros((len(urls), maxlen), dtype=np.int32)
    for i, u in enumerate(urls):
        s = u[:maxlen]
        arr[i, : len(s)] = [CHAR2IDX.get(c, UNK_IDX) for c in s]
    return arr


class FeatureBuilder:
    """Fit on the training URLs only; transform any split."""

    def __init__(self, fg, max_features=5000, ngram=(1, 3)):
        self.fg, self.max_features, self.ngram = fg, max_features, ngram
        self.vec = None; self.scaler = None

    def fit(self, urls):
        if self.fg == "FG1":
            self.vec = TfidfVectorizer(analyzer="char", ngram_range=self.ngram, max_features=self.max_features, lowercase=False, dtype=np.float32)
            self.vec.fit(urls)
        elif self.fg == "FG3":
            self.vec = CountVectorizer(analyzer="char", ngram_range=self.ngram, max_features=self.max_features, lowercase=False, dtype=np.float32)
            self.vec.fit(urls)
        elif self.fg == "FG4":
            self.scaler = MinMaxScaler().fit(handcrafted(urls))
        return self

    def transform(self, urls, embed=False):
        if self.fg in ("FG1", "FG3"):
            return self.vec.transform(urls).astype(np.float32)
        if self.fg == "FG2":
            ids = encode_urls(urls)
            return ids if embed else (ids.astype(np.float32) / UNK_IDX)
        return np.clip(self.scaler.transform(handcrafted(urls)), 0, 1).astype(np.float32)


# ======================================================================================
# 3. MODELS
# ======================================================================================
def to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


class BatchedGNB(BaseEstimator, ClassifierMixin):
    """Gaussian NB that never densifies the whole sparse matrix (5000 features x 190k rows)."""

    def __init__(self, batch=4000):
        self.batch = batch

    def fit(self, X, y):
        self.m_ = GaussianNB(var_smoothing=1e-6)
        y = np.asarray(y)
        for i in range(0, X.shape[0], self.batch):
            self.m_.partial_fit(to_dense(X[i:i + self.batch]), y[i:i + self.batch], classes=[0, 1])
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        return np.vstack([self.m_.predict_proba(to_dense(X[i:i + self.batch])) for i in range(0, X.shape[0], self.batch)])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_classical(name, a):
    q = a.quick
    if name == "MNB":
        return MultinomialNB()
    if name == "LR":
        return LogisticRegression(solver="liblinear", max_iter=300, random_state=SEED)
    if name == "GNB":
        return BatchedGNB()
    if name == "RF":
        return RandomForestClassifier(n_estimators=50 if q else 100, n_jobs=-1, random_state=SEED, max_features="sqrt")
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100 if q else 300, max_depth=8, learning_rate=0.1, tree_method="hist",
                             n_jobs=-1, random_state=SEED, eval_metric="logloss", subsample=0.9, colsample_bytree=0.8)
    raise ValueError(name)


TUNE_GRIDS = {
    "LR": {"C": [0.1, 1, 10, 100]},
    "RF": {"n_estimators": [100, 200, 300], "max_depth": [None, 20, 40], "min_samples_split": [2, 5, 10]},
    "XGB": {"max_depth": [4, 6, 8, 10], "learning_rate": [0.05, 0.1, 0.2], "n_estimators": [200, 300, 500]},
}


def tune_if_requested(name, est, X, y, a):
    if not a.tune or name not in TUNE_GRIDS:
        return est, None
    n = min(a.tune_rows, X.shape[0])
    sub = np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)
    rs = RandomizedSearchCV(est, TUNE_GRIDS[name], n_iter=a.tune_iter, cv=a.tune_cv, scoring="f1", n_jobs=1, random_state=SEED)
    rs.fit(X[sub], np.asarray(y)[sub])
    return rs.best_estimator_, rs.best_params_


def get_tf():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    return tf


def build_deep(name, fg, shape, vocab=VOCAB_SIZE):
    """Keras models. For FG2 the input is the (MAXLEN,) index sequence; otherwise a feature vector."""
    tf = get_tf(); K = tf.keras; L = K.layers
    seq = fg == "FG2"
    inp = K.Input(shape=(shape,), dtype="int32" if seq else "float32")
    if name == "CNN" and seq:                                   # the base-paper character-level CNN
        x = L.Embedding(vocab, EMB_DIM)(inp)
        for k, p in [(7, 3), (7, 3), (3, None), (3, None), (3, None), (3, None), (3, 3)]:
            x = L.Conv1D(256, k, activation="relu")(x)
            if p:
                x = L.MaxPooling1D(p)(x)
        x = L.Flatten()(x)
        x = L.Dense(2048, activation="relu")(x); x = L.Dropout(0.5)(x)
        x = L.Dense(2048, activation="relu")(x); x = L.Dropout(0.5)(x)
    elif name == "CNN":                                         # CNN over a feature vector (FG1/FG3/FG4)
        x = L.Reshape((shape, 1))(inp)
        x = L.Conv1D(32, 3, activation="relu")(x); x = L.MaxPooling1D(4 if shape > 500 else 2)(x)
        x = L.Conv1D(64, 3, activation="relu")(x); x = L.GlobalMaxPooling1D()(x)
        x = L.Dense(256, activation="relu")(x); x = L.Dropout(0.3)(x)
    elif name == "DNN":
        x = L.Flatten()(L.Embedding(vocab, 32)(inp)) if seq else inp
        for u in (1024, 512, 128):
            x = L.Dense(u, activation="relu")(x); x = L.Dropout(0.3)(x)
    elif name == "RNN":
        x = L.Embedding(vocab, 64)(inp)
        x = L.LSTM(128)(x)
        x = L.Dense(128, activation="relu")(x)
    elif name == "RCNN":
        x = L.Embedding(vocab, 64)(inp)
        x = L.Conv1D(128, 5, activation="relu", padding="same")(x); x = L.MaxPooling1D(2)(x)
        x = L.LSTM(128)(x)
        x = L.Dense(128, activation="relu")(x)
    elif name == "VDCNN":                                       # shallow-depth variant of VDCNN (Conneau et al.)
        x = L.Embedding(vocab, 16)(inp)
        x = L.Conv1D(64, 3, padding="same", activation="relu")(x)
        for f in (64, 128, 256, 512):
            for _ in range(2):
                x = L.Conv1D(f, 3, padding="same")(x); x = L.BatchNormalization()(x); x = L.Activation("relu")(x)
            if f != 512:
                x = L.MaxPooling1D(2)(x)
        x = L.GlobalMaxPooling1D()(x)
        x = L.Dense(2048, activation="relu")(x); x = L.Dense(2048, activation="relu")(x)
    else:
        raise ValueError(name)
    out = L.Dense(2, activation="softmax")(x)
    m = K.Model(inp, out, name=f"{name}_{fg}")
    m.compile(optimizer=K.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def _batches(X, y, batch, shuffle, seed=SEED):
    """Generator that densifies sparse batches on the fly."""
    n = X.shape[0]; rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(n) if shuffle else np.arange(n)
        for i in range(0, n, batch):
            ix = order[i:i + batch]
            xb = to_dense(X[ix])
            yield (xb, np.asarray(y)[ix]) if y is not None else xb
        if not shuffle:
            return


def _predict_prob(model, X, batch):
    if sp.issparse(X):
        ps = [model.predict(to_dense(X[i:i + batch]), batch_size=batch, verbose=0)[:, 1] for i in range(0, X.shape[0], batch)]
        return np.concatenate(ps)
    return model.predict(X, batch_size=batch, verbose=0)[:, 1]


def fit_deep(name, fg, Xtr, ytr, Xva, yva, a, save_path=None):
    tf = get_tf(); K = tf.keras
    model = build_deep(name, fg, Xtr.shape[1])
    cbs = [K.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    steps = int(np.ceil(Xtr.shape[0] / a.batch)); vsteps = int(np.ceil(Xva.shape[0] / a.batch))
    t0 = time.time()
    if sp.issparse(Xtr):
        tr_ds = tf.data.Dataset.from_generator(lambda: _batches(Xtr, ytr, a.batch, True), output_signature=(
            tf.TensorSpec((None, Xtr.shape[1]), tf.float32), tf.TensorSpec((None,), tf.int64))).repeat()
        va_ds = tf.data.Dataset.from_generator(lambda: _batches(Xva, yva, a.batch, False), output_signature=(
            tf.TensorSpec((None, Xva.shape[1]), tf.float32), tf.TensorSpec((None,), tf.int64)))
        hist = model.fit(tr_ds, steps_per_epoch=steps, epochs=a.epochs, validation_data=va_ds, validation_steps=vsteps, callbacks=cbs, verbose=2)
    else:
        hist = model.fit(Xtr, np.asarray(ytr), batch_size=a.batch, epochs=a.epochs, validation_data=(Xva, np.asarray(yva)), callbacks=cbs, verbose=2)
    fit_time = time.time() - t0
    if save_path:
        model.save(save_path)
    return model, fit_time, {k: [float(v) for v in vals] for k, vals in hist.history.items()}


# ======================================================================================
# 4. EVALUATION
# ======================================================================================
def compute_metrics(y, p, thr=0.5):
    y = np.asarray(y); yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
            "auc": roc_auc_score(y, p) if len(set(y)) == 2 else float("nan"),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "fpr": fp / max(fp + tn, 1), "tpr": tp / max(tp + fn, 1)}


def sub_rows(a, idx):
    """Optional down-sampling of the training rows (for quick runs)."""
    if a.max_train_rows and len(idx) > a.max_train_rows:
        return np.random.default_rng(SEED).choice(idx, a.max_train_rows, replace=False)
    return idx


class Bundle:
    """Holds the URLs/labels of one dataset split and caches feature matrices per feature group."""

    def __init__(self, df, tr, va, te, a):
        self.df, self.a = df, a
        self.tr, self.va, self.te = sub_rows(a, tr), va, te
        self.cache = {}

    def urls(self, idx): return self.df.url.values[idx]
    def y(self, idx): return self.df.y.values[idx]

    def feats(self, fg, embed=False):
        key = (fg, embed)
        if key not in self.cache:
            t0 = time.time()
            fb = FeatureBuilder(fg, self.a.max_features).fit(self.urls(self.tr))
            Xtr = fb.transform(self.urls(self.tr), embed); Xva = fb.transform(self.urls(self.va), embed)
            t1 = time.time(); Xte = fb.transform(self.urls(self.te), embed); te_feat = time.time() - t1
            self.cache[key] = (fb, Xtr, Xva, Xte, t1 - t0, te_feat)
        return self.cache[key]


def run_one(model, fg, bundle, a, save_path=None, tag="", keep_model=False):
    """Train+evaluate one (model, feature group). Returns (record, y_prob, history, keras_model)."""
    deep = model in ("DNN", "CNN", "RNN", "RCNN", "VDCNN")
    fb, Xtr, Xva, Xte, feat_train, feat_test = bundle.feats(fg, embed=deep and fg == "FG2")
    ytr, yva, yte = bundle.y(bundle.tr), bundle.y(bundle.va), bundle.y(bundle.te)
    hist = None; km = None; params = None
    if deep:
        km, fit_t, hist = fit_deep(model, fg, Xtr, ytr, Xva, yva, a, save_path)
        t0 = time.time()
        pva = _predict_prob(km, Xva, a.batch)
        p = _predict_prob(km, Xte, a.batch)
        test_t = time.time() - t0
    else:
        est = make_classical(model, a)
        est, params = tune_if_requested(model, est, Xtr, ytr, a)
        t0 = time.time(); est.fit(Xtr, ytr); fit_t = time.time() - t0
        pva = est.predict_proba(Xva)[:, 1]
        t0 = time.time(); p = est.predict_proba(Xte)[:, 1]; test_t = time.time() - t0
        if fg == "FG4" and model in ("RF", "XGB") and hasattr(est, "feature_importances_"):
            pd.DataFrame({"feature": FG4_NAMES, "importance": est.feature_importances_}).sort_values("importance", ascending=False).to_csv(Path(a.out) / f"fg4_importance_{model}{tag}.csv", index=False)
    if deep and not keep_model:         # free Keras graphs between runs (prevents memory growth)
        del km; km = None
        get_tf().keras.backend.clear_session(); gc.collect()
    val_metrics = compute_metrics(yva, pva)
    rec = {"model": model, "fg": fg, **compute_metrics(yte, p),
           "validation_accuracy": val_metrics["accuracy"],
           "validation_precision": val_metrics["precision"],
           "validation_recall": val_metrics["recall"],
           "validation_f1": val_metrics["f1"],
           "train_time_s": feat_train + fit_t, "fit_time_s": fit_t,
           "test_time_s": feat_test + test_t, "ms_per_url": 1000 * (feat_test + test_t) / max(len(yte), 1),
           "n_train": len(ytr), "n_test": len(yte), "best_params": json.dumps(params) if params else ""}
    print(f"  {model:5s} {fg}  acc={rec['accuracy']:.4f}  P={rec['precision']:.4f}  R={rec['recall']:.4f}  F1={rec['f1']:.4f}  AUC={rec['auc']:.4f}  train={rec['train_time_s']:.0f}s", flush=True)
    return rec, p, hist, km


def bundle_for(a, name="PhiUSIIL", domain=False):
    df, sp_ = load_prepared(a.out, name)
    tr, va, te = (sp_["dtr"], sp_["dva"], sp_["dte"]) if domain else (sp_["tr"], sp_["va"], sp_["te"])
    return Bundle(df, tr, va, te, a)


# ======================================================================================
# 5. EXPERIMENTS 1-5
# ======================================================================================
def load_done(path, a):
    """Rows already written by an earlier (interrupted) run, so a Colab disconnect does not lose work."""
    if getattr(a, "fresh", False) or not Path(path).exists():
        return []
    return pd.read_csv(path, keep_default_na=False).to_dict("records")


def write_fg_accuracy_reports(out):
    """Create FG reports; rank feature groups using validation accuracy, never test accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    csv_path = out / "exp1.csv"
    if not csv_path.exists():
        return
    d = pd.read_csv(csv_path)
    if d.empty:
        return

    for fg in FGS:
        part = d[d["fg"] == fg].copy()
        if part.empty:
            continue
        fg_dir = out / fg
        fg_dir.mkdir(parents=True, exist_ok=True)
        part.sort_values("validation_accuracy", ascending=False).to_csv(
            fg_dir / "results.csv", index=False
        )
        plt.figure(figsize=(8, 4.5))
        part = part.sort_values("validation_accuracy", ascending=True)
        plt.barh(part["model"], part["validation_accuracy"] * 100)
        plt.xlabel("Validation accuracy (%)")
        plt.ylabel("Model")
        plt.title(f"{fg}: model validation accuracy")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(fg_dir / "accuracy_by_model.png", dpi=180)
        plt.close()

    # Validation accuracy is the pre-test selection signal.
    pivot = d.pivot_table(index="fg", columns="model", values="validation_accuracy", aggfunc="mean")
    pivot = pivot.reindex(FGS)
    pivot.to_csv(out / "fg_validation_accuracy_summary.csv")
    plt.figure(figsize=(8, 5))
    for model in pivot.columns:
        plt.plot(pivot.index, pivot[model] * 100, marker="o", linewidth=1.8, label=model)
    plt.xlabel("Feature group")
    plt.ylabel("Mean validation accuracy (%)")
    plt.title("Validation accuracy across FG1–FG4")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out / "accuracy_by_feature_group.png", dpi=200)
    plt.close()

    mean_val = d.groupby("fg")["validation_accuracy"].mean().reindex(FGS).dropna()
    top2 = mean_val.sort_values(ascending=False).head(2)
    top2.rename("mean_validation_accuracy").to_csv(out / "top2_feature_groups.csv")
    plt.figure(figsize=(6, 4))
    plt.bar(top2.index, top2.values * 100)
    plt.xlabel("Feature group")
    plt.ylabel("Mean validation accuracy (%)")
    plt.title("Top 2 feature groups selected on validation data")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out / "top2_feature_groups_accuracy.png", dpi=200)
    plt.close()


def cmd_exp1(a):
    """Experiment 1: FG1-FG4 x classifiers on PhiUSIIL (Table 5)."""
    b = bundle_for(a); rows = load_done(Path(a.out) / "exp1.csv", a)
    done = {(r["model"], r["fg"]) for r in rows}
    models = a.models.split(",") if a.models else MODELS_CLASSIC + MODELS_DEEP_FG
    fgs = a.fgs.split(",") if a.fgs else FGS
    for fg in fgs:
        for m in models:
            if m == "GNB" and a.skip_gnb:
                continue
            if (m, fg) in done:
                print(f"  {m} {fg}: already in exp1.csv, skipped (use --fresh to redo)"); continue
            rec, *_ = run_one(m, fg, b, a); rec["experiment"] = "exp1"; rows.append(rec)
            pd.DataFrame(rows).to_csv(Path(a.out) / "exp1.csv", index=False)
    write_fg_accuracy_reports(a.out)


def best_fg_per_model(a):
    d = pd.read_csv(Path(a.out) / "exp1.csv")
    d = d.sort_values(["model", "f1", "recall"], ascending=[True, False, False])
    return d.groupby("model").first()["fg"].to_dict()


def cmd_exp3(a):
    """Experiment 3: FG2 with RNN, RCNN, DNN, VDCNN, CNN on PhiUSIIL (Table 7)."""
    b = bundle_for(a); rows = load_done(Path(a.out) / "exp3.csv", a)
    done = {r["model"] for r in rows}
    for m in (a.models.split(",") if a.models else MODELS_EXP3):
        if m in done:
            print(f"  {m}: already in exp3.csv, skipped (use --fresh to redo)"); continue
        rec, p, hist, _ = run_one(m, "FG2", b, a); rec["experiment"] = "exp3"; rows.append(rec)
        json.dump(hist, open(Path(a.out) / f"history_exp3_{m}.json", "w"))
        pd.DataFrame(rows).to_csv(Path(a.out) / "exp3.csv", index=False)


def threshold_for_recall(y, p, target=0.99):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.5


def cmd_exp4(a):
    """Evaluate the proposed CNN on PhiUSIIL using stratified and domain-disjoint splits."""
    Path(a.out, "models").mkdir(exist_ok=True)
    proposed = a.proposed
    rows = []
    bundle = bundle_for(a)
    rec, p, hist, km = run_one(
        proposed, "FG2", bundle, a,
        save_path=str(Path(a.out) / "models" / "char_cnn.keras"),
        keep_model=True
    )
    rec.update(experiment="exp4", setting="PhiUSIIL_stratified")
    rows.append(rec)
    yte = bundle.y(bundle.te)
    pd.DataFrame({"url": bundle.urls(bundle.te), "y": yte, "prob": p}).to_csv(
        Path(a.out) / "preds_PhiUSIIL.csv", index=False
    )
    json.dump(hist, open(Path(a.out) / "history_exp4.json", "w"))
    pva = _predict_prob(km, bundle.feats("FG2", True)[2], a.batch)
    meta = {
        "maxlen": MAXLEN, "vocab_size": VOCAB_SIZE,
        "model": "char_cnn.keras", "default_threshold": 0.5,
        "threshold_recall_99": threshold_for_recall(bundle.y(bundle.va), pva, 0.99),
        "proposed": proposed
    }
    json.dump(meta, open(Path(a.out) / "models" / "meta.json", "w"), indent=2)
    save_plots(a.out, yte, p, hist, rec)
    del km
    get_tf().keras.backend.clear_session()
    gc.collect()

    domain_bundle = bundle_for(a, domain=True)
    rec, *_ = run_one(proposed, "FG2", domain_bundle, a, tag="_domain")
    rec.update(experiment="exp4", setting="PhiUSIIL_domain_disjoint")
    rows.append(rec)
    pd.DataFrame(rows).to_csv(Path(a.out) / "exp4.csv", index=False)
    if a.cv_folds:
        cv_classical(a, "exp4_cv_note")


def cv_classical(a, fname):
    """5-fold cross-validated recall/F1 of the strongest classical baselines on the training data."""
    b = bundle_for(a); rows = []
    for m, fg in (("RF", "FG4"), ("XGB", "FG4"), ("RF", "FG1"), ("XGB", "FG1")):
        fb, Xtr, *_ = b.feats(fg)
        y = b.y(b.tr)
        for metric in ("recall", "f1"):
            sc = cross_val_score(make_classical(m, a), Xtr, y, cv=StratifiedKFold(a.cv_folds or 5, shuffle=True, random_state=SEED), scoring=metric, n_jobs=1)
            rows.append({"model": m, "fg": fg, "metric": metric, "mean": sc.mean(), "std": sc.std(), "folds": a.cv_folds or 5})
        print("  CV", m, fg, rows[-2]["mean"], rows[-1]["mean"], flush=True)
    pd.DataFrame(rows).to_csv(Path(a.out) / "cv_baselines.csv", index=False)


# ======================================================================================
# 6. ERROR ANALYSIS, PLOTS, LATENCY
# ======================================================================================
def save_plots(out, y, p, hist, rec):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = Path(out)
    # ROC/AUC curve intentionally omitted. AUC remains available as a numeric metric in CSV tables.
    cm = np.array([[rec["tn"], rec["fp"]], [rec["fn"], rec["tp"]]])
    plt.figure(figsize=(3.6, 3.2)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Benign", "Phishing"]); plt.yticks([0, 1], ["Benign", "Phishing"]); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out / "confusion_PhiUSIIL.png", dpi=200); plt.close()
    if hist:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].plot(hist["accuracy"], label="train"); ax[0].plot(hist["val_accuracy"], label="validation"); ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
        ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="validation"); ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
        plt.tight_layout(); plt.savefig(out / "curves_PhiUSIIL.png", dpi=200); plt.close()


def cmd_errors(a):
    """Error analysis (Section 4.7): by URL length, sensitive words, TLD; threshold sweep."""
    out = Path(a.out); d = pd.read_csv(out / "preds_PhiUSIIL.csv", keep_default_na=False)
    d["pred"] = (d.prob >= 0.5).astype(int); d["err"] = (d.pred != d.y).astype(int)
    d["fp"] = ((d.pred == 1) & (d.y == 0)).astype(int); d["fn"] = ((d.pred == 0) & (d.y == 1)).astype(int)
    d["len"] = d.url.str.len()
    bins = [0, 20, 30, 50, 75, 100, 150, 10_000]
    d["len_bin"] = pd.cut(d["len"], bins=bins, right=False)
    g = d.groupby("len_bin", observed=True).agg(n=("err", "size"), error_rate=("err", "mean"), fp=("fp", "sum"), fn=("fn", "sum"))
    g.to_csv(out / "errors_by_length.csv")
    low = d.url.str.lower()
    d["sensitive"] = low.apply(lambda s: any(w in s for w in SENSITIVE)).astype(int)
    s = d.groupby("sensitive").agg(n=("err", "size"), error_rate=("err", "mean"), fp=("fp", "sum"), fn=("fn", "sum"), phish_share=("y", "mean"))
    s.to_csv(out / "errors_by_sensitive_word.csv")
    d["tld"] = d.url.map(tld_of)
    top = d.tld.value_counts().head(15).index
    d[d.tld.isin(top)].groupby("tld").agg(n=("err", "size"), error_rate=("err", "mean"), fp=("fp", "sum"), fn=("fn", "sum")).sort_values("n", ascending=False).to_csv(out / "errors_by_tld.csv")
    sweep = []
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(d.y.values, d.prob.values, t)
        sweep.append({"threshold": round(float(t), 2), "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "fp": m["fp"], "fn": m["fn"]})
    pd.DataFrame(sweep).to_csv(out / "threshold_sweep.csv", index=False)
    d[d.fn == 1].sort_values("prob").head(50)[["url", "prob"]].to_csv(out / "examples_false_negatives.csv", index=False)
    d[d.fp == 1].sort_values("prob", ascending=False).head(50)[["url", "prob"]].to_csv(out / "examples_false_positives.csv", index=False)
    print(g, "\n", s)


def predict_urls(model, urls, maxlen=MAXLEN, batch=256):
    """Shared by the evaluation scripts and the Streamlit app: URL strings -> phishing probability."""
    ids = encode_urls([normalise_url(u) for u in urls], maxlen)
    return model.predict(ids, batch_size=batch, verbose=0)[:, 1]


def cmd_latency(a):
    """Measures time per URL for single-URL inference (what a web service sees) and for batches."""
    import tensorflow as tf
    out = Path(a.out); meta = json.load(open(out / "models" / "meta.json"))
    model = tf.keras.models.load_model(out / "models" / meta["model"])
    d = pd.read_csv(out / "preds_PhiUSIIL.csv", keep_default_na=False)
    urls = d.url.sample(min(500, len(d)), random_state=SEED).tolist()
    predict_urls(model, urls[:5])                                           # warm-up
    t0 = time.time()
    for u in urls[:200]:
        predict_urls(model, [u])
    single = 1000 * (time.time() - t0) / len(urls[:200])
    t0 = time.time(); predict_urls(model, urls, batch=128); batched = 1000 * (time.time() - t0) / len(urls)
    t0 = time.time(); encode_urls([normalise_url(u) for u in urls]); enc = 1000 * (time.time() - t0) / len(urls)
    res = {"single_url_ms": single, "batched_ms_per_url": batched, "encoding_ms_per_url": enc, "base_paper_reported_ms_per_url": 0.47,
           "hardware_note": "report CPU/GPU used"}
    json.dump(res, open(out / "latency.json", "w"), indent=2); print(res)


# ======================================================================================
# 7. TABLES (filled from the result files; paste into the TBD cells of the paper)
# ======================================================================================
def pct(x): return "TBD" if pd.isna(x) else f"{100 * x:.2f}"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def cmd_tables(a):
    out = Path(a.out); parts = []
    st = json.load(open(out / "data_stats.json")) if (out / "data_stats.json").exists() else {}
    if "PhiUSIIL" in st:
        s = st["PhiUSIIL"]
        rows = [["PhiUSIIL (raw)", f"{s['raw_class_counts']['benign']:,}", f"{s['raw_class_counts']['phishing']:,}", f"{s['raw_rows']:,}"],
                [f"PhiUSIIL after de-duplication", f"{s['final_class_counts']['benign']:,}", f"{s['final_class_counts']['phishing']:,}", f"{s['final_rows']:,}"],
                ["PhiUSIIL training set", f"{s['split_tr']['benign']:,}", f"{s['split_tr']['phishing']:,}", f"{s['split_tr']['rows']:,}"],
                ["PhiUSIIL validation set", f"{s['split_va']['benign']:,}", f"{s['split_va']['phishing']:,}", f"{s['split_va']['rows']:,}"],
                ["PhiUSIIL test set", f"{s['split_te']['benign']:,}", f"{s['split_te']['phishing']:,}", f"{s['split_te']['rows']:,}"]]
        parts.append("### Table 3. Dataset distributions (raw duplicates removed: %d)\n" % s["raw_duplicate_urls_removed"] + md_table(["Dataset", "Benign", "Phishing", "Total"], rows))
    H = ["Model", "Features", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "AUC (%)"]
    if (out / "exp1.csv").exists():
        d = pd.read_csv(out / "exp1.csv")
        parts.append("### Table 5. Results on PhiUSIIL\n" + md_table(H, [[MODEL_LABEL[r.model], r.fg, pct(r.accuracy), pct(r.precision), pct(r.recall), pct(r.f1), pct(r.auc)] for r in d.itertuples()]))
    if (out / "exp3.csv").exists():
        d = pd.read_csv(out / "exp3.csv")
        parts.append("### Table 7. FG2 with different deep models on PhiUSIIL\n" + md_table(["Model", "AUC", "F1 (%)", "Precision (%)", "Recall (%)", "Accuracy (%)", "Training time (min)"],
                     [[r.model, pct(r.auc), pct(r.f1), pct(r.precision), pct(r.recall), pct(r.accuracy), f"{r.train_time_s / 60:.1f}"] for r in d.itertuples()]))
    if (out / "exp4.csv").exists():
        d = pd.read_csv(out / "exp4.csv")
        parts.append("### Table 8. Proposed model\n" + md_table(["Setting", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "AUC (%)", "Training time (s)", "Test time (s)"],
                     [[r.setting, pct(r.accuracy), pct(r.precision), pct(r.recall), pct(r.f1), pct(r.auc), f"{r.train_time_s:.0f}", f"{r.test_time_s:.1f}"] for r in d.itertuples()]))
        r = d.iloc[0]
        parts.append("### Table 9 row (this work, URL string only)\n" + md_table(["Study", "Method", "Inputs", "Reported result (%)"],
                     [["Proposed model (this work)", "Character-level CNN", "URL string only", f"Accuracy {pct(r.accuracy)}; F1 {pct(r.f1)}; AUC {pct(r.auc)}"]]))
    if (out / "latency.json").exists():
        parts.append("### Latency\n```\n" + open(out / "latency.json").read() + "\n```")
    open(out / "tables.md", "w").write("\n\n".join(parts) + "\n")
    print(f"wrote {out / 'tables.md'}")


# ======================================================================================
# 8. CLI
# ======================================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("commands", nargs="+", help="prepare exp1 exp3 exp4 errors latency tables")
    p.add_argument("--data", help="PhiUSIIL CSV (columns URL, label)")
    p.add_argument("--label-is-phish", action="store_true", help="set if label 1 already means phishing (PhiUSIIL: 1 = legitimate)")
    p.add_argument("--out", default="results")
    p.add_argument("--models"); p.add_argument("--fgs"); p.add_argument("--skip-gnb", action="store_true")
    p.add_argument("--epochs", type=int, default=8); p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-features", type=int, default=2000); p.add_argument("--max-train-rows", type=int, default=40000)
    p.add_argument("--quick", action="store_true", help="smaller RF/XGB for a fast pass")
    p.add_argument("--tune", action="store_true"); p.add_argument("--tune-rows", type=int, default=30000)
    p.add_argument("--tune-iter", type=int, default=8); p.add_argument("--tune-cv", type=int, default=5)
    p.add_argument("--fresh", action="store_true", help="ignore existing exp1/exp3 results and start over")
    p.add_argument("--proposed", default="CNN"); p.add_argument("--cv-folds", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    cmds = a.commands
    if "all" in cmds:
        cmds = ["prepare", "exp1", "exp3", "exp4", "tables"]
    table = {"prepare": cmd_prepare, "exp1": cmd_exp1, "exp3": cmd_exp3, "exp4": cmd_exp4,
             "errors": cmd_errors, "latency": cmd_latency, "tables": cmd_tables}
    for c in cmds:
        print(f"\n===== {c} =====", flush=True)
        saved_models = a.models
        table[c](a)
        a.models = saved_models


if __name__ == "__main__":
    main()
