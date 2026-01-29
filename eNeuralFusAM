import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)
from sklearn.utils import shuffle as sk_shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

# Local modules (expected in your repo)
from fusion_artmap import FusionARTMAP
from fuzzy_art import complement_code

# ======================= CONFIG =======================
DATA_CSV = os.path.expanduser("~/Desktop/Datasets/spambase.dat.csv")
LABEL_LAST_COL = True  # if True, last column is label

# Spambase channels: 48 word, 6 char, 3 capitals
SLICE_WORD = slice(0, 48)
SLICE_CHAR = slice(48, 54)
SLICE_CAPS = slice(54, 57)

# ANN settings (mirrors your reference model)
EMBED_DIM = 4
EPOCHS_FOLD0 = 20
EPOCHS_ONLINE = 20
BATCH_SIZE = 32
ANN_LR = 1e-3

# Fusion ARTMAP params (aligned conceptually with your FAM defaults)
FAM_ALPHA  = 0.01   # choice param (not learning rate)
FAM_BETA   = 1.0    # learning rate
FAM_RHO_C  = 0.0
FAM_RHO_A  = 0.0
FAM_RHO_B  = 1.0
FAM_RHO_AB = 1.0
FAM_EPS    = 0.0001
FAM_MAX_PMT_ITERS = 10
FAM_ARTA_COMPLEMENT = False  # align with your baseline

# Scaling mode: to strictly mimic the single-ANN pipeline, use ONE global MinMax
USE_GLOBAL_MINMAX = True

# Runs
RUNS = 10  # seeds 42..51

# ======================= UTILS =======================
def set_all_seeds(seed: int):
    import random, os
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def short_sheet(name: str, maxlen: int = 31) -> str:
    bad = {":", "\\", "/", "?", "*", "[", "]"}
    cleaned = "".join(ch if ch not in bad else "-" for ch in str(name))
    return cleaned[:maxlen]


def load_spambase(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)
    if LABEL_LAST_COL:
        y = df.iloc[:, -1].astype(int).values
        X = df.iloc[:, :-1].values
        cols = [f"f{i}" for i in range(X.shape[1])]
    else:
        # label named column path (not used by default)
        label_name = "target"
        y = df[label_name].astype(int).values
        X = df.drop(columns=[label_name]).values
        cols = [c for c in df.columns if c != label_name]
    return X, y, cols


def build_ann(input_dim: int, embed_dim: int = 4, lr: float = 1e-3, name: str = "c1"):
    x = L.Input(shape=(input_dim,), name=f"{name}_in")
    h = L.Dense(64, activation="relu", name=f"{name}_h1")(x)
    h = L.Dense(32, activation="relu", name=f"{name}_h2")(h)
    z = L.Dense(embed_dim, activation="relu", name=f"{name}_z")(h)
    yhat = L.Dense(1, activation="sigmoid", name=f"{name}_y")(z)
    model = keras.Model(x, yhat, name=f"ann_{name}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    embedder = keras.Model(x, z, name=f"embedder_{name}")
    return model, embedder


class FusionAdapter:
    """
    Wrapper to apply (global or per-channel) MinMax, clip to [0,1], and expose
    predict_with_scores() that returns (y_pred, pos_score) where pos_score
    mirrors your single-ANN FAM's notion of a positive-class choice score.
    """
    def __init__(self, fam: FusionARTMAP, mode: str, scalers):
        self.fam = fam
        self.mode = mode  # 'global' or 'per'
        self.scalers = scalers

    def _prep_batch(self, Z_list: List[np.ndarray]) -> List[np.ndarray]:
        if self.mode == 'global':
            mins, maxs, sl1, sl2, sl3 = self.scalers
            out = []
            for Z, sl in zip(Z_list, [sl1, sl2, sl3]):
                Zt = (Z - mins[sl]) / (maxs[sl] - mins[sl] + 1e-12)
                out.append(np.clip(Zt, 0.0, 1.0))
            return out
        else:
            m1, m2, m3 = self.scalers
            return [np.clip(m1.transform(Z_list[0]), 0, 1),
                    np.clip(m2.transform(Z_list[1]), 0, 1),
                    np.clip(m3.transform(Z_list[2]), 0, 1)]

    def partial_fit(self, Z1: np.ndarray, Z2: np.ndarray, Z3: np.ndarray, y: np.ndarray):
        Z1, Z2, Z3 = self._prep_batch([Z1, Z2, Z3])
        for i in range(len(y)):
            self.fam.train_one([Z1[i], Z2[i], Z3[i]], int(y[i]), verbose=False)
        return self

    def _predict_one_with_score(self, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray):
        fam = self.fam
        compressed = []
        # ART_c inference per channel
        for art, zk in zip(fam.art_channels, [z1, z2, z3]):
            I_cc = complement_code(np.clip(zk, 0.0, 1.0))
            j, Z, rr = art.process_infer(I_cc, rho=fam.rho_c, forbid=set())
            if j is None:
                # fallback: best-choice code
                T = art._choice_values(I_cc)
                j = int(np.argmax(T))
                w = art.weights[j]
                Z = (1.0 - art.beta) * w + fam.beta * np.minimum(I_cc, w)
            compressed.append(Z)

        # ART_a inference
        A = np.concatenate(compressed, axis=0)
        A_in = complement_code(A) if fam.art_a_complement else A
        j_a, Z_a, rr_a = fam.art_a.process_infer(A_in, rho=fam.rho_a, forbid=set())
        T_all = fam.art_a._choice_values(A_in)
        if j_a is None:
            j_a = int(np.argmax(T_all))

        # Map readout to predicted label
        if int(j_a) in fam.mapW:
            W = fam.mapW[int(j_a)]
            y_pred = int(np.argmax(W))
        else:
            order = np.argsort(-T_all)
            y_pred = 0
            for j in order:
                if int(j) in fam.mapW and fam.art_a._resonance_ratio(A_in, int(j)) >= fam.rho_a:
                    y_pred = int(np.argmax(fam.mapW[int(j)]))
                    j_a = int(j)
                    break

        # Score for AUROC: choice value of the selected ART_a category (to mimic single-ANN FAM)
        pos_score = float(T_all[int(j_a)])
        return y_pred, pos_score

    def predict_with_scores(self, Z1: np.ndarray, Z2: np.ndarray, Z3: np.ndarray):
        Z1, Z2, Z3 = self._prep_batch([Z1, Z2, Z3])
        n = Z1.shape[0]
        y_pred = np.zeros(n, dtype=int)
        pos_score = np.zeros(n, dtype=float)
        for i in range(n):
            y_pred[i], pos_score[i] = self._predict_one_with_score(Z1[i], Z2[i], Z3[i])
        return y_pred, pos_score


# ===================== ONE EXPERIMENT =====================

def run_one_experiment(X: np.ndarray, y: np.ndarray, seed: int, ts: str, save_online=True):
    set_all_seeds(seed)

    # shuffle & fold
    X_shuf, y_shuf = sk_shuffle(X, y, random_state=seed)
    X_folds = np.array_split(X_shuf, 10)
    y_folds = np.array_split(y_shuf, 10)

    # Standardize on Fold-0 (reused for all folds)
    scaler0 = StandardScaler().fit(X_folds[0])
    X0s = scaler0.transform(X_folds[0]).astype("float32")

    # Split channels
    X0_c1 = X0s[:, SLICE_WORD]
    X0_c2 = X0s[:, SLICE_CHAR]
    X0_c3 = X0s[:, SLICE_CAPS]

    # Build & train three ANNs on Fold-0
    (ann1, emb1) = build_ann(X0_c1.shape[1], EMBED_DIM, ANN_LR, name="c1")
    (ann2, emb2) = build_ann(X0_c2.shape[1], EMBED_DIM, ANN_LR, name="c2")
    (ann3, emb3) = build_ann(X0_c3.shape[1], EMBED_DIM, ANN_LR, name="c3")

    ann1.fit(X0_c1, y_folds[0], epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)
    ann2.fit(X0_c2, y_folds[0], epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)
    ann3.fit(X0_c3, y_folds[0], epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)

    # Fold-0 embeddings
    Z1_0 = emb1.predict(X0_c1, verbose=0)
    Z2_0 = emb2.predict(X0_c2, verbose=0)
    Z3_0 = emb3.predict(X0_c3, verbose=0)

    # MinMax on Fold-0 features (global or per-channel)
    if USE_GLOBAL_MINMAX:
        Z0_concat = np.concatenate([Z1_0, Z2_0, Z3_0], axis=1)
        mins = Z0_concat.min(axis=0)
        maxs = Z0_concat.max(axis=0)
        s1, s2, s3 = Z1_0.shape[1], Z2_0.shape[1], Z3_0.shape[1]
        idx1 = slice(0, s1); idx2 = slice(s1, s1+s2); idx3 = slice(s1+s2, s1+s2+s3)
        def apply_global(z, sl):
            return np.clip((z - mins[sl]) / (maxs[sl] - mins[sl] + 1e-12), 0, 1)
        Z1_0 = apply_global(Z1_0, idx1)
        Z2_0 = apply_global(Z2_0, idx2)
        Z3_0 = apply_global(Z3_0, idx3)
        scaler_pack = (mins, maxs, idx1, idx2, idx3)
        adapter_mode = 'global'
    else:
        mms1 = MinMaxScaler().fit(Z1_0); Z1_0 = np.clip(mms1.transform(Z1_0), 0, 1)
        mms2 = MinMaxScaler().fit(Z2_0); Z2_0 = np.clip(mms2.transform(Z2_0), 0, 1)
        mms3 = MinMaxScaler().fit(Z3_0); Z3_0 = np.clip(mms3.transform(Z3_0), 0, 1)
        scaler_pack = (mms1, mms2, mms3)
        adapter_mode = 'per'

    # Build Fusion core & adapter
    fam_core = FusionARTMAP(
        channel_dims=[EMBED_DIM, EMBED_DIM, EMBED_DIM],
        n_classes=2,
        alpha=FAM_ALPHA,
        beta=FAM_BETA,
        rho_c=FAM_RHO_C,
        rho_a=FAM_RHO_A,
        rho_b=FAM_RHO_B,
        rho_ab=FAM_RHO_AB,
        eps=FAM_EPS,
        max_pmt_iters=FAM_MAX_PMT_ITERS,
        art_a_complement=FAM_ARTA_COMPLEMENT,
        reset_vigilance_each_sample=True,
    )
    adapter = FusionAdapter(fam_core, adapter_mode, scaler_pack)

    # Train Fusion on Fold-0
    adapter.partial_fit(Z1_0, Z2_0, Z3_0, y_folds[0])

    # Collectors
    perf_rows = []
    pred_sheets = []
    cm_sheets = []
    weight_sheets = []

    # ======== Folds 1..9: TEST -> log -> TRAIN ========
    for i in range(1, 10):
        Xi = scaler0.transform(X_folds[i]).astype("float32")
        yi = y_folds[i]
        Xi_c1, Xi_c2, Xi_c3 = Xi[:, SLICE_WORD], Xi[:, SLICE_CHAR], Xi[:, SLICE_CAPS]

        # TEST (before training)
        Z1 = emb1.predict(Xi_c1, verbose=0)
        Z2 = emb2.predict(Xi_c2, verbose=0)
        Z3 = emb3.predict(Xi_c3, verbose=0)
        y_pred, pos_score = adapter.predict_with_scores(Z1, Z2, Z3)

        acc  = accuracy_score(yi, y_pred)
        prec = precision_score(yi, y_pred, zero_division=0)
        rec  = recall_score(yi, y_pred, zero_division=0)
        f1   = f1_score(yi, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(yi, pos_score)
        except Exception:
            auc = float("nan")
        mcc  = matthews_corrcoef(yi, y_pred)
        cm   = confusion_matrix(yi, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        perf_rows.append({
            "Fold": f"{i-1}→{i}",
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "AUROC": auc, "MCC": mcc, "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })

        pred_df = pd.DataFrame({"True Label": yi, "Predicted": y_pred})
        pred_sheets.append((short_sheet(f"Fold_{i}_Predictions"), pred_df))
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        cm_sheets.append((short_sheet(f"Fold_{i}_ConfusionMatrix"), cm_df))

        # TRAIN (fine-tune ANNs + Fusion)
        ann1.fit(Xi_c1, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)
        ann2.fit(Xi_c2, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)
        ann3.fit(Xi_c3, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)

        Z1_tr = emb1.predict(Xi_c1, verbose=0)
        Z2_tr = emb2.predict(Xi_c2, verbose=0)
        Z3_tr = emb3.predict(Xi_c3, verbose=0)
        adapter.partial_fit(Z1_tr, Z2_tr, Z3_tr, yi)

        # Weight snapshots AFTER training on fold i
        def weights_to_df(weights: List[np.ndarray], prefix: str) -> pd.DataFrame:
            if len(weights) == 0:
                return pd.DataFrame()
            W = np.vstack([w.reshape(1, -1) for w in weights])
            dfw = pd.DataFrame(W, columns=[f"{prefix}_d{j}" for j in range(W.shape[1])])
            dfw.index.name = f"{prefix}_F2"
            return dfw

        w_c1 = weights_to_df(fam_core.art_channels[0].weights, f"C1")
        w_c2 = weights_to_df(fam_core.art_channels[1].weights, f"C2")
        w_c3 = weights_to_df(fam_core.art_channels[2].weights, f"C3")
        w_a  = weights_to_df(fam_core.art_a.weights,            f"A")

        weight_sheets.extend([
            (short_sheet(f"Fold_{i}_C1W"), w_c1),
            (short_sheet(f"Fold_{i}_C2W"), w_c2),
            (short_sheet(f"Fold_{i}_C3W"), w_c3),
            (short_sheet(f"Fold_{i}_ARTaW"), w_a),
        ])

    perf_df = pd.DataFrame(perf_rows)

    # Save single-run workbook (seed 42 only)
    if save_online and seed == 42:
        out_online = os.path.expanduser(os.path.join("~/Desktop", f"ANN3_Fusion_Online_{ts}.xlsx"))
        with pd.ExcelWriter(out_online) as writer:
            perf_df.to_excel(writer, index=False, sheet_name=short_sheet("Performance Metrics"))
            for nm, df in pred_sheets:
                df.to_excel(writer, index=False, sheet_name=nm)
            for nm, df in cm_sheets:
                df.to_excel(writer, sheet_name=nm)
            for nm, df in weight_sheets:
                if not df.empty:
                    df.to_excel(writer, sheet_name=nm)
        print(f"📁 Single online pass saved to: {out_online}")

    # Return per-fold metrics for aggregator (keep a numeric Fold column like baseline)
    perf_df = perf_df.copy()
    perf_df["Fold"] = perf_df["Fold"].str.split("→").str[-1].astype(int)
    return perf_df


# =========================== MAIN ============================
if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")

    TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    X, y, _ = load_spambase(DATA_CSV)
    assert X.shape[1] >= 57, f"Expected at least 57 features (got {X.shape[1]})."

    # ------- Single online pass (seed 42) -------
    _ = run_one_experiment(X, y, seed=42, ts=TS, save_online=True)

    # ------- 10 independent runs (seeds 42..51) -------
    all_runs = []
    for r in range(RUNS):
        seed = 42 + r
        print(f"🚀 Run {r+1}/{RUNS}  (seed={seed})")
        perf_df = run_one_experiment(X, y, seed=seed, ts=TS, save_online=False)
        perf_df.insert(0, "Run", r+1)
        all_runs.append(perf_df)

    out_runs = os.path.expanduser(os.path.join("~/Desktop/ann3_spam_results", f"ANN3_Fusion_10Runs_{TS}.xlsx"))
    with pd.ExcelWriter(out_runs) as writer:
        for r, df in enumerate(all_runs, start=1):
            df.to_excel(writer, index=False, sheet_name=short_sheet(f"Run_{r:02d}"))
        combined = pd.concat(all_runs, ignore_index=True)
        avg = (combined
               .groupby("Fold", as_index=False)
               .mean(numeric_only=True)
               .drop(columns=["Run"]))
        combined.to_excel(writer, index=False, sheet_name=short_sheet("All_10_Runs"))
        avg.to_excel(writer, index=False, sheet_name=short_sheet("Average_Performance"))
    print(f"📁 10 runs saved to: {out_runs}")
