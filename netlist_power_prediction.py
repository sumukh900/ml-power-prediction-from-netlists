#!/usr/bin/env python3
"""
combined_power_model.py

Merged implementation combining:
 - TF-IDF + structural features + instance token counts
 - Positive Linear Regression for area estimation from instance counts
 - RandomForest for cell_count (with OOF predictions)
 - MultiOutput RandomForest for power targets (log-transformed for skewed columns)
 - Optional randomized hyperparameter search for the MultiOutput regressor

Edit configuration at the top. Designed to be self-contained and readable.
"""
import os
import re
import joblib
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import train_test_split, cross_val_predict, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# ---------- Config ----------
DATA_PATH = r"C:\Users\91986\Downloads\Untitled spreadsheet.xlsx"
MODEL_PICKLE = r"C:\Sumukh\power_prediction\merged_model.pkl"
SUMMARY_OUT = r"C:\Sumukh\power_prediction\merged_summary.txt"
RANDOM_STATE = 42
TEST_SIZE = 0.15

# Toggle optional hyperparameter search for the multi-output power model.
DO_HYPERPARAM_SEARCH = False
HYPERPARAM_N_ITERS = 12
HYPERPARAM_CV = 3

# column name candidates
NETLIST_NAME_CANDIDATES = ["netlist", "net list", "rtl", "verilog", "vhdl", "design"]
target_name_candidates = {
    "leakage_power": ["leakage_power", "leakage", "leakage_pwr", "leakage power"],
    "internal_power": ["internal_power", "internal", "internal_pwr", "internal power"],
    "switching_power": ["switching_power", "switching", "switching_pwr", "switching power"],
    "total_power": ["total_power", "total", "total_pwr", "total power", "power_total"],
    "cell_count": ["cell_count", "cells", "cell count", "cell_count_total", "cellcount"],
    "total_area": ["total_area", "area", "total area", "die_area"]
}

# instance tokens list (extend as needed)
INSTANCE_TOKENS = [
    "DFFQX1","DFFRX1","NOR2XL","NOR2BXL","NOR2X1","NAND2XL","INVX1","CLKBUF","BUF",
    "AOI21XL","OAI21XL","MX2XL","XOR2X1","XNOR2X1"
]

# token regex patterns
TOKEN_RE = re.compile(r"[A-Za-z0-9_\.]+")
INSTANCE_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")

# ---------- Helpers ----------
def find_column_by_candidates(df, candidate_list):
    cols = df.columns.tolist()
    lower_cols = [c.lower() for c in cols]
    for cand in candidate_list:
        if cand.lower() in lower_cols:
            return cols[lower_cols.index(cand.lower())]
    # fallback substring match
    for cand in candidate_list:
        for i, lc in enumerate(lower_cols):
            if cand.lower() in lc:
                return cols[i]
    return None

def count_instance_tokens(text, tokens_list):
    found = INSTANCE_TOKEN_RE.findall(str(text))
    counts = {t: 0 for t in tokens_list}
    for f in found:
        if f in counts:
            counts[f] += 1
    return [counts[t] for t in tokens_list]

def make_intermediate(netlist_series):
    intermediate = []
    for s in netlist_series:
        s_str = str(s)
        toks = TOKEN_RE.findall(s_str)
        intermediate.append([s_str, len(s_str), len(toks), len(set(toks))])
    return intermediate

def safe_log_transform(Y, cols_to_log_mask):
    Yt = Y.copy()
    for i, do in enumerate(cols_to_log_mask):
        if do:
            Yt[:, i] = np.log1p(np.maximum(Yt[:, i], 0.0))
    return Yt

def safe_log_inverse(Yt, cols_to_log_mask):
    Y = Yt.copy()
    for i, do in enumerate(cols_to_log_mask):
        if do:
            Y[:, i] = np.expm1(Y[:, i])
    return Y

# ---------- Feature builder ----------
class TFIDFWithInstanceCounts(BaseEstimator, TransformerMixin):
    """
    TF-IDF on netlist text + structural numeric columns (len, tokens, unique tokens)
    + instance-token counts appended.
    """
    def __init__(self, instance_tokens, max_features=6000, ngram_range=(1,2)):
        self.instance_tokens = instance_tokens
        self.vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z0-9_\.]+",
                                          max_features=max_features,
                                          ngram_range=ngram_range)
    def fit(self, X, y=None):
        texts = [row[0] for row in X]
        self.vectorizer.fit(texts)
        return self
    def transform(self, X):
        texts = [row[0] for row in X]
        X_text = self.vectorizer.transform(texts)
        struct_rows = []
        inst_rows = []
        for row in X:
            text = row[0]
            toks = TOKEN_RE.findall(text)
            struct_rows.append([len(text), len(toks), len(set(toks))])
            inst_rows.append(count_instance_tokens(text, self.instance_tokens))
        struct_np = np.array(struct_rows, dtype=float)
        inst_np = np.array(inst_rows, dtype=float)
        combined = sp.hstack([X_text, sp.csr_matrix(struct_np), sp.csr_matrix(inst_np)], format="csr")
        return combined

# ---------- Main ----------
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("[INFO] Reading dataset...")
    df = pd.read_excel(DATA_PATH)

    # detect netlist column
    netlist_col = find_column_by_candidates(df, NETLIST_NAME_CANDIDATES)
    if netlist_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == object]
        if len(object_cols) == 0:
            raise ValueError("No netlist/text column found.")
        netlist_col = object_cols[0]
        print(f"[WARNING] Netlist column auto-picked as '{netlist_col}'.")

    # detect targets in a fixed order
    targets = []
    target_order = ["leakage_power","internal_power","switching_power","total_power","cell_count","total_area"]
    for key in target_order:
        col = find_column_by_candidates(df, target_name_candidates[key])
        if col is None:
            # substring fallback
            found = None
            for c in df.columns:
                if key.replace("_"," ") in c.lower() or key in c.lower():
                    found = c; break
            if found is None:
                raise ValueError(f"Missing target column for {key}")
            col = found
        targets.append(col)

    print("[INFO] Using columns:")
    print(" Netlist: ", netlist_col)
    for k, c in zip(target_order, targets):
        print(f"  {k} -> {c}")

    # Prepare df
    df = df[[netlist_col] + targets].copy()
    df[netlist_col] = df[netlist_col].fillna("").astype(str)
    y_df = df[targets].apply(pd.to_numeric, errors="coerce")
    mask = ~y_df.isna().any(axis=1)
    if mask.sum() != len(df):
        print(f"[INFO] Dropping {len(df) - mask.sum()} rows with missing target values.")
        df = df[mask]
        y_df = y_df[mask]
    y = y_df.values

    # intermediate features
    X_intermediate = make_intermediate(df[netlist_col])

    # ----- Area model (non-negative linear) using instance counts -----
    print("[INFO] Building instance counts for area model...")
    inst_counts = np.vstack([count_instance_tokens(s, INSTANCE_TOKENS) for s in df[netlist_col]])
    area_idx = target_order.index("total_area")
    lr_area = LinearRegression(positive=True)  # enforces non-negative coefficients/preds
    print("[INFO] Fitting positive LinearRegression for total_area from instance counts...")
    lr_area.fit(inst_counts, y[:, area_idx])
    area_by_inst_all = lr_area.predict(inst_counts)

    # ----- TFIDF + instance + struct builder -----
    print("[INFO] Fitting TF-IDF + instance-count feature builder...")
    tf_builder = TFIDFWithInstanceCounts(INSTANCE_TOKENS, max_features=6000, ngram_range=(1,2))
    tf_builder.fit(X_intermediate)
    X_base = tf_builder.transform(X_intermediate)  # tfidf + struct + inst counts

    # append area_by_inst as a numeric column to X_base
    X_with_area = sp.hstack([X_base, sp.csr_matrix(area_by_inst_all.reshape(-1,1))], format="csr")

    # ----- cell_count model (RF) with OOF predictions -----
    cell_idx = target_order.index("cell_count")
    rf_cell = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

    print("[INFO] Generating OOF cell_count predictions (cross_val_predict)...")
    cell_oof = cross_val_predict(rf_cell, X_with_area, y[:, cell_idx], cv=5, n_jobs=-1, method='predict')

    # fit RF on full data for final cell model
    print("[INFO] Fitting final RandomForest for cell_count on full data...")
    rf_cell.fit(X_with_area, y[:, cell_idx])

    # ----- Build feature matrix for power model: include area_by_inst and OOF cell predictions -----
    X_for_power = sp.hstack([X_with_area, sp.csr_matrix(cell_oof.reshape(-1,1))], format="csr")

    # ----- power model: MultiOutput RandomForest on log-transformed power columns -----
    # log mask: first 4 are power columns (log-transform), last two (cell_count, total_area) are not
    log_mask = [True, True, True, True, False, False]
    y_log = safe_log_transform(y.copy(), log_mask)

    base_reg = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    multi_reg = MultiOutputRegressor(base_reg, n_jobs=-1)

    if DO_HYPERPARAM_SEARCH:
        print("[INFO] Running randomized hyperparameter search for multi-output regressor...")
        param_dist = {
            'estimator__n_estimators': [200, 400, 800],
            'estimator__max_depth': [None, 20, 40, 80],
            'estimator__min_samples_leaf': [1, 2, 5, 10],
            'estimator__max_features': ['sqrt', 0.2, 0.5, 0.8]
        }
        rsearch = RandomizedSearchCV(
            multi_reg, param_distributions=param_dist, n_iter=HYPERPARAM_N_ITERS,
            cv=HYPERPARAM_CV, verbose=2, random_state=RANDOM_STATE, n_jobs=1
        )
        # split for training search
        X_train_search, X_test_search, y_train_search, y_test_search = train_test_split(
            X_for_power, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        rsearch.fit(X_train_search, y_train_search)
        print("[INFO] Best params found:", rsearch.best_params_)
        best_model = rsearch.best_estimator_
    else:
        print("[INFO] Training MultiOutput RandomForest (no hyperparam search)...")
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X_for_power, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        multi_reg.fit(X_train_full, y_train_full)
        best_model = multi_reg
        # keep test splits for evaluation below
        X_test_for_eval = X_test_full
        y_test_log_for_eval = y_test_full

    # If we did hyperparam search we still need a held-out test split to evaluate
    if DO_HYPERPARAM_SEARCH:
        # use fresh test split for evaluation
        X_train_full, X_test_for_eval, y_train_full, y_test_log_for_eval = train_test_split(
            X_for_power, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        # ensure best_model is trained on full training partition (if rsearch already fitted on a subset this is ok);
        # here we assume rsearch.best_estimator_ is ready.

    # ----- Evaluation -----
    print("[INFO] Evaluating models on hold-out test set...")
    # Evaluate cell_count RF: build the corresponding test split for X_with_area
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_with_area, y[:, cell_idx], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    cell_pred_test = rf_cell.predict(Xc_test)
    cell_r2 = r2_score(yc_test, cell_pred_test)
    cell_mae = mean_absolute_error(yc_test, cell_pred_test)

    # Evaluate multi-output on powers: predict and invert logs
    y_pred_log = best_model.predict(X_test_for_eval)
    y_test_orig = safe_log_inverse(y_test_log_for_eval, log_mask)
    y_pred_orig = safe_log_inverse(y_pred_log, log_mask)

    r2s = [r2_score(y_test_orig[:, i], y_pred_orig[:, i]) for i in range(y_test_orig.shape[1])]
    maes = [mean_absolute_error(y_test_orig[:, i], y_pred_orig[:, i]) for i in range(y_test_orig.shape[1])]

    # ----- Save model bundle -----
    model_bundle = {
        "netlist_col": netlist_col,
        "targets": targets,
        "tf_builder": tf_builder,
        "instance_tokens": INSTANCE_TOKENS,
        "lr_area": lr_area,
        "rf_cell": rf_cell,
        "multi_reg": best_model,
        "log_mask": log_mask
    }
    os.makedirs(os.path.dirname(MODEL_PICKLE), exist_ok=True)
    joblib.dump(model_bundle, MODEL_PICKLE)
    print(f"[INFO] Saved model bundle to: {MODEL_PICKLE}")

    # ----- Print & write summary -----
    with open(SUMMARY_OUT, "w") as f:
        f.write("Evaluation & prediction summary\n")
        f.write(f"cell_count: R2={cell_r2:.4f}, MAE={cell_mae:.6g}\n")
        print(f" cell_count: R2={cell_r2:.4f}, MAE={cell_mae:.6g}")
        for name, r2v, maev in zip(targets, r2s, maes):
            line = f"{name}: R2={r2v:.4f}, MAE={maev:.6g}\n"
            f.write(line)
            print(" ", line.strip())
        f.write("\n")

    # ---------- Hardcoded netlist prediction (example) ----------
    hardcoded_netlist = """
   module johnson_counter_32(clk, reset, count);
  input clk, reset;
  output [31:0] count;
  wire clk, reset;
  wire [31:0] count;
  wire n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7;
  wire n_8, n_9, n_10, n_11, n_12, n_13, n_14, n_15;
  wire n_16, n_17, n_18, n_19, n_20, n_21, n_22, n_23;
  wire n_24, n_25, n_26, n_27, n_28, n_29, n_30, n_31;
  DFFQX1 \count_reg[0] (.CK (clk), .D (n_31), .Q (count[0]));
  NOR2XL g5__2398(.A (reset), .B (count[31]), .Y (n_31));
  DFFQX1 \count_reg[31] (.CK (clk), .D (n_30), .Q (count[31]));
  NOR2BXL g7__5107(.AN (count[30]), .B (reset), .Y (n_30));
  DFFQX1 \count_reg[30] (.CK (clk), .D (n_29), .Q (count[30]));
  NOR2BXL g9__6260(.AN (count[29]), .B (reset), .Y (n_29));
  DFFQX1 \count_reg[29] (.CK (clk), .D (n_28), .Q (count[29]));
  NOR2BXL g11__4319(.AN (count[28]), .B (reset), .Y (n_28));
  DFFQX1 \count_reg[28] (.CK (clk), .D (n_27), .Q (count[28]));
  NOR2BXL g13__8428(.AN (count[27]), .B (reset), .Y (n_27));
  DFFQX1 \count_reg[27] (.CK (clk), .D (n_26), .Q (count[27]));
  NOR2BXL g15__5526(.AN (count[26]), .B (reset), .Y (n_26));
  DFFQX1 \count_reg[26] (.CK (clk), .D (n_25), .Q (count[26]));
  NOR2BXL g17__6783(.AN (count[25]), .B (reset), .Y (n_25));
  DFFQX1 \count_reg[25] (.CK (clk), .D (n_24), .Q (count[25]));
  NOR2BXL g19__3680(.AN (count[24]), .B (reset), .Y (n_24));
  DFFQX1 \count_reg[24] (.CK (clk), .D (n_23), .Q (count[24]));
  NOR2BXL g21__1617(.AN (count[23]), .B (reset), .Y (n_23));
  DFFQX1 \count_reg[23] (.CK (clk), .D (n_22), .Q (count[23]));
  NOR2BXL g23__2802(.AN (count[22]), .B (reset), .Y (n_22));
  DFFQX1 \count_reg[22] (.CK (clk), .D (n_21), .Q (count[22]));
  NOR2BXL g25__1705(.AN (count[21]), .B (reset), .Y (n_21));
  DFFQX1 \count_reg[21] (.CK (clk), .D (n_20), .Q (count[21]));
  NOR2BXL g27__5122(.AN (count[20]), .B (reset), .Y (n_20));
  DFFQX1 \count_reg[20] (.CK (clk), .D (n_19), .Q (count[20]));
  NOR2BXL g29__8246(.AN (count[19]), .B (reset), .Y (n_19));
  DFFQX1 \count_reg[19] (.CK (clk), .D (n_18), .Q (count[19]));
  NOR2BXL g31__7098(.AN (count[18]), .B (reset), .Y (n_18));
  DFFQX1 \count_reg[18] (.CK (clk), .D (n_17), .Q (count[18]));
  NOR2BXL g33__6131(.AN (count[17]), .B (reset), .Y (n_17));
  DFFQX1 \count_reg[17] (.CK (clk), .D (n_16), .Q (count[17]));
  NOR2BXL g35__1881(.AN (count[16]), .B (reset), .Y (n_16));
  DFFQX1 \count_reg[16] (.CK (clk), .D (n_15), .Q (count[16]));
  NOR2BXL g37__5115(.AN (count[15]), .B (reset), .Y (n_15));
  DFFQX1 \count_reg[15] (.CK (clk), .D (n_14), .Q (count[15]));
  NOR2BXL g39__7482(.AN (count[14]), .B (reset), .Y (n_14));
  DFFQX1 \count_reg[14] (.CK (clk), .D (n_13), .Q (count[14]));
  NOR2BXL g41__4733(.AN (count[13]), .B (reset), .Y (n_13));
  DFFQX1 \count_reg[13] (.CK (clk), .D (n_12), .Q (count[13]));
  NOR2BXL g43__6161(.AN (count[12]), .B (reset), .Y (n_12));
  DFFQX1 \count_reg[12] (.CK (clk), .D (n_11), .Q (count[12]));
  NOR2BXL g45__9315(.AN (count[11]), .B (reset), .Y (n_11));
  DFFQX1 \count_reg[11] (.CK (clk), .D (n_10), .Q (count[11]));
  NOR2BXL g47__9945(.AN (count[10]), .B (reset), .Y (n_10));
  DFFQX1 \count_reg[10] (.CK (clk), .D (n_9), .Q (count[10]));
  NOR2BXL g49__2883(.AN (count[9]), .B (reset), .Y (n_9));
  DFFQX1 \count_reg[9] (.CK (clk), .D (n_8), .Q (count[9]));
  NOR2BXL g51__2346(.AN (count[8]), .B (reset), .Y (n_8));
  DFFQX1 \count_reg[8] (.CK (clk), .D (n_7), .Q (count[8]));
  NOR2BXL g53__1666(.AN (count[7]), .B (reset), .Y (n_7));
  DFFQX1 \count_reg[7] (.CK (clk), .D (n_6), .Q (count[7]));
  NOR2BXL g55__7410(.AN (count[6]), .B (reset), .Y (n_6));
  DFFQX1 \count_reg[6] (.CK (clk), .D (n_5), .Q (count[6]));
  NOR2BXL g57__6417(.AN (count[5]), .B (reset), .Y (n_5));
  DFFQX1 \count_reg[5] (.CK (clk), .D (n_4), .Q (count[5]));
  NOR2BXL g59__5477(.AN (count[4]), .B (reset), .Y (n_4));
  DFFQX1 \count_reg[4] (.CK (clk), .D (n_3), .Q (count[4]));
  NOR2BXL g61__2398(.AN (count[3]), .B (reset), .Y (n_3));
  DFFQX1 \count_reg[3] (.CK (clk), .D (n_2), .Q (count[3]));
  NOR2BXL g63__5107(.AN (count[2]), .B (reset), .Y (n_2));
  DFFQX1 \count_reg[2] (.CK (clk), .D (n_1), .Q (count[2]));
  NOR2BXL g65__6260(.AN (count[1]), .B (reset), .Y (n_1));
  DFFQX1 \count_reg[1] (.CK (clk), .D (n_0), .Q (count[1]));
  NOR2BXL g67__4319(.AN (count[0]), .B (reset), .Y (n_0));
endmodule




    """
    print("\n[INFO] Predicting for provided hardcoded netlist sample...")
    tokens = TOKEN_RE.findall(hardcoded_netlist)
    inter = [[hardcoded_netlist, len(hardcoded_netlist), len(tokens), len(set(tokens))]]
    X_feat = tf_builder.transform(inter)
    # compute instance counts and area_by_inst
    hard_inst_counts = np.array([count_instance_tokens(hardcoded_netlist, INSTANCE_TOKENS)])
    hard_area_by_inst = lr_area.predict(hard_inst_counts)[0]
    X_feat_with_area = sp.hstack([X_feat, sp.csr_matrix(np.array([[hard_area_by_inst]]))], format="csr")
    # predict cell_count using rf_cell, round it
    pred_cell = rf_cell.predict(X_feat_with_area)[0]
    pred_cell_round = int(round(pred_cell))
    # append predicted cell as the final feature column (like OOF cell used in training)
    X_feat_for_power = sp.hstack([X_feat_with_area, sp.csr_matrix(np.array([[pred_cell]]))], format="csr")
    pred_log = best_model.predict(X_feat_for_power)[0]
    pred_orig = safe_log_inverse(pred_log.reshape(1, -1), log_mask)[0]
    pred_area = float(hard_area_by_inst)
    pred_cell_final = pred_cell_round

    print("\nPredicted Results for Provided Netlist:")
    print(f" LEAKAGE POWER (W): {pred_orig[0]:.12f}")
    print(f" INTERNAL POWER (W): {pred_orig[1]:.12f}")
    print(f" SWITCHING POWER (W): {pred_orig[2]:.12f}")
    print(f" TOTAL POWER (W): {pred_orig[3]:.12f}")
    print(f" CELL COUNT : {pred_cell_final}")
    print(f" TOTAL AREA : {pred_area:.12f}")

    with open(SUMMARY_OUT, "a") as f:
        f.write("\nPredicted Results for Provided Netlist:\n")
        f.write(f"LEAKAGE POWER (W): {pred_orig[0]:.12f}\n")
        f.write(f"INTERNAL POWER (W): {pred_orig[1]:.12f}\n")
        f.write(f"SWITCHING POWER (W): {pred_orig[2]:.12f}\n")
        f.write(f"TOTAL POWER (W): {pred_orig[3]:.12f}\n")
        f.write(f"CELL COUNT : {pred_cell_final}\n")
        f.write(f"TOTAL AREA : {pred_area:.12f}\n")

    print(f"\n[INFO] Summary written to: {SUMMARY_OUT}")

if __name__ == "__main__":
    main()
