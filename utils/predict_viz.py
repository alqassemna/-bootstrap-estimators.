"""
Enhanced Predictive Analytics and Visualization utilities.

Usage example:
    df = pd.read_csv("orders.csv")
    df = predict_order_completion_risk(
        df, threshold=5.0, model_type="random_forest",
        compute_prediction_interval=True, n_bootstraps=100
    )
    visualize_data(df, group_by_customer=True, output_path="dashboard.png")
"""
from typing import Optional, Tuple, Union, Sequence
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed

sns.set_style("whitegrid")
logger = logging.getLogger(__name__)


def _build_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    model_type: str = "linear",
    random_state: int = 42,
):
    """Construct a fresh sklearn Pipeline (preprocessor + regressor)."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # drop="if_binary" keeps binary encoding compact but will be ignored for multi-cat
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ],
        remainder="drop",
    )

    if model_type == "linear":
        reg = LinearRegression()
    elif model_type == "random_forest":
        reg = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError("model_type must be 'linear' or 'random_forest'")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", reg)])
    return pipeline


def predict_order_completion_risk(
    df: pd.DataFrame,
    *,
    amount_col: str = "amount",
    items_col: str = "items_count",
    status_col: str = "status",
    target_col: str = "expected_days_to_close",
    threshold: float = 5.0,
    model_type: str = "linear",  # choices: "linear", "random_forest"
    compute_prediction_interval: bool = False,
    n_bootstraps: int = 100,
    random_state: int = 42,
    save_model_path: Optional[str] = None,
    return_model: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Pipeline]]:
    """
    Train a regression model to predict days-to-close and flag risky orders.

    Returns:
    - df_out (or (df_out, model) if return_model=True): DataFrame copy with added columns.
    """
    logger.info("Running Enhanced Predictive Analytics on Sales Orders")
    df_in = df.copy()

    required_cols = {amount_col, items_col, status_col, target_col}
    missing = required_cols - set(df_in.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # Drop rows where target is missing (can't train)
    initial_len = len(df_in)
    df_in = df_in.dropna(subset=[target_col])
    if len(df_in) < initial_len:
        logger.info("Dropped %d rows with missing target '%s'.", initial_len - len(df_in), target_col)

    numeric_features = [amount_col, items_col]
    categorical_features = [status_col]

    pipeline = _build_pipeline(numeric_features, categorical_features, model_type, random_state)

    X = df_in[numeric_features + categorical_features]
    y = df_in[target_col].astype(float)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )

    # Fit
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_test = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    r2 = r2_score(y_test, y_pred_test)
    logger.info("Model (%s) evaluation on test set: MAE=%.3f, RMSE=%.3f, R2=%.3f", model_type, mae, rmse, r2)

    # Cross-validated score (safely)
    try:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        cv_mae = -float(np.mean(cv_scores))
        logger.info("5-fold CV MAE: %.3f", cv_mae)
    except Exception as e:
        logger.debug("Cross-validation failed/skipped: %s", e)

    # Predict on full dataset
    X_full = df_in[numeric_features + categorical_features]
    preds = pipeline.predict(X_full)
    preds = np.where(preds < 0, 0.0, preds)

    df_out = df_in.copy()
    df_out["predicted_days"] = preds
    df_out["risk_score"] = df_out["predicted_days"] / float(threshold)
    df_out["risk_flag"] = df_out["predicted_days"] > float(threshold)

    if compute_prediction_interval:
        logger.info(
            "Computing bootstrap prediction intervals with n_bootstraps=%d (this may take a while)...",
            n_bootstraps
        )
        rng = np.random.RandomState(random_state)
        n_samples = len(X_full)
        all_boot_preds = np.zeros((n_bootstraps, n_samples), dtype=float)

        # Reset index to allow integer-based sampling
        X_train_full = X_train.reset_index(drop=True)
        y_train_full = y_train.reset_index(drop=True)

        def _fit_and_predict(seed_idx: int):
            "Fit a cloned pipeline on a bootstrap sample and predict on X_full."
            # ensure different randomness per bootstrap
            rs = rng.randint(0, 2**31 - 1)
            idxs = np.random.RandomState(rs).randint(0, len(X_train_full), size=len(X_train_full))
            Xb = X_train_full.iloc[idxs]
            yb = y_train_full.iloc[idxs]
            try:
                cloned_pipeline = _build_pipeline(numeric_features, categorical_features, model_type, random_state)
                # If reg has random_state, the builder above already set it; otherwise clone for safety
                cloned_pipeline.fit(Xb, yb)
                return cloned_pipeline.predict(X_full)
            except Exception as exc:
                logger.debug("Bootstrap fit failed: %s", exc)
                # fallback to main predictions
                return preds

        # Run in parallel; keep n_jobs to 1 if n_bootstraps is small
        n_jobs = -1 if n_bootstraps > 10 else 1
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_and_predict)(i) for i in range(n_bootstraps)
        )
        for i, r in enumerate(results):
            all_boot_preds[i, :] = r

        lower = np.percentile(all_boot_preds, 2.5, axis=0)
        upper = np.percentile(all_boot_preds, 97.5, axis=0)
        lower = np.where(lower < 0, 0.0, lower)

        df_out["pred_lower"] = lower
        df_out["pred_upper"] = upper
        df_out["risk_flag_conservative"] = df_out["pred_upper"] > float(threshold)

    if save_model_path:
        try:
            joblib.dump(pipeline, save_model_path)
            logger.info("Trained pipeline saved to: %s", save_model_path)
        except Exception as e:
            logger.warning("Could not save model to %s: %s", save_model_path, e)

    logger.info("Prediction complete. Added 'predicted_days', 'risk_score', and 'risk_flag' columns.")
    if compute_prediction_interval:
        logger.info("Also added 'pred_lower', 'pred_upper', and 'risk_flag_conservative' columns.")

    if return_model:
        return df_out, pipeline
    return df_out


def visualize_data(
    df: pd.DataFrame,
    *,
    customer_col: str = "customer_name",
    amount_col: str = "amount",
    predicted_col: str = "predicted_days",
    risk_col: str = "risk_flag",
    group_by_customer: bool = True,
    top_n: Optional[int] = 20,
    figsize: tuple = (12, 8),
    cmap: str = "RdYlGn_r",  # reversed so red = risky
    annotate: bool = True,
    output_path: Optional[str] = None,
):
    """
    Generate a readable dashboard-style plot.

    Returns nothing; shows or saves a figure.
    """
    logger.info("Generating Enhanced Data Visualization")
    required = {customer_col, amount_col, predicted_col, risk_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for visualization: {missing}")

    df_viz = df.copy()

    if group_by_customer:
        agg = df_viz.groupby(customer_col).agg(
            total_amount=(amount_col, "sum"),
            avg_predicted_days=(predicted_col, "mean"),
            risk_rate=(risk_col, "mean"),
            n_orders=(predicted_col, "size"),
        ).reset_index()
        df_plot = agg.sort_values("total_amount", ascending=False)
    else:
        # ensure we have columns named consistently for plotting
        if predicted_col not in df_viz.columns:
            df_viz["avg_predicted_days"] = df_viz.get(predicted_col, np.nan)
        else:
            df_viz["avg_predicted_days"] = df_viz[predicted_col]
        if risk_col not in df_viz.columns:
            df_viz["risk_rate"] = df_viz.get(risk_col, 0).astype(float)
        else:
            df_viz["risk_rate"] = df_viz[risk_col].astype(float)
        df_plot = df_viz.rename(columns={amount_col: "total_amount"}).sort_values("total_amount", ascending=False)

    if top_n:
        df_plot = df_plot.head(top_n)

    # Ensure consistent column references
    if "total_amount" not in df_plot.columns:
        if amount_col in df_plot.columns:
            df_plot = df_plot.rename(columns={amount_col: "total_amount"})
        else:
            raise ValueError("Could not find a column to use for total amount in plotting.")

    df_plot = df_plot.sort_values("total_amount", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(df_plot))

    norm = plt.Normalize(0, 1)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(df_plot.get("risk_rate", df_plot.get(risk_col, 0).astype(float))))

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.barh(y_positions, df_plot["total_amount"], color=colors, edgecolor="k", alpha=0.9)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(df_plot.get(customer_col, df_plot.index.astype(str)))
    ax1.set_xlabel("Sales Amount (USD)")
    ax1.set_title("Total Sales Amount per Customer (colored by risk rate)")

    if annotate and len(df_plot) > 0:
        max_amount = df_plot["total_amount"].max()
        offset = max_amount * 0.005 if max_amount > 0 else 0.01
        for i, v in enumerate(df_plot["total_amount"]):
            ax1.text(v + offset, i, f"${v:,.0f}", va="center", fontsize=9)

    ax2 = ax1.twiny()
    ax2.scatter(df_plot.get("avg_predicted_days", df_plot.get(predicted_col, np.nan)), y_positions,
                color="blue", s=40, zorder=10, label="Avg Predicted Days")
    for xi, yi in zip(df_plot.get("avg_predicted_days", []), y_positions):
        try:
            ax2.plot([0, xi], [yi, yi], color="blue", alpha=0.15, linewidth=0.8)
        except Exception:
            continue

    ax2.set_xlabel("Average Predicted Days to Close")
    ax2.legend(loc="upper right")

    # Add a colorbar to communicate the mapping
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("Risk Rate (0=no risk, 1=all risky)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info("Figure saved to %s", output_path)

    plt.show()
    logger.info("Visualization complete.")
