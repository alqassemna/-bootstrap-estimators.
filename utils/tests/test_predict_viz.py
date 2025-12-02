import os
import tempfile

import numpy as np
import pandas as pd

from utils.predict_viz import predict_order_completion_risk, visualize_data

def _make_synthetic(n=200, random_state=0):
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({
        "amount": rng.exponential(scale=200.0, size=n),
        "items_count": rng.poisson(lam=3.0, size=n),
        "status": rng.choice(["new", "processing", "on_hold", "complete"], size=n, p=[0.4, 0.3, 0.2, 0.1]),
    })
    # synthetic target correlated with amount and items
    df["expected_days_to_close"] = (df["amount"] / 100.0) + df["items_count"] * 0.5 + rng.normal(scale=2.0, size=n)
    return df

def test_predict_and_visualize_runs():
    df = _make_synthetic()
    df_pred = predict_order_completion_risk(
        df,
        amount_col="amount",
        items_col="items_count",
        status_col="status",
        target_col="expected_days_to_close",
        threshold=5.0,
        model_type="random_forest",
        compute_prediction_interval=True,
        n_bootstraps=10,  # small for test speed
        random_state=42,
    )
    # Basic assertions
    assert "predicted_days" in df_pred.columns
    assert "risk_flag" in df_pred.columns
    assert df_pred["predicted_days"].dtype.kind in ("f", "i")

    # Visualization should run without exception and may save file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        visualize_data(df_pred, customer_col=None, amount_col="amount", predicted_col="predicted_days", risk_col="risk_flag", group_by_customer=False, output_path=tmp.name)
        assert os.path.exists(tmp.name)
    finally:
        os.unlink(tmp.name)
