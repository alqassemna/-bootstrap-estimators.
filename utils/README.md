# Predictive Analytics & Visualization Utilities

This module provides:
- `predict_order_completion_risk`: robust pipeline to predict days-to-close and flag risky orders.
- `visualize_data`: readable dashboard-style plotting.

Quick start:
1. Install dependencies: scikit-learn, pandas, numpy, matplotlib, seaborn, joblib
2. Use functions:

```python
from utils.predict_viz import predict_order_completion_risk, visualize_data
df = pd.read_csv("orders.csv")
df_pred = predict_order_completion_risk(df, model_type="random_forest", compute_prediction_interval=True, n_bootstraps=50)
visualize_data(df_pred, output_path="dashboard.png")
```
