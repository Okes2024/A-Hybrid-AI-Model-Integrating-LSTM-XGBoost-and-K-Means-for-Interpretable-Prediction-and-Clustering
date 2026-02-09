
# Install required packages (uncomment if running locally)
# !pip install shap lime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("Water Parameters.csv")

# Prepare features and target
X = df.drop(columns=["FID", "Lat", "long", "Town", "pH"])
y_continuous = df["pH"]
binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
y = binner.fit_transform(y_continuous.values.reshape(-1, 1)).ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Report
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values[0], X_test)

# LIME
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['Acidic', 'Neutral', 'Alkaline'],
    mode='classification'
)

# Explain one instance
i = 0  # Index of instance to explain
exp = explainer_lime.explain_instance(X_test.values[i], model.predict_proba)
exp.show_in_notebook(show_table=True)
