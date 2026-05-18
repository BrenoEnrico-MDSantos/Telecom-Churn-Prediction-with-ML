import shap
import pandas as pd

# 1. Amostragem do background para acelerar o cálculo intercional
bg_sample = shap.sample(X_joined_enc, 200) if len(X_joined_enc) > 200 else X_joined_enc

explainer = shap.TreeExplainer(
    best_model, 
    data=bg_sample, 
    model_output="raw", 
    feature_perturbation="interventional"
)

# 2. Cálculo dos valores SHAP
shap_values_obj = explainer(X_joined_enc)

# 3. Extração e conversão para float32 (economiza 50% de memória)
shap_matrix = shap_values_obj.values.astype('float32')

# 4. Reaproveita o DataFrame existente evitando cópias
features_df = X_joined_enc.reset_index(drop=True)

# 5. Criação direta do DataFrame SHAP
shap_cols = [f"shap_{col}" for col in features_df.columns]
shap_df = pd.DataFrame(shap_matrix, columns=shap_cols)

# 6. Extração otimizada do valor base (suporta escalar ou array)
base_val = shap_values_obj.base_values
if isinstance(base_val, (list, tuple, type(shap_matrix))) and len(base_val) == len(features_df):
    base_values_col = base_val
else:
    base_values_col = base_val[0] if hasattr(base_val, "__len__") else base_val

# 7. Concatenação direta e eficiente
final_export_df = pd.concat([
    pd.Series(joined_df['Customer_ID'].values, name='Customer_ID'),
    pd.Series(joined_df['Churn_Score'].values, name='Churn_Prob'),
    pd.Series(base_values_col, name='shap_base_value', index=features_df.index),
    features_df,
    shap_df
], axis=1)

print(final_export_df.info())

final_export_df.to_csv("pbi_operational_churn_data.csv", index=False)

Once your table is imported into Power BI, create a Python Visual, drag your columns into its field bucket, and use this exact script. Because XGBoost models often deal with dozens of features, we will add max_display=10 to keep the chart clean, legible, and uncluttered for the marketing team:

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if not dataset.empty:
    # 1. Grab the single customer row
    row = dataset.iloc[0]
    
    # 2. Dynamic Mapping (Matches raw features to their shap_ columns perfectly)
    shap_names = [col for col in dataset.columns if col.startswith('shap_') and col != 'shap_base_value']
    feature_names = [col.replace('shap_', '') for col in shap_names]
    
    feature_values = row[feature_names].values
    shap_values = row[shap_names].values
    base_value = row['shap_base_value']
    
    # 3. Rebuild the SHAP Explanation object
    exp = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_names
    )
    
    # 4. Generate the plot on an explicit Matplotlib Axis
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.waterfall_plot(exp, max_display=10, show=False)
    
    # 5. FORCE PERCENTAGE FORMATTING ON THE X-AXIS AND TEXT LABELS
    # Convert x-axis ticks to percentages (e.g., 0.2 -> 20%)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    
    # Find the text labels inside the chart (+/- impacts and numbers) and format them
    for text in ax.texts:
        try:
            val = float(text.get_text().replace('=', '').replace('+', '').strip())
            # If it's a small decimal value, turn it into an integer percentage string
            if -1.0 <= val <= 1.0 and val != 0:
                prefix = "+" if val > 0 else ""
                text.set_text(f"{prefix}{int(round(val * 100))}%")
        except ValueError:
            pass # Skip labels that aren't numeric (like feature values)

    # 6. Visual polish
    plt.title(f"Churn Drivers for Customer {row['Customer_ID']}", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()
