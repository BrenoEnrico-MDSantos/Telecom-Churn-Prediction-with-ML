import pandas as pd
import shap

# 1. Force the explainer to calculate in probability space natively
# NOTE: Passing feature data (e.g., X_train or X_recently_joined) is required 
# by SHAP when model_output="probability" to calculate conditional expectations.
explainer = shap.TreeExplainer(model, data=X_recently_joined, model_output="probability")
shap_values_obj = explainer(X_recently_joined)

# 2. Extract raw features, SHAP values, and the base value
# (The rest of your code works perfectly as written!)
features_df = pd.DataFrame(X_recently_joined, columns=X_recently_joined.columns).reset_index(drop=True)
shap_matrix = shap_values_obj.values 

# 3. Create a DataFrame for the SHAP values with a 'shap_' prefix
shap_df = pd.DataFrame(shap_matrix, columns=[f"shap_{col}" for col in features_df.columns])

# 4. Extract the base value
if isinstance(shap_values_obj.base_values, (list, tuple, type(shap_matrix))):
    base_values = shap_values_obj.base_values
else:
    base_values = [shap_values_obj.base_values] * len(features_df)

meta_df = pd.DataFrame({
    'Customer_ID': ids_recently_joined,  
    'Churn_Prob': churn_probabilities,   
    'shap_base_value': base_values
}).reset_index(drop=True)

# 5. Concatenate everything horizontally
final_export_df = pd.concat([meta_df, features_df, shap_df], axis=1)
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
