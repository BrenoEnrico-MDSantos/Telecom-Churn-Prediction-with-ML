# Telecom Churn Prediction with ML
Churner prediction using binary classifiers in python, alongside PowerBI for report and A/B testing.

## Table of contents
1. [Key Info](#key-info)
2. [Cluster Profiling](#cluster-profiling)

### Key info
- The original table has **6418 rows, and 32 columns**; the features used to segment clusters + analyse survivability curves were treated and scaled to achieve valuable results (**5 clusters**, and good Churn Scoring).
-  The formula for ```Total Revenue``` (total past revenue thus far) is: ```Total_Revenue = (Total_Charges + Total_Long_Distance_Charges + Total_Extra_Data_Charges) - Total_Refunds```. As such, for readability, the column **Monthly_Avg** was made from ```Total_Revenue / Tenure_in_Months```, to represent an understandable approximation of customer financial value.   
- By grouping with clusters made with **Contract, Tenure, Revenue and Internet Type**, actionable routes for prioritizing campaigns can be divised, without excessive cost or high risk of mistaken allocation;
- An *Uplift* analysis could very well be implemented further on, but simulating A/B testing and weighted reactions to different campaigns excels the scope of this Churn Prediction project.
- ```Value_Deal``` appears to identify customers that already were subject to some bundle or campaign; since I have no access to the cost of each contract and/or services, the referred column was ignored in the analysis, also for it's composition (only 2870 non-nulls). So, this column is seen in the operational dashboard, as customer info.
- The means of ```Number_of_Referrals``` by clusters are too homogenous; same treatment as Value_Deal.
- **Value-Added Services (VAS)** play a major role in this project and in real industry practice, hence all of them were considered to determine behavioral diffs and improvement points, even for states.
- All **status-related columns** hold crucial data, but ```Churn_Category``` and ```Churn_Reason``` have too many nulls; same treatment as Value_Deal, but those cols might aid in regional tailoring.

| Dimension Type | Columns |
|---|---|
| Profile | ```'Customer_ID', 'Gender', 'Age', 'Married', 'State', 'Number_of_Referrals', 'Tenure_in_Months', 'Value_Deal', 'Payment_Method', 'Paperless_Billing'``` |
| Core Services | ```'Phone_Service', 'Multiple_Lines', 'Internet_Service', 'Internet_Type'``` |
| VAS | ```'Online_Security', 'Online_Backup', 'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 'Streaming_Movies', 'Streaming_Music', 'Unlimited_Data'``` |
| Financial | ```'Monthly_Charge', 'Total_Charges', 'Total_Refunds', 'Total_Extra_Data_Charges', 'Total_Long_Distance_Charges', 'Total_Revenue'``` |
| Status & Churn | ```'Customer_Status', 'Churn_Category', 'Churn_Reason'``` |

- In the final cust. table (Churn + Stayed + predicted Joined), these new columns are present:

| New Columns |
| --- |
| ```'Clusters', 'Age_Range', 'Tenure_Range', 'RLV', 'Predicted'  ``` |

### Cluster Profiling

| No. | Cluster | Profile | Action | Churn Risk (L/M/H) | Value (L/M/H) | Regional Priority |
|---|---|---|---|---|---|---|
| 1 | *Loyalist* | Fiber-heavy who trusts and expects quality + VAS; strong survivability and highest ARPU point a **massive long-term RLV**; younger public **(20-40s)**| *Gamification* of contract (discount by tenure, reward feedback giving); stretch longer contract lock w/ premium bundling; focus on expanding this profile across states and ages. | 🟢 L | 🟢 H | Grow on southern states (e.g. *Tamil Nadu, Kerala, Karnataka*) |
| 2 | *Minimalist* | Older, **off-line** public; no fluff; long contracts, expect phone calls to work; lowest ARPU, RLV falls from 6th month onwards; high survivability  | Cost-efficient maintenance; warrant reliability on phone services and land support; slow transition into basic internet infrastructure, and bundle with family incetives; ensure means for physical payment while nudging towards paperless, and keep the recently joined in | 🔴 H | 🔴 L | Critical at *Uttar Pradesh, Telangana, Assam* |
| 3 | *Savvy Nomad* | Largest customer count and revenue sum; comprises most ages, especially 40-70s; **squishy** (Month-to-Month); appreciates VAS; massive base and promising ARPU make for RLV priority, but got the worst surivavl curve | Crucial onboarding on early tenure (0-3 months), and diplomatic **One-Year contract lock** w/ incetives; MKT efforts to avoid rival *poaching*; emphasize Quality of Life add-ons | 🔴 H | 🟢 H | Immediate act in *Jammu & Kashmir, Uttar Pradesh, Telangana, Bihar and Maharashtra*; rework VIP plans and mobilize client support teams. Due to cluster size, tailored age-related approaches are needed |
| 4 | *Steady Surfer* | Dislikes phone services, and likes some on-line VAS; more mature public (40-60s; stable survivability, low-mid ARPU and mid RLV; second largest sum of revenue, worth treating | If costs allow, try phone upsell; **professional life** MKT approaches; gradual migration to Fiber Optic | 🟡 M | 🟡 M | Implement bundling and improve/fix DSL in *Tamil Nadu, Kerala, Karnataka, Maharashtra*; consider appealing "stay-bonuses" for *Nomad* and *Surfer* clusters |
| 5 | *Zen Tech* | Cable prerefence; seeks peace of mind with **security add-ons**; survivalibilty concern second only to *Nomad*'s ; small current slice for total revenue; mid RLV and ARPU; another mid group, small priority | Bundle other VAS with highlighted security upsides, and fix potential flaws in provided plans; slowly work phone provision; Fiber probably isn't worth the cost | 🟡 M | 🔴 L | *Maharashtra, Punjab, Bihar Andhra Pradesh, Madhya Pradesh* |

> [!NOTE]
> The original table has no **cost** info, rendering impossible to calculate metrics such as standard **Customer Lifetime Value, Cost of Acquisition, Return Over Investment** and other indicators that in a real scenario back up interventions. *"Wins"* tracking can *somewhat* be deduced by the amount of recently Joined customers, but again, table integration would be needed for precise calculations.
> Nonetheless, with the available data, it is already possible to detect distinct patterns, groups, regional preferences and moment to act.

## :ringed_planet: To use K-Medoids
- The library ```scikit-learn-extra``` is available only in **python <= 3.11**. Use a custom venv as kernel for projects, and install libraries with ```%p``` (due to compatibility, ```%pip install "numpy<2"```)
- Check all versions installed at CMD with ```py -0``` (asterisk marks the current)
    - ```dir``` for checking created venvs
- Install pip and upgrade if needed: ```python -m ensurepip --upgrade```
- Kernel creation: ```pip install ipyernel```, then ```python -m ipykernel install --user --name=venv_kmedoids --display-name "Python 3.11 (KMedoids)"```
    - For checking all kernels, ```jupyter kernelspec list```, and to uninstall: ```jupyter kernelspec uninstall venv_kmedoids```
- Activate the venv ```venv_kmedoids\Scripts\activate```
- Open JupyterLab: ```jupyter lab``` or [URL](http://localhost:8888/lab?token=04f81664309fa076b10351753c52ec47dd26265eca202945)
- Create the project with new kernel _Python 3.11 (KMedoids)_. Remember to use ```%pip install```

Uplift posting: [here](https://cmr.berkeley.edu/2025/11/to-treat-or-not-to-treat-five-lessons-learned-from-using-uplift-modeling-to-optimize-marketing-campaigns/)

# Steps
1. Extract -> Load -> Transform:
    - Understand columns for each tool, and create an average for direct monetary reading; 
    - Outlier detection and dealing with Tukey method;
    - Standardize numerical data with StandardScaler for FAMD;
2. Customer Clustering and Profiling:
    - FAMD coordinates with high explainability components fed on KMedoids model;
    - Cluster distribution and means/modes display;
    - Survivability curves with Kaplan-Meier Fitter;
    - Laplace Smoothed Odds Ratio + Lift for cluster prominent traits;
    - Labeling and expert domain strategy with AI;
        -  'State' table with top drivers
        - CLV and Residual CLV calculations (profiles based on heatmap)
3. Churn Prediction for Recently Joined:
    - Weight of Evidence calculations for event of churn;
    - COX PH Hazard Ratios with relevant Information Value columns;
    - XGBoost + Optuna with unscaled numericals and WOEEncoded categoricals;
        - SHAP values for feature understanding;
    - 'Joined' status customers churn profiling with optimal threshold;
        - Custom function for individual cases with SHAP waterfall
4. Uplift Analysis on customers at risk:
    - A/B groups (simulated intervention and response across clusters);
    - The predicted churn labeling becomes the target for WOEEncoder;
    - Uplift must be paired with Residual CLV for even stronger priority 
    -  Calculation of Uplift and profiling to assess feasibility of interventions;
5. PBI Dashboards for churn situation, clusters, states and campaign predictions.   

## AI for Cluster Analysis:
These prints were passed for Gemini, which acted as a market expert and suggested labels for each created segment; the **Chain of Thought** was:
> _"Act as a Telecom market expert. Analyze these cluster preferences, distribution, survivability curves and Residual CLV runways. Brainstorm at least 3 names for each of the segments applying domain knowledge, and suggest measures for customer retention, prioritizing those at highest risk. Let's think this step by step."_

Based on the provided data, here is an expert analysis of your five customer segments, focusing on their churn risk, financial potential, and strategic retention measures.
------------------------------
## 1. Segment Profiling & Naming

| Cluster | Key Characteristics | Risk Level | CLV Runway |
|---|---|---|---|
| 1 | DSL, Month-to-month, No Phone, High support usage. | High | Mid-range |
| 2 | No Internet, Two-Year Contract, Low spend. | Lowest | Low/Stable |
| 3 | Fiber Optic, One-Year Contract, High spend ($270 avg). | Low | Highest |
| 4 | Fiber Optic, Month-to-month, Heavy data/streaming. | Critical | Mid-High |
| 5 | Cable, Month-to-month, No Phone, High Security/Support. | High | Mid-range |

## Segment Names

* Cluster 1: The Legacy Data-Onlys, DSL Lifeliners, The Minimalist Streamers.
* Cluster 2: The Secure Seniors, Voice-First Traditionalists, Low-Maintenance Loyals.
* Cluster 3: Digital Elites, High-Value Power Users, Premium Tech-Adopters.
* Cluster 4: The Flighty Gamers, High-Speed Nomads, Month-to-Month Maximizers.
* Cluster 5: Cable Cord-Cutters, Secure Surfers, Niche Utility Users.

------------------------------
## 2. Survival & Risk Analysis
The Survival Curves clearly show that Cluster 4 is your biggest leak. The probability of retention for this group drops sharply after month 10, with almost half the cluster churning by month 30.
Cluster 1 and 5 follow a similar downward trajectory. These month-to-month users represent "at-will" customers who lack the "contractual handcuffs" of Clusters 2 and 3.
------------------------------
## 3. Prioritized Retention Measures## Priority 1: Cluster 4 (The Critical Flight Risk)
Despite being the largest group (34.8%), they have the highest churn volume.

* Contract Conversion: Offer a "Loyalty Discount" or a free speed upgrade for moving from Month-to-Month to a 12-month commitment.
* Bundling Incentives: Since they "dislike" streaming TV, offer a bundle that includes high-value services they do use (like Unlimited Data) at a locked-in price.
* Proactive "Save" Desk: Use the 10-15 month mark (where the survival curve dips) to trigger automated "Customer Appreciation" offers.

## Priority 2: Cluster 3 (The High-Value Growth Opportunity)
These customers have the highest RCLV ($9k+ in the 7-12m window). You cannot afford to lose even one.

* Concierge Support: Provide a dedicated "Gold" support line to maintain their high monthly spend ($270).
* Hardware Refresh: Offer early router/modem upgrades at the 18-month mark to keep them on the cutting edge of Fiber Optic tech.

## Priority 3: Clusters 1 & 5 (The Service-Heavy DSL/Cable Users)
These users value security and support but are on month-to-month plans.

* Service Bundling: Integrate their "Liked" features (Online Security/Premium Support) into a slightly discounted annual plan.
* Infrastructure Migration: Transition Cluster 1 (DSL) to Fiber/Cable where available. DSL is a high-churn technology; moving them to a more modern infrastructure increases perceived value and stickiness.

------------------------------
## 4. RCLV Strategic Summary
The RCLV Heatmap indicates that your "Gold Mine" is Cluster 3 in the 0-12 month tenure range. If you can keep them past the first year, their residual value remains higher than any other group. Conversely, Cluster 2 is very stable but offers low financial upside, making them a "maintenance" segment rather than a "growth" segment.
Would you like to drill down into a specific marketing campaign draft for the high-risk Cluster 4 users?
