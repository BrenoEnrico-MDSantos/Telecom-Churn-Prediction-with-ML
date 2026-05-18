import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from category_encoders import WOEEncoder

np.random.seed(42)
n_total = 20000

# 1. SETUP DE DADOS E ENCODER
df = pd.DataFrame({
    'Customer_ID': [f'CUST_{i:05d}' for i in range(n_total)],
    'Cluster': np.random.choice(['Premium', 'Budget', 'Standard', 'Retention_Risk'], n_total, p=[0.15, 0.35, 0.40, 0.10]),
    'State': np.random.choice(['SP', 'RJ', 'MG', 'RS'], n_total, p=[0.45, 0.25, 0.15, 0.15]),
    'Tenure_Months': np.random.randint(1, 72, n_total),
    'Monthly_Charges': np.random.uniform(30, 250, n_total),
    'Total_Revenue': np.random.uniform(100, 5000, n_total) # Coluna que será ignorada por não estar no CPH
})

y_ficticio_inicial = np.random.binomial(1, 0.3, n_total)
encoder = WOEEncoder(cols=['Cluster', 'State'])
df_encoded = encoder.fit_transform(df, y_ficticio_inicial)

# =========================================================================
# CLASSE MOCK PARA SIMULAR O cph_final APÓS O FIT (Substitua pelo seu modelo real)
# =========================================================================
class MockCPH:
    def __init__(self):
        # Apenas colunas de interesse do negócio participam. 'Total_Revenue' não está aqui.
        self.params_ = pd.Series({
            'Cluster': 1.25,
            'State': 0.85,
            'Tenure_Months': -0.04,
            'Monthly_Charges': 0.005,
            'treatment_1': -0.55,
            'treatment_2': -0.65,
            'treatment_3': -0.40
        })
cph_final = MockCPH()
# =========================================================================

# 2. VETORIZAÇÃO COMPLETA DO RISCO BASE (Utilizando apenas as colunas do CPH)
colunas_cph = [col for col in df_encoded.columns if col in cph_final.params_.index]
log_hazard_base = df_encoded[colunas_cph].dot(cph_final.params_[colunas_cph])
df['Baseline_Churn_Prob'] = 1 / (1 + np.exp(-log_hazard_base))

df_piloto, df_holdout = train_test_split(df, test_size=0.5, random_state=42)
df_piloto, df_holdout = df_piloto.copy(), df_holdout.copy()

# 3. RANDOMIZAÇÃO RESTRITA OPERACIONAL
cph_elegibilidade = pd.DataFrame({
    0: [1, 1, 1, 1], 1: [0, 1, 1, 0], 2: [1, 0, 1, 0], 3: [1, 0, 0, 1]
}, index=['Premium', 'Budget', 'Standard', 'Retention_Risk'])

matriz_elegibilidade = cph_elegibilidade.loc[df_piloto['Cluster']].values
pesos_aleatorios = np.random.uniform(0, 1, matriz_elegibilidade.shape)
df_piloto['treatment'] = np.argmax(matriz_elegibilidade * pesos_aleatorios, axis=1)

# 4. CÁLCULO VETORIZADO DO EFEITO DE INTERVENÇÃO (AUTOMATIZADO VIA cph_final)
df_piloto_encoded = encoder.transform(df_piloto)
tratamentos_piloto = df_piloto['treatment'].values

# Busca os coeficientes de tratamento específicos gerados pelo cph_final (Ex: treatment_1, treatment_2...)
cols_tratamento_cph = [f'treatment_{t}' for t in tratamentos_piloto]
coefs_tratamento = np.array([cph_final.params_.get(col, 0.0) for col in cols_tratamento_cph])

# O efeito interage diretamente com o risco proporcional (WoE) do cliente de forma matricial
df_piloto['efeito_intervencao'] = (df_piloto_encoded['Cluster'] * coefs_tratamento) + np.random.normal(0, 0.02, len(df_piloto))
df_piloto['Churn_Prob_Real'] = np.clip(df_piloto['Baseline_Churn_Prob'] + df_piloto['efeito_intervencao'], 0, 1)
df_piloto['Churn_Label'] = np.random.binomial(1, df_piloto['Churn_Prob_Real'])

# 5. TREINAMENTO DO MODELO OPERACIONAL DE UPLIFT (S-LEARNER)
X_piloto = encoder.transform(df_piloto[['Cluster', 'State', 'Tenure_Months', 'Monthly_Charges']])
X_piloto['treatment'] = df_piloto['treatment']
y_piloto = df_piloto['Churn_Label']

model_uplift = XGBClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
model_uplift.fit(X_piloto, y_piloto)

# 6. ENGENHARIA DE MATRIZ SCENARIO (VETORIZAÇÃO TOTAL NO HOLDOUT)
df_holdout_encoded = encoder.transform(df_holdout[['Cluster', 'State', 'Tenure_Months', 'Monthly_Charges']])
n_holdout = len(df_holdout_encoded)
lista_estrategias = [0, 1, 2, 3]

# Cria uma super-matriz empilhando todos os cenários de tratamento para remover loops
X_cenarios_completos = pd.concat([
    df_holdout_encoded.assign(treatment=t) for t in lista_estrategias
], axis=0)

# Predição em lote (Batch Prediction) - infinitamente mais rápido que for loops
todas_predicoes = model_uplift.predict_proba(X_cenarios_completos)[:, 1]
predicoes_divididas = todas_predicoes.reshape(len(lista_estrategias), n_holdout)

# Organiza as colunas de Uplift de forma dinâmica
for t in lista_estrategias[1:]:
    df_holdout[f'Uplift_Pacote_{t}'] = predicoes_divididas[0] - predicoes_divididas[t]

# 7. TOMADA DE DECISÃO DETERMINÍSTICA VETORIZADA (MÁXIMO UPLIFT)
matriz_uplifts = df_holdout[[f'Uplift_Pacote_{t}' for t in lista_estrategias[1:]]].values
max_uplifts = np.max(matriz_uplifts, axis=1)
indices_max_uplift = np.argmax(matriz_uplifts, axis=1) + 1

df_holdout['Melhor_Ação_Automatica'] = np.where(
    max_uplifts <= 0, 
    'Sem Intervenção (Controle)', 
    f'Enviar Pacote ' + pd.Series(indices_max_uplift).astype(str).values
)

# 8. PROFILING DE ENTREGA PARA O NEGÓCIO
tree_profiler = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=42)
tree_profiler.fit(df_holdout_encoded, df_holdout['Melhor_Ação_Automatica'])

print("--- Distribuição no Holdout ---")
print(df_holdout['Melhor_Ação_Automatica'].value_counts(normalize=True) * 100)
print("\n--- Regras de Profiling ---")
print(export_text(tree_profiler, feature_names=list(df_holdout_encoded.columns)))




