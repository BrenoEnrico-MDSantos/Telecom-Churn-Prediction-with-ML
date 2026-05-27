import pandas as pd
import numpy as np

def processar_e_indexar_shap(df_clientes, df_shap, customer_id_col='Customer_ID'):
    """
    Une a base de clientes aos seus respectivos valores SHAP usando indexação 
    do Pandas e prepara o dataset para consumo otimizado no Power BI.
    """
    # 1. Garantir que o ID do Cliente seja o índice em ambos os DataFrames para busca O(1)
    df_clientes = df_clientes.set_index(customer_id_col)
    df_shap = df_shap.set_index(customer_id_col)
    
    # 2. Identificar e renomear as colunas SHAP para evitar colisões
    # Remove colunas de metadados se existirem no arquivo SHAP (ex: base_value se tratada separadamente)
    shap_cols = df_shap.columns
    df_shap_renamed = df_shap.rename(columns={col: f"shap_{col}" for col in shap_cols if not col.startswith('shap_')})
    
    # 3. Alinhamento e Junção via Índice (Inner join garante que apenas clientes com SHAP mapeado avancem)
    df_final = df_clientes.join(df_shap_renamed, how='inner')
    
    # 4. Formatação de Performance para o Power BI
    # Resetar o índice para que o Customer_ID volte a ser uma coluna comum utilizável em filtros/slicers
    df_final = df_final.reset_index()
    
    return df_final

# --- EXEMPLO DE EXECUÇÃO ---
if __name__ == "__main__":
    # Simulando dados brutos do CRM / Faturamento
    dados_clientes = pd.DataFrame({
        'Customer_ID': ['C101', 'C102', 'C103'],
        'Idade': [34, 45, 23],
        'Score_Credito': [710, 580, 690],
        'Tempo_Contrato': [12, 24, 3]
    })
    
    # Simulando a matriz SHAP exportada pelo modelo de Machine Learning
    # Importante: O valor base (geralmente o log-odds médio ou a probabilidade média) deve ser incluído
    dados_shap = pd.DataFrame({
        'Customer_ID': ['C101', 'C102', 'C103'],
        'Idade': [0.05, -0.12, 0.02],
        'Score_Credito': [-0.22, 0.35, -0.05],
        'Tempo_Contrato': [0.10, -0.08, 0.40],
        'shap_base_value': [0.15, 0.15, 0.15] # Valor de referência global do modelo
    })
    
    # Executa a indexação e união eficiente
    dataset_pbi = processar_e_indexar_shap(dados_clientes, dados_shap)
    
    # Exporta para o CSV final que alimentará o Power BI de forma ultra leve
    dataset_pbi.to_csv("mkt_churn_shap_ready.csv", index=False)
    print("Dataset unificado e indexado com sucesso para o Power BI!")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import shap

# 1. Trava de segurança: se o usuário selecionar múltiplos clientes na lista,
# exibe uma mensagem amigável em vez de tentar calcular ou quebrar.
if dataset.empty or dataset['Customer_ID'].nunique() > 1:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(0.5, 0.5, "Selecione um cliente na lista\npara ver os drivers de churn.", 
            color='#666666', fontsize=12, weight='bold', ha='center', va='center')
    ax.axis('off')
    plt.show()
    plt.close(fig)
else:
    # 2. Captura a linha do cliente selecionado
    row = dataset.iloc[0]
    
    # 3. Mapeamento dinâmico das colunas SHAP inseridas no visual
    # Filtra as colunas que começam com 'shap_' vindas da tabela relacionada
    shap_names = [col for col in dataset.columns if col.startswith('shap_') and col != 'shap_base_value']
    feature_names = [col.replace('shap_', '') for col in shap_names]
    
    # Se você trouxe as variáveis originais no dataset, mapeia os valores reais, senão deixa vazio
    # Procurar a coluna original correspondente (ex: procura 'Idade' se o shap for 'shap_Idade')
    feature_values = []
    for f in feature_names:
        if f in dataset.columns:
            feature_values.append(row[f])
        else:
            feature_values.append(np.nan) # Caso não queira exibir o valor numérico bruto ao lado do nome
            
    shap_values = row[shap_names].values.astype('float32')
    base_value = float(row['shap_base_value'])
    
    # 4. Reconstrói o objeto de explicação do SHAP
    exp = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_values if any(not np.isnan(x) for x in feature_values) else None,
        feature_names=feature_names
    )
    
    # 5. Renderização controlada do Matplotlib
    plt.rcParams['figure.dpi'] = 100
    fig = plt.figure(figsize=(7, 4.5))
    
    shap.waterfall_plot(exp, max_display=10, show=False)
    ax = plt.gca()
    
    # Formata o eixo X para porcentagem
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    
    # Limpa e formata os rótulos de impacto (+15%, -8%, etc.)
    for text in ax.texts:
        t_str = text.get_text()
        if any(char in t_str for char in ['=', '+', '-']):
            try:
                cleaned = t_str.replace('=', '').replace('+', '').replace('$', '').strip()
                val = float(cleaned)
                if -1.0 <= val <= 1.0 and val != 0:
                    prefix = "+" if val > 0 else ""
                    text.set_text(f"{prefix}{int(round(val * 100))}%")
            except ValueError:
                pass

    # 6. Título dinâmico baseado no cliente ativo
    plt.title(f"Principais Fatores de Churn - Cliente: {row['Customer_ID']}", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    
    # 7. Exibe e limpa a memória imediatamente
    plt.show()
    plt.close(fig)

# MUST UNPIVOT SHAP TABLE BEFORE 

# 1. Create a base identifier dataframe
id_df = pd.DataFrame({
    'Customer_ID': joined_df['Customer_ID'].values,
    'Churn_Prob': joined_df['Churn_Score'].values,
    'shap_base_value': base_values_col
})

# 2. Add ID to features and melt (Wide -> Long)
features_df['Customer_ID'] = id_df['Customer_ID']
features_long = features_df.melt(
    id_vars=['Customer_ID'], 
    var_name='Feature_Name', 
    value_name='Feature_Value'
)

# 3. Add ID to SHAP values and melt (Wide -> Long)
# Strip the 'shap_' prefix during melt so the feature names match exactly
shap_df['Customer_ID'] = id_df['Customer_ID']
shap_long = shap_df.melt(
    id_vars=['Customer_ID'], 
    var_name='Feature_Name', 
    value_name='SHAP_Value'
)
shap_long['Feature_Name'] = shap_long['Feature_Name'].str.replace('shap_', '', regex=False)

# 4. Merge the long dataframes together
long_features_merged = pd.merge(
    features_long, 
    shap_long, 
    on=['Customer_ID', 'Feature_Name']
)

# 5. Bring back the global metrics (Churn_Prob and Base Value)
final_export_df = pd.merge(id_df, long_features_merged, on='Customer_ID')

print(f'⌛ Long-format table created in {round(time.time() - start,2)} secs!')