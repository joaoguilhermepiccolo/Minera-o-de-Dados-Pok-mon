import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

# Carregar o dataset (testando diferentes encodings)
print("Carregando dataset...")
try:
    df = pd.read_csv('pokemons_dataset.csv', encoding='utf-8')
    print("✅ Dataset carregado com encoding UTF-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv('pokemons_dataset.csv', encoding='iso-8859-1')
        print("✅ Dataset carregado com encoding ISO-8859-1")
    except UnicodeDecodeError:
        df = pd.read_csv('pokemons_dataset.csv', encoding='windows-1252')
        print("✅ Dataset carregado com encoding Windows-1252")

# Exploração inicial
print("\n=== INFORMAÇÕES BÁSICAS DO DATASET ===")
print(f"Número de registros: {len(df)}")
print(f"Número de colunas: {len(df.columns)}")
print(f"Colunas: {list(df.columns)}")
print("\n=== PRIMEIRAS 5 LINHAS ===")
print(df.head())
print("\n=== INFORMAÇÕES DOS TIPOS DE DADOS ===")
print(df.info())

# Análise exploratória mais detalhada
print("\n=== ANÁLISE EXPLORATÓRIA ===")
print("Distribuição dos tipos primários:")
print(df['Primary Type'].value_counts().head(10))

print("\nEstatísticas dos atributos:")
stats_cols = ['Attack', 'Defense', 'HP', 'Sp.Attack', 'Sp.Defense', 'Speed', 'Total']
print(df[stats_cols].describe())

# Verificar valores nulos
print("\nValores nulos por coluna:")
print(df.isnull().sum())

print("\n=== PREPARANDO PARA CLASSIFICAÇÃO ===")
# Vamos prever o Primary Type baseado nos stats
print("Tipos primários únicos:", df['Primary Type'].nunique())
print("Tipos mais comuns:")
print(df['Primary Type'].value_counts().head(8))

print("\n" + "="*50)
print("🤖 APLICANDO TÉCNICA DE CLASSIFICAÇÃO")
print("="*50)

# Preparar dados para classificação
# Objetivo: Prever o Primary Type baseado nos stats de batalha
X = df[['Attack', 'Defense', 'HP', 'Sp.Attack', 'Sp.Defense', 'Speed']].copy()
y = df['Primary Type'].copy()

print(f"✅ Dados preparados: {len(X)} registros com {len(X.columns)} características")
print(f"✅ Vamos prever {y.nunique()} tipos diferentes de Pokémon")

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"✅ Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste")

# Treinar modelo Random Forest
print("\n🔄 Treinando modelo Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Fazer predições
y_pred = rf_model.predict(X_test)

# Avaliar modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 RESULTADOS DA CLASSIFICAÇÃO:")
print(f"✅ Acurácia do modelo: {accuracy:.2%}")

print(f"\n📊 Relatório detalhado:")
print(classification_report(y_test, y_pred))

# Importância das características
feature_importance = pd.DataFrame({
    'Característica': X.columns,
    'Importância': rf_model.feature_importances_
}).sort_values('Importância', ascending=False)

print(f"\n⭐ IMPORTÂNCIA DAS CARACTERÍSTICAS:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['Característica']}: {row['Importância']:.3f}")