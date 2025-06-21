import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier # Um classificador simples para começar
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- 1. Carregar e Pré-processar Dados ---
try:
    df = pd.read_csv('sdss_data.csv', skiprows=1, low_memory=False)
    print(f"Dados brutos carregados: {len(df)} linhas")
except FileNotFoundError:
    print("Erro: sdss_data.csv não encontrado. Certifique-se de que o arquivo está na mesma pasta.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    exit()

df.columns = df.columns.str.lower()

# --- 2. Selecionar Características (Features) e Rótulo (Target) ---
# As magnitudes (u, g, r, i, z) são ótimas características numéricas.
# Podemos também calcular cores (diferença entre magnitudes).

# Assumindo que as colunas de magnitude são 'u', 'g', 'r', 'i', 'z'.
# Se no seu CSV forem 'psfmag_u', 'psfmag_g', etc., ajuste aqui.
MAGNITUDE_COLS = ['u', 'g', 'r', 'i', 'z']

# Verificar se as colunas de magnitude e 'class' existem
required_cols = MAGNITUDE_COLS + ['class']
if not all(col in df.columns for col in required_cols):
    print(f"Erro: Algumas colunas necessárias ({required_cols}) não encontradas no CSV.")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    print("Verifique o cabeçalho do seu sdss_data.csv e ajuste 'MAGNITUDE_COLS'.")
    exit()

# Remover linhas com valores NaN nas colunas relevantes
df.dropna(subset=required_cols, inplace=True)
df = df[df['redshift'] > 0] # Manter apenas objetos com redshift positivo (exclui algumas estrelas, etc.)

# Calcular algumas cores como características adicionais
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']

# Características a serem usadas para o modelo
features = MAGNITUDE_COLS + ['u_g', 'g_r', 'r_i', 'i_z']

X = df[features] # Matriz de características
y = df['class']  # Rótulo de classe (GALAXY, STAR, QSO)

print(f"Dados processados para ML: {len(X)} linhas.")
print(f"Características usadas: {features}")
print(f"Classes encontradas: {y.unique()}")

# --- 3. Codificar Rótulos de Texto para Números ---
# Modelos de ML precisam de rótulos numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Mapeamento para entender os resultados
class_names = label_encoder.classes_
print(f"Classes mapeadas para números: {list(zip(class_names, range(len(class_names))))}")

# --- 4. Dividir Dados em Treino e Teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

print(f"Dados para ML divididos: Treino={len(X_train)}, Teste={len(X_test)}")

# --- 5. Escalar Características ---
# Escalar as características é crucial para muitos algoritmos de ML
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. Construir e Treinar o Modelo ---
# Usaremos um classificador K-Nearest Neighbors (KNN) como exemplo, pois é simples e eficaz
print("Treinando o modelo K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(n_neighbors=5) # n_neighbors pode ser ajustado
knn_model.fit(X_train_scaled, y_train)
print("Treinamento concluído.")

# --- 7. Fazer Previsões e Avaliar o Modelo ---
y_pred = knn_model.predict(X_test_scaled)

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\n--- Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.show()

print("Classificação numérica de galáxias concluída.")