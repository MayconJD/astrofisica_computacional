import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- 1. Carregar Dados ---
# *** LINHA CORRIGIDA AQUI (removendo sep=r'\s+') ***
try:
    df = pd.read_csv('sdss_data.csv', skiprows=1, low_memory=False)
    print(f"Dados carregados: {len(df)} linhas")
    print(df.head())
except FileNotFoundError:
    print("Erro: sdss_data.csv não encontrado. Certifique-se de que o arquivo está na mesma pasta.")
    print("Baixe um catálogo de galáxias simplificado ou use o CasJobs do SDSS para gerar um.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    print("Verifique se o arquivo está na pasta correta e se o formato corresponde ao esperado (cabeçalho na segunda linha, colunas separadas por vírgulas).")
    exit()

# Renomear colunas para consistência (minúsculas)
df.columns = df.columns.str.lower()

# Verifique as colunas esperadas. Pela sua saída, 'ra', 'dec', 'redshift' estão lá.
required_cols = ['ra', 'dec', 'redshift']
if not all(col in df.columns for col in required_cols):
    print(f"Erro: Colunas esperadas {required_cols} não encontradas no CSV.")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    exit()

# Remover linhas com valores NaN no redshift, se houver
df.dropna(subset=['redshift'], inplace=True)
df = df[df['redshift'] > 0] # Redshifts devem ser positivos

# --- 2. Converter Coordenadas (RA, Dec, Redshift para X, Y, Z) ---
H0 = 70.0 # Constante de Hubble (aproximado) em km/s/Mpc
c = 299792.458 # Velocidade da luz em km/s

df['distance_mpc'] = df['redshift'] * c / H0

ra_rad = np.deg2rad(df['ra'])
dec_rad = np.deg2rad(df['dec'])

df['x'] = df['distance_mpc'] * np.cos(dec_rad) * np.cos(ra_rad)
df['y'] = df['distance_mpc'] * np.cos(dec_rad) * np.sin(ra_rad)
df['z_cartesian'] = df['distance_mpc'] * np.sin(dec_rad)

# --- 3. (Opcional) Agrupamento Simples para "Aglomerados" ---
print("Executando K-Means para identificar aglomerados...")
kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
df['cluster_id'] = kmeans.fit_predict(df[['x', 'y', 'z_cartesian']])
print("Agrupamento concluído.")

# --- 4. Criar Visualização 3D Interativa com Plotly ---
print("Gerando visualização 3D interativa (pode levar alguns segundos)...")
fig = px.scatter_3d(df, x='x', y='y', z='z_cartesian',
                    color='cluster_id',
                    size_max=2, opacity=0.7,
                    title='Distribuição 3D de Galáxias e Aglomerados',
                    labels={'x': 'Distância X (Mpc)', 'y': 'Distância Y (Mpc)', 'z_cartesian': 'Distância Z (Mpc)'})

fig.update_traces(marker=dict(size=1))
fig.show()
print("Visualização gerada. Verifique seu navegador.")

# --- 5. (Opcional) Visualização 2D com Matplotlib (para referência) ---
print("Gerando visualização 2D com Matplotlib (feche para continuar)...")
plt.figure(figsize=(10, 8))
plt.scatter(df['x'], df['y'], s=1, c=df['cluster_id'], cmap='viridis', alpha=0.5)
plt.xlabel('X (Mpc)')
plt.ylabel('Y (Mpc)')
plt.title('Distribuição 2D de Galáxias (Projeção XY)')
plt.colorbar(label='ID do Aglomerado')
plt.grid(True)
plt.show()
print("Visualização 2D concluída.")