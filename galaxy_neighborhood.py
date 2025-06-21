import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- 1. Carregar Dados ---
# *** LINHA DE CARREGAMENTO ADAPTADA PARA SDSS_DATA.CSV ***
try:
    df = pd.read_csv('sdss_data.csv', skiprows=1, low_memory=False) # Adaptação para sdss_data.csv
    print(f"Dados carregados: {len(df)} galáxias do SDSS")
    print(df.head())
except FileNotFoundError:
    print("Erro: sdss_data.csv não encontrado. Certifique-se de que o arquivo está na mesma pasta.")
    print("Baixe o catálogo do SDSS (usado no Projeto 1) ou verifique o nome do arquivo.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    print("Verifique se o arquivo sdss_data.csv está na pasta correta e no formato esperado.")
    exit()

# Renomear colunas para consistência (minúsculas)
df.columns = df.columns.str.lower()

# *** VERIFICAR E TRATAR NOMES DE COLUNAS ***
# O seu sdss_data.csv deve ter 'ra', 'dec', 'redshift', 'g' e 'r'.
# Se ele tiver 'psfmag_g' e 'psfmag_r' (magnitudes de PSF), é melhor usá-las para cores.
# Caso contrário, 'g' e 'r' referem-se a modelMag_g e modelMag_r (fluxo do modelo),
# que também podem ser usados para cor, mas magnitudes de PSF são comuns.
# Para este exemplo, vou assumir que as colunas 'g' e 'r' (lowercase) são as magnitudes que você precisa.
# Se no seu CSV as colunas de magnitude são 'psfmag_g' e 'psfmag_r', altere as linhas abaixo.
# Ou se for 'u', 'g', 'r', 'i', 'z' de modelFlux ou petroFlux, ajuste para os nomes corretos.

required_cols = ['ra', 'dec', 'redshift', 'g', 'r'] # Adicionando 'g' e 'r' para calcular a cor
if not all(col in df.columns for col in required_cols):
    print(f"Erro: Colunas esperadas {required_cols} não encontradas no CSV do SDSS.")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    print("Verifique o cabeçalho do seu sdss_data.csv e ajuste 'required_cols' e o cálculo de 'g_r_color'.")
    exit()

# Remover linhas com valores NaN no redshift, g, ou r, se houver
df.dropna(subset=['redshift', 'g', 'r'], inplace=True)
df = df[df['redshift'] > 0] # Redshifts devem ser positivos

# --- 2. Calcular a Cor g-r ---
# *** LINHA ADICIONADA PARA CALCULAR A COR ***
df['g_r_color'] = df['g'] - df['r'] # Calcule a cor g-r

# --- 3. Converter Coordenadas (RA, Dec, Redshift para X, Y, Z) ---
# Usando as mesmas constantes de conversão do Projeto 1 e 4
H0 = 70.0  # Constante de Hubble em km/s/Mpc
c = 299792.458 # Velocidade da luz em km/s

df['distance_mpc'] = df['redshift'] * c / H0
ra_rad = np.deg2rad(df['ra'])
dec_rad = np.deg2rad(df['dec'])

df['x'] = df['distance_mpc'] * np.cos(dec_rad) * np.cos(ra_rad)
df['y'] = df['distance_mpc'] * np.cos(dec_rad) * np.sin(ra_rad)
df['z_cartesian'] = df['distance_mpc'] * np.sin(dec_rad)

# Coordenadas a serem usadas para a árvore KDTree
coords = df[['x', 'y', 'z_cartesian']].values

# --- 4. Criar KDTree para Busca de Vizinhos ---
print("Construindo KDTree...")
kdtree = KDTree(coords)
print("KDTree construída.")

# --- 5. Calcular Densidade de Vizinhança ---
# Raio para buscar vizinhos (em Mpc). Ajuste este valor.
# Um raio típico para aglomerados é ~1 Mpc
SEARCH_RADIUS_MPC = 1.0 # Exemplo: 1 Megaparsec

# Inicializar coluna para densidade de vizinhança
df['n_neighbors'] = 0

print(f"Calculando número de vizinhos dentro de {SEARCH_RADIUS_MPC} Mpc para cada galáxia...")
# Para cada galáxia, conte quantos outros objetos estão dentro do raio
# Este loop pode demorar um pouco com 100.000 galáxias.
for i, galaxy_coords in enumerate(coords):
    neighbors_indices = kdtree.query_ball_point(galaxy_coords, SEARCH_RADIUS_MPC)
    df.loc[i, 'n_neighbors'] = len(neighbors_indices) - 1 # Subtrair 1 para não contar a própria galáxia
    if i % 5000 == 0 and i > 0: # Imprimir progresso a cada 5000 galáxias
        print(f"Processadas {i} galáxias...")

df['n_neighbors'] = df['n_neighbors'].astype(int) # Garantir que n_neighbors é int

print("Cálculo de vizinhança concluído.")
print(df.head())

# --- 6. Análise e Visualização ---

plt.figure(figsize=(15, 6))

# Gráfico 1: Histograma da Densidade de Vizinhança
plt.subplot(1, 2, 1)
bins = np.arange(df['n_neighbors'].min(), df['n_neighbors'].max() + 2) - 0.5
sns.histplot(df['n_neighbors'], bins=bins, kde=False)
plt.title('Distribuição do Número de Vizinhos')
plt.xlabel(f'Número de Vizinhos dentro de {SEARCH_RADIUS_MPC} Mpc')
plt.ylabel('Frequência')
# Ajustar xticks para números inteiros apenas nos bins
plt.xticks(np.arange(df['n_neighbors'].min(), df['n_neighbors'].max() + 1, 5)) # Pode ajustar o passo (5)

# Gráfico 2: Cor da Galáxia vs. Número de Vizinhos (scatter plot)
plt.subplot(1, 2, 2)
# 'g_r_color' agora deve estar disponível
sns.scatterplot(x='n_neighbors', y='g_r_color', data=df, alpha=0.3, s=10) # Ajuste alpha e s para muitos pontos
plt.title('Cor da Galáxia vs. Número de Vizinhos')
plt.xlabel(f'Número de Vizinhos dentro de {SEARCH_RADIUS_MPC} Mpc')
plt.ylabel('Cor (g-r)')
plt.grid(True)
plt.xticks(np.arange(df['n_neighbors'].min(), df['n_neighbors'].max() + 1, 5)) # Ajuste o passo

plt.tight_layout()
plt.show()

# Visualização 3D colorida pela densidade de vizinhança com Plotly
print("Gerando visualização 3D de galáxias coloridas por densidade de vizinhança...")
fig_3d = px.scatter_3d(df, x='x', y='y', z='z_cartesian',
                        color='n_neighbors', # Colore os pontos pelo número de vizinhos
                        size_max=2, opacity=0.7,
                        title=f'Galáxias do SDSS coloridas pelo Nº de Vizinhos ({SEARCH_RADIUS_MPC} Mpc)',
                        labels={'x': 'X (Mpc)', 'y': 'Y (Mpc)', 'z_cartesian': 'Z (Mpc)'},
                        color_continuous_scale=px.colors.sequential.Plasma)

fig_3d.update_traces(marker=dict(size=1)) # Ajusta o tamanho dos pontos para melhor visualização
fig_3d.show()
print("Análise de vizinhança concluída com dados do SDSS.")