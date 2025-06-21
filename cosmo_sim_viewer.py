import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Gerar Dados de Simulação Sintética ---
def generate_mock_cosmic_structure(num_points=10000, num_clusters=5, cluster_density_factor=5):
    """
    Gera um conjunto de pontos 3D que simulam estruturas cósmicas.
    Inclui um 'campo' de pontos aleatórios e alguns 'aglomerados' densos.
    """
    data = np.random.rand(num_points, 3) * 100 # Pontos aleatórios no cubo [0, 100]^3

    # Adicionar alguns aglomerados (grupos densos de pontos)
    cluster_centers = np.random.rand(num_clusters, 3) * 100

    for center in cluster_centers:
        # Gerar pontos em torno do centro com uma distribuição gaussiana
        num_points_in_cluster = num_points // (num_clusters * 2) # Ajuste para ter mais ou menos pontos

        # Adicionar mais pontos no cluster para densidade
        cluster_points = np.random.randn(num_points_in_cluster * cluster_density_factor, 3) * 5 + center
        data = np.vstack((data, cluster_points))

    # Opcional: adicionar um "filamento" rudimentar
    # Ajuste os parâmetros para controlar a forma do filamento
    filament_length = 100
    num_filament_points = 200
    t = np.linspace(0, 1, num_filament_points) # Parâmetro de 0 a 1

    # Exemplo de um filamento curvo
    filament_x = t * filament_length
    filament_y = np.sin(t * np.pi * 4) * 15 + (filament_length / 2)
    filament_z = np.cos(t * np.pi * 2) * 10 + (filament_length / 2)

    # Adicionar ruído ao filamento
    filament_x += np.random.randn(num_filament_points) * 2
    filament_y += np.random.randn(num_filament_points) * 2
    filament_z += np.random.randn(num_filament_points) * 2

    filament_coords = np.hstack((filament_x.reshape(-1, 1), filament_y.reshape(-1, 1), filament_z.reshape(-1, 1)))
    data = np.vstack((data, filament_coords))

    print(f"Gerados {len(data)} pontos simulados.")
    return data

# Gerar os dados
sim_data = generate_mock_cosmic_structure(num_points=20000, num_clusters=8, cluster_density_factor=7)

# --- 2. Visualização 3D com Plotly (Interativo - Recomendado) ---
print("Gerando visualização 3D interativa com Plotly...")
fig_plotly = go.Figure(data=[go.Scatter3d(
    x=sim_data[:, 0],
    y=sim_data[:, 1],
    z=sim_data[:, 2],
    mode='markers',
    marker=dict(
        size=1,        # Tamanho do marcador
        opacity=0.7,   # Transparência
        color=sim_data[:, 2], # Colorir por coordenada Z para um gradiente visual
        colorscale='Viridis', # Esquema de cores
        colorbar_title="Z Coordinate"
    )
)])

fig_plotly.update_layout(
    title='Visualização 3D de Estrutura Cósmica Simulada (Plotly)',
    scene=dict(
        xaxis_title='X Axis (Mpc)',
        yaxis_title='Y Axis (Mpc)',
        zaxis_title='Z Axis (Mpc)',
        aspectmode='cube' # Mantém as proporções para uma visualização cúbica
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig_plotly.show()
print("Visualização Plotly gerada. Verifique seu navegador.")

# --- 3. Visualização 3D com Matplotlib (Estático/Básico) ---
print("Gerando visualização 3D estática com Matplotlib (feche para continuar)...")
fig_mpl = plt.figure(figsize=(10, 8))
ax_mpl = fig_mpl.add_subplot(111, projection='3d')

ax_mpl.scatter(sim_data[:, 0], sim_data[:, 1], sim_data[:, 2],
               s=0.5, # Tamanho do ponto
               alpha=0.5, # Transparência
               c=sim_data[:, 2], # Colorir por coordenada Z
               cmap='viridis')

ax_mpl.set_xlabel('X Axis (Mpc)')
ax_mpl.set_ylabel('Y Axis (Mpc)')
ax_mpl.set_zlabel('Z Axis (Mpc)')
ax_mpl.set_title('Visualização 3D de Estrutura Cósmica Simulada (Matplotlib)')

# Definir limites para manter proporções
max_range = np.array([sim_data[:,0].max()-sim_data[:,0].min(), 
                      sim_data[:,1].max()-sim_data[:,1].min(), 
                      sim_data[:,2].max()-sim_data[:,2].min()]).max() / 2.0

mid_x = (sim_data[:,0].max()+sim_data[:,0].min()) * 0.5
mid_y = (sim_data[:,1].max()+sim_data[:,1].min()) * 0.5
mid_z = (sim_data[:,2].max()+sim_data[:,2].min()) * 0.5

ax_mpl.set_xlim(mid_x - max_range, mid_x + max_range)
ax_mpl.set_ylim(mid_y - max_range, mid_y + max_range)
ax_mpl.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
print("Visualização Matplotlib concluída.")
print("Visualizador de simulações concluído.")