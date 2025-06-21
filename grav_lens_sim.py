import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Para carregar/salvar imagens
from scipy.ndimage import map_coordinates # Para interpolação

# --- Parâmetros ---
IMAGE_SIZE = 256 # Tamanho da imagem (pixels x pixels)
LENS_STRENGTH = 50 # Força da lente (ajuste para ver diferentes distorções, valores maiores = mais distorção)
LENS_CENTER_X = IMAGE_SIZE // 2
LENS_CENTER_Y = IMAGE_SIZE // 2

# --- 1. Criar ou Carregar Imagem de Fundo (Fonte) ---
def create_grid_image(size):
    img = np.zeros((size, size), dtype=np.uint8) # Fundo preto
    # Adicionar linhas de grade
    for i in range(0, size, size // 10):
        img[i, :] = 255
        img[:, i] = 255
    # Adicionar um círculo no centro
    center_x, center_y = size // 2, size // 2
    radius = size // 8
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    img[dist_from_center < radius] = 128 # Cor cinza para o círculo
    return img

def load_image(path, size):
    try:
        img_pil = Image.open(path).convert('L') # Abrir e converter para escala de cinza
        img_pil = img_pil.resize((size, size)) # Redimensionar
        return np.array(img_pil)
    except FileNotFoundError:
        print(f"Erro: Imagem '{path}' não encontrada. Gerando grade de exemplo.")
        return create_grid_image(size)
    except Exception as e:
        print(f"Erro ao carregar imagem: {e}. Gerando grade de exemplo.")
        return create_grid_image(size)

# Escolha entre gerar grade ou carregar imagem
# Se você tem uma imagem 'background_galaxy.png' na mesma pasta, ela será usada.
# Caso contrário, uma grade será gerada.
source_image = load_image('background_galaxy.png', IMAGE_SIZE) 

# Se você QUISER FORÇAR a grade, descomente a linha abaixo e comente a de cima:
# source_image = create_grid_image(IMAGE_SIZE)

# --- 2. Preparar Grade de Coordenadas ---
# Coordenadas do plano de deflexão (onde a lente está)
x_lens = np.arange(IMAGE_SIZE) - LENS_CENTER_X
y_lens = np.arange(IMAGE_SIZE) - LENS_CENTER_Y
X_lens, Y_lens = np.meshgrid(x_lens, y_lens)

# Distância radial do centro da lente
R_lens = np.sqrt(X_lens**2 + Y_lens**2)
# Evitar divisão por zero no centro da lente
R_lens[R_lens == 0] = 1e-9 # Pequeno valor para evitar erro de divisão por zero

# --- 3. Calcular Vetor de Deflexão (Modelo Simplificado) ---
# Vetor de deflexão (alfa_x, alfa_y) para cada ponto no plano da lente
# Uma forma simplificada onde o desvio é inversamente proporcional à distância
# Multiplicamos por LENS_STRENGTH para controlar a magnitude da distorção
alpha_x = -LENS_STRENGTH * (X_lens / R_lens**2)
alpha_y = -LENS_STRENGTH * (Y_lens / R_lens**2)

# --- 4. Mapeamento Inverso ---
# Coordenadas do plano da fonte (onde o pixel deveria vir da imagem original)
# x_source = x_deflected - alpha_x
# y_source = y_deflected - alpha_y
# As coordenadas X_lens, Y_lens já são as coordenadas no plano de deflexão (distância do centro da lente)
source_coords_x = (X_lens + alpha_x) + LENS_CENTER_X # Adiciona o offset de volta para coordenadas absolutas
source_coords_y = (Y_lens + alpha_y) + LENS_CENTER_Y # Adiciona o offset de volta para coordenadas absolutas

# --- 5. Interpolar para Criar Imagem Distorcida ---
# Use map_coordinates para obter os valores de pixel da imagem original
# `order=1` para interpolação bilinear (mais suave)
# `cval=0` para preencher pixels fora dos limites com preto
lensed_image = map_coordinates(source_image, [source_coords_y, source_coords_x], order=1, cval=0)

# --- 6. Visualização ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Imagem Original (Fonte)')
plt.imshow(source_image, cmap='gray', origin='lower') # 'origin='lower'' para consistência se a imagem for cartesiana
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagem Lenteada (Distorted)')
plt.imshow(lensed_image, cmap='gray', origin='lower')
plt.axis('off')

plt.tight_layout()
plt.show()

print("Simulação de lente gravitacional concluída.")