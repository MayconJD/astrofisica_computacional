import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# --- 1. Definir Caminhos e Parâmetros ---
IMAGE_DIR = 'images'  # Pasta onde as imagens estão (deve ser uma subpasta no mesmo diretório do script)
NUM_SAMPLES = 16      # Número de imagens para exibir (pode ajustar)
IMAGE_SIZE = 64       # Tamanho para redimensionar as amostras para exibição

# --- 2. Listar Todos os Arquivos de Imagem ---
# Verificar se o diretório de imagens existe
if not os.path.isdir(IMAGE_DIR):
    print(f"Erro: O diretório de imagens '{IMAGE_DIR}' não foi encontrado.")
    print("Certifique-se de que a pasta 'images' (com as imagens .jpg descompactadas) está na mesma pasta do script.")
    exit()

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
if not image_files:
    print(f"Erro: Nenhuma imagem .jpg encontrada na pasta '{IMAGE_DIR}'.")
    print("Certifique-se de que a pasta 'images' existe e contém as imagens descompactadas do Galaxy Zoo.")
    exit()

# --- 3. Selecionar Amostras Aleatórias ---
if len(image_files) < NUM_SAMPLES:
    sample_files = image_files # Se houver menos arquivos do que o desejado, usa todos
else:
    sample_files = random.sample(image_files, NUM_SAMPLES)

# --- 4. Criar a Visualização em Grade ---
rows = int(np.sqrt(NUM_SAMPLES))
cols = int(np.ceil(NUM_SAMPLES / rows))

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
axes = axes.flatten()  # Para facilitar a iteração sobre os subplots

for i, img_file in enumerate(sample_files):
    img_path = os.path.join(IMAGE_DIR, img_file)
    try:
        img = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        axes[i].imshow(img)
        # O nome do arquivo é o ID da galáxia. Remove a extensão .jpg
        axes[i].set_title(os.path.splitext(img_file)[0], fontsize=8) 
        axes[i].axis('off') # Desliga os eixos para uma visualização mais limpa
    except FileNotFoundError:
        print(f"Aviso: Arquivo não encontrado: {img_path}")
        axes[i].set_title("Não encontrado")
        axes[i].axis('off')
    except Exception as e:
        print(f"Erro ao abrir imagem {img_path}: {e}")
        axes[i].set_title("Erro")
        axes[i].axis('off')

# Remover quaisquer subplots vazios se NUM_SAMPLES não for um quadrado perfeito
for j in range(len(sample_files), NUM_SAMPLES):
    fig.delaxes(axes[j])

plt.tight_layout() # Ajusta o layout para evitar sobreposição
plt.show()

print("Visualização de amostras de galáxias concluída.")