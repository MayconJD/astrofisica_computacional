import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# --- 1. Definir Parâmetros e Caminhos ---
DATA_DIR = '.' # Onde training_solutions.csv está
IMAGE_DIR = os.path.join(DATA_DIR, 'images') # Pasta onde as imagens foram descompactadas
IMAGE_SIZE = 64 # Redimensionar todas as imagens para 64x64 pixels (um tamanho menor acelera o treinamento)
NUM_CLASSES = 3 # 0=Elíptica, 1=Espirais, 2=Irregulares (Simplificação)
EPOCHS = 5 # Número de épocas de treinamento (pode aumentar para melhor resultado)
BATCH_SIZE = 32

# --- 2. Carregar Rótulos (Soluções) ---
solutions_df = pd.read_csv(os.path.join(DATA_DIR, 'training_solutions.csv'))
print(f"Dados de soluções carregados: {len(solutions_df)} linhas")

# --- Simplificação dos rótulos para 3 classes principais ---
# Esta é uma simplificação para fins didáticos e práticos.
# O dataset real do Galaxy Zoo tem probabilidades para muitas sub-classes.
# Aqui, usamos as probabilidades mais altas para as 3 categorias principais.
def get_simplified_label(row):
    if row['Class1.1'] >= 0.5: # Smooth (Suave) -> Elíptica
        return 0 # Elíptica
    elif row['Class2.1'] >= 0.5: # Spiral (Espiral) -> Espiral
        return 1 # Espiral
    elif row['Class7.1'] >= 0.5: # Irregular -> Irregular
        return 2 # Irregular
    else:
        return -1 # Para galáxias que não se encaixam claramente ou são de outras classes

solutions_df['simplified_label'] = solutions_df.apply(get_simplified_label, axis=1)

# Filtrar galáxias que não se encaixam nas 3 classes principais
solutions_df = solutions_df[solutions_df['simplified_label'] != -1].copy()

# Criar um mapeamento entre GalaxyID e Simplified_Label
galaxy_labels_map = solutions_df.set_index('GalaxyID')['simplified_label'].to_dict()

# --- 3. Carregar e Pré-processar Imagens ---
# Obter lista de IDs de galáxias que temos rótulos
galaxy_ids_with_labels = solutions_df['GalaxyID'].tolist()

images_list = []
labels_list = []

# Carregue apenas um subconjunto razoável de imagens para começar
# O dataset completo é muito grande e pode exigir muita RAM.
# Ajuste NUM_IMAGES_TO_PROCESS conforme sua RAM e tempo disponível.
NUM_IMAGES_TO_PROCESS = 10000 # Comece com 10.000, aumente se quiser mais dados

processed_count = 0
for galaxy_id in galaxy_ids_with_labels:
    if processed_count >= NUM_IMAGES_TO_PROCESS:
        break # Parar após processar o número desejado de imagens

    img_path = os.path.join(IMAGE_DIR, f"{galaxy_id}.jpg")
    
    if not os.path.exists(img_path):
        # print(f"Aviso: Imagem {img_path} não encontrada. Pulando.") # Descomente para ver avisos
        continue # Pular se a imagem não existir

    try:
        img = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        images_list.append(np.array(img))
        labels_list.append(galaxy_labels_map[galaxy_id])
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Carregadas {processed_count} imagens...")
    except Exception as e:
        print(f"Erro ao carregar imagem {img_path}: {e}. Pulando.")
        
if not images_list:
    print("Nenhuma imagem carregada. Verifique o caminho IMAGE_DIR e se o zip foi descompactado corretamente.")
    print("Certifique-se de que a pasta 'images' e 'training_solutions.csv' estão na mesma pasta do script.")
    exit()

X = np.array(images_list) / 255.0 # Normalizar pixels para [0, 1]
y = to_categorical(np.array(labels_list), num_classes=NUM_CLASSES) # One-hot encoding dos rótulos

# --- 4. Dividir Dados (Treino, Validação, Teste) ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Dados prontos: Treino={len(X_train)}, Validação={len(X_val)}, Teste={len(X_test)}")

# --- 5. Construir Modelo CNN ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Regularização para evitar overfitting
    Dense(NUM_CLASSES, activation='softmax') # Saída para 3 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 6. Treinar Modelo ---
print("Iniciando treinamento do modelo...")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val))
print("Treinamento concluído.")

# --- 7. Avaliar Modelo ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")

# --- 8. Visualizar Histórico de Treinamento ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda de Treino')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.tight_layout()
plt.show()

# --- 9. Fazer uma Predição (Exemplo) ---
if len(X_test) > 0:
    sample_image_index = np.random.randint(0, len(X_test))
    sample_image = X_test[sample_image_index]
    true_label_one_hot = y_test[sample_image_index]
    true_label = np.argmax(true_label_one_hot)

    # Labels mapeados para nomes
    label_names = {0: 'Elíptica', 1: 'Espiral', 2: 'Irregular'}

    prediction = model.predict(np.expand_dims(sample_image, axis=0))[0]
    predicted_label = np.argmax(prediction)

    plt.imshow(sample_image)
    plt.title(f"Real: {label_names.get(true_label, 'Desconhecido')}\nPredito: {label_names.get(predicted_label, 'Desconhecido')} (Confiança: {prediction[predicted_label]:.2f})")
    plt.axis('off')
    plt.show()
else:
    print("Conjunto de teste vazio para demonstração de predição.")

print("Classificador de galáxias com CNN concluído.")