# Bibliotecas
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

# Parâmetros
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
NUM_CLASSES = 4
EPOCHS = 1
DATASET_DIR = 'datasets'

# Nomes das classes
class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Aplicar filtros espaciais
def apply_spatial_filters(img):
    img = img.astype(np.float32) / 255.0  # normaliza valores para 0-1
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)  # bordas horizontais
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)  # bordas verticais
    combined = np.stack([img, sobel_x, sobel_y], axis=-1)  # cria imagem 3 canais
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)  # normalização
    return combined

# Função para carregar imagens e rótulos de um diretório
def load_dataset_from_dir(base_dir):
    images = []
    labels = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(base_dir, class_name)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)   # leitura em escala de cinza
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) # redimensionamento
            img = apply_spatial_filters(img)               # aplica filtros espaciais
            images.append(img)
            labels.append(label_idx)                       # atribui rótulo da classe
    return np.array(images, dtype=np.float32), np.array(labels)

# Carregamento dos conjuntos de treino, validação e teste
X_train, y_train = load_dataset_from_dir(os.path.join(DATASET_DIR, 'train'))
X_val, y_val = load_dataset_from_dir(os.path.join(DATASET_DIR, 'val'))
X_test, y_test = load_dataset_from_dir(os.path.join(DATASET_DIR, 'test'))

# Conversão dos rótulos para one-hot encoding
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

# Data Augmentation (aumenta a diversidade das imagens de treino)
train_datagen = ImageDataGenerator(
    rotation_range=15,        # pequenas rotações
    width_shift_range=0.1,    # deslocamento horizontal
    height_shift_range=0.1,   # deslocamento vertical
    zoom_range=0.1,           # zoom aleatório
    horizontal_flip=True      # inversão horizontal
)

val_datagen = ImageDataGenerator()   # sem aumento de dados (validação)
test_datagen = ImageDataGenerator()  # sem aumento de dados (teste)

# Geradores de dados para treino, validação e teste
train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val_cat, batch_size=BATCH_SIZE)
test_generator = test_datagen.flow(X_test, y_test_cat, batch_size=BATCH_SIZE, shuffle=False)

# Definição da Rede Neural Convolucional (CNN)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # evita overfitting
    Dense(NUM_CLASSES, activation='softmax')  # camada de saída (classificação)
])

# Compilação do modelo (otimizador, função de perda e métricas)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Resumo da arquitetura do modelo
model.summary()

# Treinamento do modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Avaliação do modelo no conjunto de teste
test_preds = model.predict(X_test)  # gera previsões
test_preds_classes = np.argmax(test_preds, axis=1)  # pega a classe de maior probabilidade

# Relatório de métricas
print("Relatório de classificação:")
print(classification_report(y_test, test_preds_classes, target_names=class_names))

# Matriz de confusão
cm = confusion_matrix(y_test, test_preds_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Função para visualizar previsões do modelo
def plot_predictions(X, y_true, y_pred, class_names, n=9):
    plt.figure(figsize=(12,12))
    indices = np.random.choice(len(X), n)
    for i, idx in enumerate(indices):
        plt.subplot(3,3,i+1)
        img = X[idx][:,:,0]
        plt.imshow(img, cmap='gray')
        plt.title(f"Real: {class_names[y_true[idx]]}, Previsto: {class_names[y_pred[idx]]}")
        plt.axis('off')
    plt.show()

# Exibe algumas previsões
plot_predictions(X_test, y_test, test_preds_classes, class_names)
