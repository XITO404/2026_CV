import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 0~255 사이의 픽셀 값을 0~1 사이로 정규화 (Normalization)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Sequential 모델 구축
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # 28x28 2차원 배열을 1차원 벡터로 평탄화
    layers.Dense(128, activation='relu'),   # 은닉층: 128개의 노드와 ReLU 활성화 함수 사용
    layers.Dense(10, activation='softmax')  # 출력층: 10개 숫자(0~9) 분류를 위해 10개 노드와 Softmax 사용
])

# 3. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 모델 훈련 (5회 반복)
history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, verbose=2)    # 한 줄씩 요약 출력

# 5. 모델 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n테스트 손실: {test_loss:.4f}')
print(f'\n테스트 정확도: {test_acc:.4f}')

# 6. 훈련 과정 시각화
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()