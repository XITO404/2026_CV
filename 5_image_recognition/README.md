## 과제 1 간단한 이미지 분류기 구현
- 손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구

### 요구사항
- MNIST 데이터셋을 로드
- 데이터를 훈련 세트와 테스트 세트로 분할
- 간단한 신경망 모델을 구축
- 모델을 훈련시키고 정확도를 평가

### 힌트
- tensorflow.keras.datasets에서 MNIST 데이터셋을 불러올 수 있음
- Sequential 모델과 Dense 레이어를 활용하여 신경망을 구성
- 손글씨 숫자 이미지는 28x28 픽셀 크기의 흑백 이미지

<details>
<summary><h3><b>코드 - 1.py</b></h3></summary>
<div markdown="1">

```python
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
```

</div>
</details>

### 실행 결과

![과제 1 결과](./1_result.png)
<br><br>

![과제 1 시각화](./Figure_1.png)
<br>


---
## 과제 2 CIFAR-10 데이터셋을 활용한 CNN 모델 구축
- CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고 이미지 분류를 수행

### 요구사항
- CIFAR-10 데이터셋을 로드
- 데이터 전처리(정규화 등)를 수행
- CNN 모델을 설계하고 훈련
- 모델의 성능을 평가하고, 테스트 이미지(dog.jpg)에 대한 예측을 수행

### 힌트
- tensorflow.keras.datasets에서 CIFAR-10 데이터셋을 불러올 수 있음
- Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN을 구성
- 데이터 전처리 시 픽셀 값을 0~1 범위로 정규화하면 모델의 수렴이 빨라질 수 있음


<details>
<summary><h3><b>코드 - 2.py</b></h3></summary>
<div markdown="1">

```python
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np
import cv2 as cv

# 1. CIFAR-10 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 픽셀 값 정규화 (0~1 범위)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 클래스 이름 정의 (예측 결과 확인용)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'horse', 'ship', 'truck', 'frog']

# 2. CNN 모델 설계
model = models.Sequential([
    # 특징 추출부 (Convolutional Base)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 분류부 (Classifier)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10개 클래스 분류
])

# 3. 모델 컴파일 및 훈련
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("CNN 모델 훈련 시작")
model.fit(x_train, y_train, epochs=10, validation_split=0.1, verbose=2)

# 4. 모델 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')

# 5. 이미지(dog.jpg) 예측 수행
try:
    # 이미지 로드 및 전처리
    img = cv.imread('dog.jpg')
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # BGR -> RGB
    img_resized = cv.resize(img_rgb, (32, 32))   # 모델 입력 크기에 맞게 조절
    img_normalized = img_resized / 255.0         # 정규화
    img_input = np.expand_dims(img_normalized, axis=0) # 4차원 배열로 변환 (1, 32, 32, 3)

    # 예측
    predictions = model.predict(img_input)
    score = np.argmax(predictions)
    
    print(f"\n예측 결과: {class_names[score]} ({predictions[0][score]*100:.2f}%)")
except Exception as e:
    print(f"\n이미지 예측 중 오류 발생: {e}")
```

</div>
</details>

### 실행 결과

![과제 2 결과](./2_result.png)
<br><br>


