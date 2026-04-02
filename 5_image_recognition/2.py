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