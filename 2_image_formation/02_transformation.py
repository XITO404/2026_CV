import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 로드
img = cv2.imread('rose.png')

if img is None:
    print("이미지를 찾을 수 없습니다.")
else:
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # 2. 회전 및 크기 조절 행렬 생성 (+30도, 0.8배)
    M = cv2.getRotationMatrix2D(center, 30, 0.8)

    # 3. 평행이동 적용 (x:+80, y:-40)
    M[0, 2] += 80
    M[1, 2] -= 40

    # 4. Affine 변환 적용
    dst = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    # 5. Matplotlib을 사용하여 동시에 시각화
    plt.figure(figsize=(12, 6))

    # 왼쪽: 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 오른쪽: 변환된 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title('Rotation + Scale + Translation')
    plt.axis('off')

    plt.tight_layout()
    plt.show()