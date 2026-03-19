import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 연산을 위한 넘파이 라이브러리
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 이미지 불러오기
img = cv.imread('coffee cup.jpg')

# 2. GrabCut을 위한 초기 마스크 생성 (이미지 크기와 동일하게 0으로 초기화)
mask = np.zeros(img.shape[:2], np.uint8)

# 3. GrabCut 알고리즘 내부에서 사용할 배경/전경 모델 생성 (힌트 반영)
bgdModel = np.zeros((1, 65), np.float64) # 배경 모델 초기화
fgdModel = np.zeros((1, 65), np.float64) # 전경 모델 초기화

# 4. 초기 사각형 영역(ROI) 설정: (x, y, width, height)
# 이미지 내의 커피컵이 포함되도록 적절한 좌표 설정 (이미지에 따라 조정 필요)
rect = (50, 50, img.shape[1]-100, img.shape[0]-100)

# 5. cv.grabCut()을 사용하여 반복적인 분할 수행 (5회 반복)
# cv.GC_INIT_WITH_RECT 모드로 사각형 기반 초기화를 수행함
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# 6. 마스크 값 처리 (cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD 사용)
# 확실한 배경(0)과 배경일 것 같은 곳(2)은 0으로, 전경(1, 3)은 1로 변경 (np.where 사용)
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 7. 원본 이미지에 마스크를 곱하여 배경 제거 (배경은 0이 곱해져 검은색이 됨)
# mask2는 1채널이므로 3채널로 확장하여 곱함
img_grabcut = img * mask2[:, :, np.newaxis]

# 8. Matplotlib을 사용하여 세 개의 이미지를 나란히 시각화
plt.figure(figsize=(15, 5))

# --- 왼쪽: 원본 이미지 ---
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# --- 가운데: 마스크 이미지 (0 또는 1로 구성된 결과) ---
plt.subplot(1, 3, 2)
plt.imshow(mask2, cmap='gray')
plt.title('Mask Image')
plt.axis('off')

# --- 오른쪽: 배경 제거 이미지 ---
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(img_grabcut, cv.COLOR_BGR2RGB))
plt.title('Background Removed')
plt.axis('off')

plt.tight_layout()
plt.show()