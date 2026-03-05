import cv2 as cv
import numpy as np
import sys

# 1. cv.imread()를 사용하여 이미지 로드
img=cv.imread('./soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 2. cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# np.hstack()을 위해 그레이스케일 이미지를 3채널(BGR) 형식으로 변환 (차원 맞추기)
gray_3channel=cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 3. np.hstack() 함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결
combined_img=np.hstack((img, gray_3channel))

# 4. cv.imshow()와 cv.waitKey()를 사용해 결과를 표시
# cv.imshow('Original vs Grayscale', combined_img)

# 5. 이미지 사이즈를 30%로 축소 및 저장 (너무 커서 잘리는 현상 수정)
resized_img = cv.resize(combined_img, (0, 0), fx=0.3, fy=0.3)
cv.imshow('Result', resized_img)
cv.imwrite('./soccer_result.jpg', resized_img)
# 아무 키나 누르면 창이 닫히도록 설정
cv.waitKey()
cv.destroyAllWindows()