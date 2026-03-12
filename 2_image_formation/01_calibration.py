import cv2
import numpy as np
import glob
import os

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건 (반복 횟수 30회 또는 정밀도 0.001 도달 시 종료)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = [] # 3D 실제 세계 좌표
imgpoints = [] # 2D 이미지 내 좌표

images = glob.glob("./calibration_images/left*.jpg")
img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = (gray.shape[1], gray.shape[0])

    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너 검출에 성공한 경우에만 좌표 추가
    if ret == True:
        objpoints.append(objp)

        # 코너 위치 정밀화 (Sub-pixel accuracy)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 검출된 코너 시각화 (확인용)
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)

cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# ret: 재투영 오차, K: 내부 행렬, dist: 왜곡 계수, rvecs: 회전 벡터, tvecs: 이동 벡터
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 첫 번째 이미지를 불러와서 테스트
# 리스트의 첫 번째 이미지를 읽어와 테스트 진행
img = cv2.imread(images[0])
h, w = img.shape[:2]

# 왜곡 보정을 위한 최적의 새 카메라 행렬 계산
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

# 왜곡 보정 수행 (계산된 new_K 적용)
undistorted = cv2.undistort(img, K, dist, None, new_K)

# 결과 비교를 위해 원본과 보정본을 가로로 결합
comparison = np.hstack((img, undistorted))

# 화면 출력
cv2.imshow("Comparison", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()