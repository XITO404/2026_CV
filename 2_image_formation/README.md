## 과제 1 체크보드 기반 카메라 캘리브레이션
  - 이미지에서 체크보드 코너를 검출, 실제 좌표와 이미지 좌표의 대응 관계를 이용하여 카메라 파라미터 추정
  - 체크보드 패턴이 촬영된 여러 장의 이미지를 이용, 카메라 내부 행렬과 왜곡 계수를 계산하여 왜곡 보정
  
### 코드 
- 01_calibration.py

```python
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

    # 체크보드 이미지에서 체크보드 corner 검출
    # cv2.findChessboardCorners(image, corner 개수)
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
# cv2.calibrateCamera(): 체크보드 corner와 실제 좌표를 이용해서 카메라 파라미터 계산하는 함수
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
  ```



### 실행 결과

Camera Matrix K:<br>
[[536.07345314   0.         342.37046827]<br>
 [  0.         536.01636274 235.53687064]<br>
 [  0.           0.           1.        ]]<br>
<br>
Distortion Coefficients:<br>
[[-0.26509039 -0.0467422   0.00183302 -0.00031469  0.25231221]]



![과제 1 결과](./01_result.png)
<br><br>




## 과제 2 이미지 Rotation & Transformation
  - 한 장의 이미지에 회전, 크기 조절, 평행이동 적용

### 코드 
- 02_transformation.py
```python
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

    # 3. 평행이동 적용 (x:+80px, y:-40px)
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
```

### 실행 결과

![과제 2 결과](./02_result.png)
<br><br>




## 과제 3 Stereo Disparity 기반 Depth 추정
  - 같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 이미지를 이용해 깊이 추정
  - 두 이미지에서 같은 물체가 얼마나 옆으로 이동해 보이는지 계산, 물체가 카메라에서 얼마나 떨어져 있는지 (depth) 계산


### 코드 
- 03_depth.py
```python
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("left.png")
right_color = cv2.imread("right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 카메라 파라미터
f = 700.0  # 초점 거리
B = 0.12   # 베이스라인 (카메라 사이의 거리)

# ROI 설정 (x, y, w, h)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# numDisparities: 탐색할 시차 범위 (16의 배수)
# blockSize: 매칭 블록 크기 (홀수)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산 (Z = fB / d)
# -----------------------------
# disparity가 0 이하인 곳은 계산 불가능하므로 제외 (Zero Division 방지)
valid_mask = disparity > 0
depth_map = np.zeros_like(disparity)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 통계 계산
# -----------------------------
results = {}  # 결과 저장용

for name, (x, y, w, h) in rois.items():  # 각 ROI에 대해 반복
    roi_disp = disparity[y:y+h, x:x+w]   # ROI 영역의 Disparity 추출
    roi_depth = depth_map[y:y+h, x:x+w]  # ROI 영역의 Depth 추출
    
    # 유효한 픽셀 마스크 생성
    roi_valid = roi_disp > 0

    if np.any(roi_valid):  # 유효한 픽셀이 있을 경우
        mean_disp = np.mean(roi_disp[roi_valid])   # 유효 픽셀의 평균 Disparity
        mean_depth = np.mean(roi_depth[roi_valid]) # 유효 픽셀의 평균 Depth
    else:                  # 유효한 픽셀이 없을 경우
        mean_disp = 0.0    # 평균 Disparity 0으로 설정
        mean_depth = 0.0   # 평균 Depth 0으로 설정

    results[name] = {
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }                      # 딕셔너리에 결과 저장

# -----------------------------
# 4. 분석 결과 출력
# -----------------------------
print("\n[ Analysis Results ]")
for name, value in results.items():
    print(f"- {name}")
    print(f"  Avg Disparity : {value['mean_disparity']:.2f}")
    print(f"  Avg Depth     : {value['mean_depth']:.4f}")

# 통계 기반 거리 판단
target_near = max(results.items(), key=lambda x: x[1]["mean_disparity"])[0]
target_far = max(results.items(), key=lambda x: x[1]["mean_depth"])[0]

print("\n[ Summary ]")
print(f"Closest target  : {target_near}")
print(f"Farthest target : {target_far}")


# -----------------------------
# 5. 시각화 로직
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

d_min, d_max = np.nanpercentile(disp_tmp, 5), np.nanpercentile(disp_tmp, 95)
disp_scaled = np.clip((disp_tmp - d_min) / (d_max - d_min + 1e-6), 0, 1)
disp_vis = (disp_scaled * 255).astype(np.uint8)
disp_vis[np.isnan(disp_tmp)] = 0
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

depth_vis_map = np.zeros_like(depth_map, dtype=np.uint8)
if np.any(valid_mask):
    z_min, z_max = np.percentile(depth_map[valid_mask], 5), np.percentile(depth_map[valid_mask], 95)
    depth_scaled = np.clip((depth_map - z_min) / (z_max - z_min + 1e-6), 0, 1)
    depth_scaled = 1.0 - depth_scaled # 가까울수록 큰 값을 갖게 하여 빨간색 유도
    depth_vis_map[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_vis_map, cv2.COLORMAP_JET)

# ROI 표시
left_vis = left_color.copy()
for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 6. 저장 및 화면 출력
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "roi_result.png"), left_vis)

# 최종 확인용 시각화
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(left_vis, cv2.COLOR_BGR2RGB)); plt.title("ROI Selection")
plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB)); plt.title("Disparity Map")
plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)); plt.title("Depth Map")
plt.show()
```

### 실행 결과
Object     | Avg Disparity   | Avg Depth (m)  <br>
---------------------------------------------<br>
Painting   | 19.06           | 4.4248         <br>
Frog       | 33.60           | 2.5119         <br>
Teddy      | 22.42           | 3.8926         <br>
<br>
The closest object is 'Frog', and the farthest object is 'Painting'.
<br>
![과제 3 결과](./03_result.png)

<br><br>
