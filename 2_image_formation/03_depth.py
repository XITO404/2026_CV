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
results = {}

for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    # 유효한 픽셀 마스크 생성
    roi_valid = roi_disp > 0

    if np.any(roi_valid):
        mean_disp = np.mean(roi_disp[roi_valid])
        mean_depth = np.mean(roi_depth[roi_valid])
    else:
        mean_disp = 0.0
        mean_depth = 0.0

    results[name] = {
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }

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