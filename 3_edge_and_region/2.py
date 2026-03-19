import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 연산을 위한 넘파이 라이브러리
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 이미지 불러오기
img = cv.imread('dabo.jpg')  # 이미지를 BGR 형태로 읽어옴
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일로 변환 (에지 검출 전처리 단계)

# 2. Canny 에지 검출
# threshold1=100, threshold2=200: 에지 판단을 위한 하한값과 상한값
edges = cv.Canny(gray, 100, 200)  # 에지 맵 생성

# 3. 허프 변환(HoughLinesP)을 사용하여 직선 검출
# rho=1: 거리 해상도 (1픽셀 단위)
# theta=np.pi/180: 각도 해상도 (1도 단위)
# threshold: 직선으로 판단하기 위한 최소 교차 횟수
# minLineLength: 검출할 직선의 최소 길이
# maxLineGap: 동일 선상의 점들을 하나의 직선으로 잇기 위한 최대 간격
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=40, maxLineGap=10)

# 4. 원본 이미지의 복사본 생성 (검출된 직선을 그리기 위함)
line_img = img.copy()  # 원본 이미지 데이터를 복제

# 5. 검출된 직선 정보를 바탕으로 이미지 위에 선 그리기
if lines is not None:  # 검출된 직선이 하나라도 있는 경우
    for line in lines:  # 검출된 모든 직선에 대해 반복
        x1, y1, x2, y2 = line[0]  # 직선의 시작점(x1, y1)과 끝점(x2, y2) 좌표 추출
        # cv.line(이미지, 시작점, 끝점, 색상, 두께)
        # (0, 0, 255): 빨간색 선
        # thickness=2: 선의 두께 2픽셀로 설정
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 이미지에 직선 그리기

# 6. Matplotlib 출력을 위해 openCV의 BGR 이미지를 RGB로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_lines_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)

# 7. Matplotlib을 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화
plt.figure(figsize=(12, 6))  # 전체 출력 창의 크기를 가로 12, 세로 6으로 설정

# --- 왼쪽: 원본 이미지 출력 ---
plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째 영역 선택
plt.imshow(img_rgb)  # 원본 이미지 표시
plt.title('Original Image')  # 그래프 제목 설정
plt.axis('off')  # 눈금 및 좌표축 숨기기

# --- 오른쪽: 직선 검출 결과 출력 ---
plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째 영역 선택
plt.imshow(img_lines_rgb)  # 직선이 그려진 결과 이미지 표시
plt.title('Hough Line Transform')  # 그래프 제목 설정
plt.axis('off')  # 눈금 및 좌표축 숨기기

# 7. 레이아웃 정렬 및 화면 출력 
plt.tight_layout()  # 이미지 간격 겹침 방지 및 자동 정렬
plt.show()  # 완성된 결과 창 띄우기