## 과제 1 소벨 에지 검출 및 결과 시각화
- edgeDetectionImage 이미지를 그레이스케일로 변환
- Sobel 필터를 사용하여 x축과 y축 방향의 에지를 검출
- 검출된 에지 강도 이미지를 시각화

### 요구사항
- cv.imread()를 사용하여 이미지를 불러옴
- cv.cvtColor()를 사용하여 그레이스케일로 변환
- cv.Sovel()을 사용하여 x축(cv.CV_64F, 1, 0)과 y축(cv.CV_64F, 0,1) 방향의 에지를 검출
- cv.magnitude()를 사용하여 에지 강도 계산
- Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화

### 힌트
- cv.Sobel()의 ksize는 3 또는 5로 설정
- cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 uint8로 변환
- plt.imshow()에서 cmap=‘gray’를 사용하여 흑백으로 시각화

<details>
<summary><h3><b>코드 - 1.py</b></h3></summary>
<div markdown="1">

```python
import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 연산을 위한 넘파이 라이브러리
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 이미지 불러오기 
img = cv.imread('edgeDetectionImage.jpg')  # 이미지 파일을 불러옴

# 이미지가 정상적으로 로드되었는지 확인하는 예외 처리
if img is None:  # 이미지 파일이 없을 경우
    print("이미지를 찾을 수 없습니다. 파일명을 확인하세요.")  # 에러 메시지 출력
else:
    # 2. 이미지를 그레이스케일로 변환 (에지 검출 전처리 단계)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 컬러 영상을 1채널 흑백 영상으로 변환

    # 3. Sobel 필터를 사용하여 x축 방향 에지 검출
    # cv.CV_64F: 미분값이 음수가 나올 수 있어 64비트 실수형으로 계산
    # 1, 0: x축 방향 미분 차수 설정
    # ksize=3: 커널 사이즈를 3x3으로 설정
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # x방향 소벨 연산 수행

    # 4. Sobel 필터를 사용하여 y축 방향 에지 검출
    # 0, 1: y축 방향 미분 차수 설정
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # y방향 소벨 연산 수행

    # 5. cv.magnitude()를 사용하여 에지 강도 계산
    # x방향과 y방향의 기울기 세기를 기하학적으로 합산 (루트 x^2 + y^2)
    edge_mag = cv.magnitude(sobel_x, sobel_y)

    # 6. cv.convertScaleAbs()를 사용하여 절댓값을 취하고 0~255 범위의 uint8 타입으로 변경
    edge_strength = cv.convertScaleAbs(edge_mag)  # 시각화 가능한 이미지 데이터로 변환

    # 7. Matplotlib을 사용하여 결과 시각화 설정
    plt.figure(figsize=(12, 6))  # 전체 출력 창의 크기를 가로 12, 세로 6으로 설정

    # --- 왼쪽: 원본 이미지 출력 영역 ---
    plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째 영역 선택
    # Matplotlib은 RGB 순서이므로 OpenCV의 BGR 이미지를 RGB로 변환하여 출력
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 원본 컬러 이미지 표시
    plt.title('Original Image')  # 그래프 제목 설정
    plt.axis('off')  # 눈금 및 좌표축 숨기기

    # --- 오른쪽: 에지 강도 이미지 출력 영역 ---
    plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째 영역 선택
    # cmap='gray'를 설정하여 흑백 에지 맵으로 시각화
    plt.imshow(edge_strength, cmap='gray')  # 최종 검출된 에지 이미지 표시
    plt.title('Edge Strength')  # 그래프 제목 설정
    plt.axis('off')  # 눈금 및 좌표축 숨기기

    # 8. 레이아웃 자동 조정 및 화면 출력
    plt.tight_layout()  # 이미지 간격 겹침 방지 및 자동 정렬
    plt.show()  # 완성된 결과 창 띄우기
```

</div>
</details>

### 실행 결과

![과제 1 결과](./figure 1.png)
<br><br>





## 과제 2 캐니 에지 및 허프 변환을 이용한 직선 검출
- dabo 이미지에 캐니 에지 검출을 사용하여 에지 맵 생성
- 허프 변환을 사용하여 이미지에서 직선 검출
- 검출된 직선을 원본 이미지에서 빨간색으로 표시

### 요구사항
- cv.Canny()를 사용하여 에지 맵 생성
- cv.HoughtLinesP()를 사용하여 직선 검출
- cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림
- Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화

### 힌트
- cv.Canny()에서 threshold1과 threshold2는 100과 200으로 설정
- cv.HoughLinesP()에서 rho, theta, threshold, minLineLength, maxLineGap 값을 조정하여 직선 검출 성능을 개선
- cv.line()에서 색상은 (0, 0, 255) (빨간색)과 두께는 2로 설정

<details>
<summary><h3><b>코드 - 2.py</b></h3></summary>
<div markdown="1">

```python
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

# 7. Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화
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
```
</div>
</details>


### 실행 결과

![과제 2 결과](./figure 2.png)
<br><br>





## 과제 3 GrabCut을 이용한 대화식 영역 분할 및 객체 추출
- coffee cup 이미지로 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체 추출
- 객체 추출 결과를 마스크 형태로 시각화
- 원본 이미지에서 배경을 제거하고 객체만 남은 이미지 출력

### 요구사항
- cv.grabCut()를 사용하여 대화식 분할을 수행
- 초기 사각형 영역은 (x, y, width, height) 형식으로 설정
- 마스크를 사용하여 원본 이미지에서 배경을 제거
- matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화

### 힌트
- cv.grabCut()에서 bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화
- 마스크 값은 cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD를 사용
- np.where()를 사용하여 마스크 값을 0 또는 1로 변경한 후 원본 이미지에 곱하여 배경을 제거

<details>
<summary><h3><b>코드 - 3.py</b></h3></summary>
<div markdown="1">

```python
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
```
</div>
</details>


### 실행 결과
![과제 3 결과](./figure 3.png)
<br><br>


