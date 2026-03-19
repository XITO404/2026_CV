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

### 핵심 코드
**(1) grayscale 이미지 변환**
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```
- cv.cvtColor(): 이미지 색상 공간을 변환
- cv.COLOR_BGR2GRAY: BGR 컬러 이미지를 Sobel 필터 적용에 필요한 1채널 grayscale로 변환


**(2) Sobel 필터를 이용한 x, y축 에지 검출**
```python
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
```
- cv.Sobel(): 이미지의 미분값을 계산하여 x축 방향(1, 0)과 y축 방향(0, 1)의 에지 검출
- cv.CV_64F: 미분 연산 중 발생하는 음수 값을 보존하기 위해 64비트 실수형으로 계산
- ksize=3: 3x3 크기의 소벨 커널


**(3) 에지 강도(Magnitude) 계산 및 시각화를 위한 uint8 변환**
```python
edge_mag = cv.magnitude(sobel_x, sobel_y)
edge_strength = cv.convertScaleAbs(edge_mag)
```
- cv.magnitude(): x축과 y축의 기울기 결과값을 기하학적으로 합산($\sqrt{x^2 + y^2}$), 전체 에지 강도 계산
- cv.convertScaleAbs(): 실수형 데이터를 절댓값 변환 후, 시각화가 가능한 0~255 범위의 8비트(uint8) 타입으로 변경


### 실행 결과

![과제 1 결과](./Figure%201.png)
<br><br>




---
## 과제 2 캐니 에지 및 허프 변환을 이용한 직선 검출
- dabo 이미지에 캐니 에지 검출을 사용하여 에지 맵 생성
- 허프 변환을 사용하여 직선 검출 후 원본 이미지에 빨간색으로 표시

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


### 핵심 코드
**(1) 캐니 에지 검출**
```python
edges = cv.Canny(gray, 100, 200)
```
- cv.Canny(): 이미지에서 강한 에지만을 남기는 에지 맵 생성
- 임계값으로 threshold1=100, threshold2=200 설정: 하한값과 상한값 사이의 픽셀들을 분석, 약한 에지 중 강한 에지와 연결된 에지 검출

**(2) 허프 변환을 이용한 직선 검출**
```python
lines = cv.HoughLinesP(edges, 1, np.pi/180, 120, minLineLength=40, maxLineGap=10)
```
- cv.HoughLinesP(): 에지 맵에서 실제 직선 성분 추출
- rho=1: 거리 해상도 (1픽셀 단위)
- theta=np.pi/180: 각도 해상도 (라디안, 약 1도 단위)
- 120: 임계값 (투표 시 직선으로 판단하기 위한 최소 교차 횟수)
- minLineLength: 검출할 직선의 최소 길이
- maxLineGap: 동일 선상의 점들을 하나의 직선으로 잇기 위한 최대 간격

**(3) 검출된 직선 그리기 및 RGB 변환**
```python
cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_lines_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)
```
- cv.line(): 검출된 좌표 (x1, y1)에서 (x2, y2)까지 두께 2의 빨간색(0, 0, 255) 직선을 그림
- OpenCV의 BGR 형식을 Matplotlib 시각화 환경에 맞추기 위해 RGB 형식으로 변환


### 실행 결과

![과제 2 결과](./Figure%202.png)
<br><br>




---
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

### 핵심 코드
**(1) GrabCut 초기화 및 사각형 영역(ROI) 설정**
```python
mask = np.zeros(img.shape[:2], np.uint8)    # GrabCut을 위한 초기 마스크 생성
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, img.shape[1]-100, img.shape[0]-100)
```
- mask를 이미지 크기와 동일하게 0으로 초기화
- 배경 모델 bgdModel, 전경 모델 fgdModel을 0으로 초기화
- rect 변수에 추출하고자 하는 객체(커피컵)가 포함된 사각형 영역을 (x, y, w, h) 형식으로 설정
- 좌표값에 img.shape 사용; 입력 이미지 해상도 변화에 유연하게 대응하기 위해 동적 ROI 설정<br>
: 상하좌우 50픽셀 정도를 여백으로 남기고, 중앙의 넓은 영역을 자동으로 ROI(관심 영역)로 설정하기 위해 이미지의 가로(shape[1])와 세로(shape[0]) 길이를 활용해 계산

**(2) cv.grabCut()을 이용한 대화식 영역 분할**
```python
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
```
- cv.grabCut(): 전경(fgdModel)과 배경(bgdModel)을 분하는 반복 연산 5회 수행
- cv.GC_INIT_WITH_RECT: 사각형 영역 ROI를 기준으로 전경 객체 추정

**(3) 마스크 값 처리 및 배경 제거**
```python
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
img_grabcut = img * mask2[:, :, np.newaxis]
```
- np.where(): 확실한 배경(0)과 배경일 가능성이 높은 영역(2)를 0으로, 전경 영역(1, 3)을 1로 이진화
- 이진 마스크 합성 및 배경 제거<br>
: 이진화된 mask2를 원본 이미지에 곱하여 전경 객체를 추출<br>
: 이때 배경은 마스크 값 0이 곱해지므로 검은색으로 처리됨
- 채널 확장 및 행렬 연산<br>
: 1채널인 mask2를 원본과 동일한 3채널로 확장(np.newaxis)하여 행렬 연산 수행<br>
: 최종적으로 배경이 제거된 3컬러 객체 이미지를 생성

  
### 실행 결과
![과제 3 결과](./Figure%203.png)
<br><br>

