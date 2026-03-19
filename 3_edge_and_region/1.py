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