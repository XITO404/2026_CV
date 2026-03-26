import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 연산을 위한 넘파이 라이브러리
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 이미지 불러오기
img = cv.imread('mot_color70.jpg')  # 이미지를 BGR 형태로 읽어옴

# 이미지가 정상적으로 로드되었는지 확인
if img is None: # 이미지 로드 실패 시 None 반환
    print("이미지를 찾을 수 없습니다. 파일명을 확인하세요.")    # 에러 메시지 출력
else:
    # 2. SIFT는 그레이스케일 이미지에서 특징을 추출하므로 변환 수행
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 컬러 이미지를 그레이스케일로 변환

    # 3. cv.SIFT_create()를 사용하여 SIFT 객체 생성
    # 힌트 반영: nfeatures를 800으로 설정하여 주요 특징점 위주로 제한 (조정 가능)
    sift = cv.SIFT_create(nfeatures=800) # 500으로 설정 후 더 촘촘한 특징점 추출 위해 800으로 조정

    # 4. detectAndCompute()를 사용하여 특징점(Keypoints)과 기술자(Descriptors) 검출
    # 기술자(des)는 이번 과제 시각화에는 쓰이지 않지만 함수 반환값이라 함께 받음
    kp, des = sift.detectAndCompute(gray, None)

    # 5. cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화
    # 힌트 반영: flags를 설정하여 특징점의 크기와 방향(Angle)까지 시각화
    img_sift = cv.drawKeypoints(img, kp, None, 
                                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 6. Matplotlib을 사용하여 원본 이미지와 특징점 이미지를 나란히 출력
    plt.figure(figsize=(15, 7))  # 전체 창 크기 설정

    # --- 왼쪽: 원본 이미지 ---
    plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째
    # OpenCV(BGR)를 Matplotlib(RGB) 형식으로 변환하여 출력
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')  # 축 눈금 숨기기

    # --- 오른쪽: SIFT 특징점이 표시된 이미지 ---
    plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째
    plt.imshow(cv.cvtColor(img_sift, cv.COLOR_BGR2RGB))
    plt.title(f'SIFT Features (n={len(kp)})')
    # nfeatures는 엄격한 제한값이 아닌 상위 특징점 추출을 위한 '목표치'
    # 알고리즘이 특징점의 점수(Response)를 매겨 상위 순으로 추출할 때, 
    # 동일 점수를 가진 점들이 포함되거나 옥타브별 계산 과정에서 설정값보다 1~2개 더 검출될 수 있음
    plt.axis('off')

    # 7. 레이아웃 조정 및 화면 출력
    plt.tight_layout()  # 이미지 간격 겹침 방지 및 자동 정렬
    plt.show()      # 완성된 결과 창 띄우기