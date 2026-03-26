import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 연산 라이브러리
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 두 개의 이미지 불러오기
img1 = cv.imread('mot_color70.jpg')  # 첫 번째 이미지 (기준)
img2 = cv.imread('mot_color83.jpg')  # 두 번째 이미지 (대상)

# 이미지가 정상적으로 로드되었는지 확인
if img1 is None or img2 is None:
    print("이미지를 찾을 수 없습니다. 파일명을 확인하세요.")
else:
    # 2. SIFT 객체 생성 (기본 파라미터 사용)
    sift = cv.SIFT_create()

    # 3. 각 이미지에서 특징점(kp)과 기술자(des)를 동시에 검출
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 4. BFMatcher 객체 생성 (SIFT는 L2 Norm을 사용함)
    # L2 Norm = 유클리드 거리 계산 방식, 제곱의 합의 루트
    # crossCheck=False로 설정하여 knnMatch를 사용 가능하게 함
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    # 5. knnMatch를 사용하여 각 특징점당 가장 유사한 2개의 매칭점 찾기
    matches = bf.knnMatch(des1, des2, k=2)

    # 6. 힌트: 최근접 이웃 거리 비율(Lowe's Ratio Test)을 적용, 매칭 정확도 향상
    # 첫 번째 매칭점이 두 번째 매칭점보다 훨씬 가까운(유사한) 경우만 선택
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 임계값 0.7 적용
            good_matches.append(m)

    # 7. cv.drawMatches()를 사용하여 매칭 결과를 하나의 이미지로 시각화
    # flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS: 매칭되지 않은 점은 숨김
    img_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 8. Matplotlib을 사용하여 매칭 결과 출력
    plt.figure(figsize=(20, 10))
    # BGR을 RGB로 변환하여 출력
    plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
    plt.title(f'SIFT Matching Results (Good Matches: {len(good_matches)})')
    plt.axis('off')
    plt.show()