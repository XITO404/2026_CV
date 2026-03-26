import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 연산 라이브러리
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 두 개의 이미지 불러오기
img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')

# 이미지가 정상적으로 로드되었는지 확인
if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
else:
    # 2. SIFT 특징점 및 기술자 추출
    sift = cv.SIFT_create()
    # 각 이미지에서 특징점(kp)과 기술자(des) 동시 추출
    # kp: 특징점의 위치 정보, des: 특징점 주변의 패턴 정보를 담은 128차원 벡터
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. BFMatcher(Brute-Force Matcher) 객체 생성
    # cv.NORM_L2: SIFT 기술자 간의 유사도 측정 위해 L2 Norm(유클리디안 거리 방식) 사용
    bf = cv.BFMatcher(cv.NORM_L2)
    # knnMatch를 이용한 특징점 매칭: 각 특징점당 가장 유사한 2개의 매칭점 찾기 (k=2)
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. Lowe's Ratio Test를 통해 좋은 매칭점(Good Matches) 선별
    good_matches = []
    for m, n in matches:
        # 첫 번째 매칭 거리(m)가 두 번째 매칭 거리(n)보다 훨씬 가까운 경우만 선택
        if m.distance < 0.7 * n.distance:   # 0.7은 힌트의 예시 반영 (경험적으로 자주 사용되는 값)
            good_matches.append(m)

    # 5. 호모그래피(Homography) 계산을 위한 좌표점 추출
    # 최소 4개 이상의 대응점 필요
    if len(good_matches) > 4:
        # 매칭된 특징점들의 좌표를 추출하여 호모그래피 함수 형식에 맞게 변환 (float32 타입 배열)
        # m.queryIdx: img1의 특징점 인덱스 / m.trainIdx: img2의 특징점 인덱스
        # reshape(-1, 1, 2): 호모그래피 계산 함수가 요구하는 넘파이 배열 차원 형식으로 변환 (N, 1, 2)  
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # cv.findHomography(): 두 이미지 평면 사이의 3x3 투영 변환 행렬 M을 계산
        # cv.RANSAC: 잘못 매칭된 점들을 무시하고 가장 많은 점이 동의하는 변환 관계를 찾는 알고리즘
        # 5.0: RANSAC 알고리즘에서 허용하는 최대 거리 오차(임계값, 픽셀 단위)
        # dst_pts, src_pts 순서: 훈련 이미지(img2)의 좌표를 기준 이미지(img1) 좌표계로 매핑
        M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

        # 6. cv.warpPerspective(): 계산된 행렬 M을 사용하여 비틀어서 변환
        # 출력 크기는 두 이미지를 합친 넉넉한 크기로 설정 (w1+w2, h)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        res_width = w1 + w2
        res_height = max(h1, h2)
        
        # cv.warpPerspective: 투영 변환 행렬 M을 적용하여 최종 결과 이미지 생성
        # img2를 img1의 평면에 맞게 변환하여 새로운 이미지 생성 (res_width, res_height 크기)
        img2_warped = cv.warpPerspective(img2, M, (res_width, res_height))

        # img1을 왼쪽에 그대로 배치
        img2_warped[0:h1, 0:w1] = img1

        # 7. 매칭 결과 시각화 이미지 생성 (특징점끼리 선으로 연결)
        # flags=NOT_DRAW_SINGLE_POINTS: 매칭되지 않은 고립된 특징점은 그리지 않음
        img_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # 8. Matplotlib을 이용한 최종 결과 시각화 출력
        plt.figure(figsize=(20, 10))

        # 위쪽: 특징점 매칭 결과, img1과 img2의 특징점 매칭 선 시각화
        plt.subplot(2, 1, 1)    # 2행 1열 중 첫 번째 위치
        plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))    # BGR을 RGB로 변환
        plt.title('Feature Matching Result')
        plt.axis('off') # 좌표축 숨김

        # 아래쪽: 호모그래피 변환 결과 (Warped Image)
        plt.subplot(2, 1, 2)    # 2행 1열 중 두 번째 위치
        plt.imshow(cv.cvtColor(img2_warped, cv.COLOR_BGR2RGB))
        plt.title('Warped Image (img2 aligned to img1)')
        plt.axis('off')

        # 레이아웃 간격 조정 및 최종 출력
        plt.tight_layout()
        plt.show()
    else:
        # 매칭점이 4개 미만인 경우 호모그래피 계산 불가 메시지 출력
        print("매칭점이 부족하여 호모그래피를 계산할 수 없습니다.")