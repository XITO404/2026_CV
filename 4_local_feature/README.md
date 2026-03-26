## 과제 1 SIFT를 이용한 특징점 검출 및 시각화
- 주어진 이미지(mot_color70.jpg)를 이용하여 SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 특징점을
검출하고 이를 시각화

### 요구사항
- cv.SIFT_create()를 사용하여 SIFT 객체를 생성
- detectAndCompute()를 사용하여 특징점을 검출
- cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화
- matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력

### 힌트
- SIFT_create()의 매개변수를 변경하며 특징점 검출 결과를 비교
- 특징점이 너무 많다면 nfeatures 값을 조정하여 제한
- cv.drawKeypoints()의 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 설정하면 특징점의
방향과 크기도 표시

<details>
<summary><h3><b>코드 - 1.py</b></h3></summary>
<div markdown="1">

```python
```

</div>
</details>

### 핵심 코드
**(1) ㅇ**
```python

```
- ㅇ
- 


**(2) ㅇ**
```python
```
- ㅇ
- 


**(3) ㅇ**
```python
```
- ㅇ
- 


### 실행 결과

![과제 1 결과](./)
<br><br>



---
## 과제 2 SIFT를 이용한 두 영상 간 특징점 매칭
- 두 개의 이미지(mot_color70.jpg, mot_color80.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화
 
### 요구사항
- cv.imread()를 사용하여 두 개의 이미지를 불러옴
- cv.SIFT_create()를 사용하여 특징점을 추출
- cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭
- cv.drawMatches()를 사용하여 매칭 결과를 시각화
- matplotlib을 이용하여 매칭 결과를 출력

### 힌트
- BFMatcher(cv.NORM_L2, crossCheck=True)를 사용하면 간단한 매칭이 가능
- FLANN 기반 매칭을 원하면 cv.FlannBasedMatcher()를 사용
- knnMatch()와 DMatch 객체를 활용하여 최근접 이웃 거리 비율을 적용하면 매칭 정확도를 높일 수 있음

<details>
<summary><h3><b>코드 - 2.py</b></h3></summary>
<div markdown="1">

```python
```

</div>
</details>

### 핵심 코드
**(1) ㅇ**
```python

```
- ㅇ
- 


**(2) ㅇ**
```python
```
- ㅇ
- 


**(3) ㅇ**
```python
```
- ㅇ
- 


### 실행 결과

![과제 2 결과](./)
<br><br>




---
## 과제 3 호모그래피를 이용한 이미지 정합 (Image Alignment)
- SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬
- 샘플파일로 img1.jpg, imag2.jpg, imag3.jpg 중 2개를 선택

### 요구사항
- cv.imread()를 사용하여 두 개의 이미지를 불러옴
- cv.SIFT_create()를 사용하여 특징점을 검출
- cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭하고, 좋은 매칭점만 선별
- cv.findHomography()를 사용하여 호모그래피 행렬을 계산
- cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
- 변환된 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력


### 힌트
- cv.findHomography()에서 cv.RANSAC을 사용하면 이상점(Outlier) 영향을 줄일 수 있음
- cv.warpPerspective()를 사용할 때 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정
- knnMatch()로 두 개의 최근접 이웃을 구한 뒤, 거리 비율이 임계값(예: 0.7) 미만인 매칭점만 선별


<details>
<summary><h3><b>코드 - 3.py</b></h3></summary>
<div markdown="1">

```python
```

</div>
</details>

### 핵심 코드
**(1) ㅇ**
```python

```
- ㅇ
- 


**(2) ㅇ**
```python
```
- ㅇ
- 


**(3) ㅇ**
```python
```
- ㅇ
- 


### 실행 결과

![과제 3 결과](./)
<br><br>

