import cv2 as cv
import numpy as np
import sys
from tkinter import filedialog
import tkinter as tk

# 전역 변수 초기화
is_dragging = False      # 마우스 드래그 상태 확인
start_x, start_y = -1, -1 # 사각형 시작 좌표
roi = None               # 잘라낸 관심 영역 저장 변수

# 한글 경로 문제를 해결하기 위한 이미지 읽기 함수
def imread_korean(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded_img = np.frombuffer(data, dtype=np.uint8)
    return cv.imdecode(encoded_img, cv.IMREAD_COLOR)

# 마우스 콜백 함수: 드래그로 사각형 그리기 및 ROI 지정
def on_mouse(event, x, y, flags, param):
    global is_dragging, start_x, start_y, img_display, roi

    # 왼쪽 버튼 클릭 시: 시작점 저장 및 드래그 시작
    if event == cv.EVENT_LBUTTONDOWN:
        is_dragging = True
        start_x, start_y = x, y

    # 마우스 이동 시: 드래그 중이면 실시간으로 사각형 그리기
    elif event == cv.EVENT_MOUSEMOVE:
        if is_dragging:
            img_display = img.copy() # 원본 복사하여 잔상 제거
            # cv.rectangle() 함수로 영역 시각화
            cv.rectangle(img_display, (start_x, start_y), (x, y), (0, 255, 0), 2)

    # 왼쪽 버튼을 떼었을 때: 드래그 종료 및 ROI 추출
    elif event == cv.EVENT_LBUTTONUP:
        is_dragging = False
        x1, y1 = min(start_x, x), min(start_y, y)
        x2, y2 = max(start_x, x), max(start_y, y)

        if x2 - x1 > 0 and y2 - y1 > 0:
            # ROI 추출은 numpy 슬라이싱 사용
            roi = img[y1:y2, x1:x2]
            cv.imshow('Extracted ROI', roi) # 별도의 창에 출력

# 파일 선택 창 실행
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="이미지 선택", filetypes=[("Images", "*.jpg *.png")])

if not file_path:
    sys.exit('파일이 선택되지 않았습니다.')

# 한글 경로 대응 함수로 이미지 로드
img = imread_korean(file_path)

if img is None:
    sys.exit('이미지를 불러올 수 없습니다. 경로를 확인하세요.')

img_display = img.copy()

cv.namedWindow('Select ROI')
cv.setMouseCallback('Select ROI', on_mouse) # 마우스 이벤트 처리 연결

print("r: 리셋, s: 저장, q: 종료")

while True:
    cv.imshow('Select ROI', img_display) # 화면 출력
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    # 요구사항: r 키를 누르면 영역 선택 리셋
    elif key == ord('r'):
        img_display = img.copy()
        if cv.getWindowProperty('Extracted ROI', 0) >= 0:
            cv.destroyWindow('Extracted ROI')
        roi = None
        print("리셋되었습니다.")

    # 요구사항: s 키를 누르면 선택한 영역을 저장
    elif key == ord('s'):
        if roi is not None:
            res, encoded_img = cv.imencode('.jpg', roi)
            if res:
                with open('./extracted_roi.jpg', 'wb') as f:
                    f.write(encoded_img)
                print("저장 완료")
        else:
            print("선택된 영역이 없습니다.")

cv.destroyAllWindows()