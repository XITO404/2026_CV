import cv2 as cv
import numpy as np
import sys

# 초기 설정값 정의
brush_size = 5          # 초기 붓 크기 5
Lbutton_down = False # 왼쪽 마우스 버튼 상태 체크용 변수
Rbutton_down = False # 오른쪽 마우스 버튼 상태 체크용 변수

# 마우스 콜백 함수: 이미지 위에서 마우스 동작 처리
def draw(event, x, y, flags, param):
    global brush_size, Lbutton_down, Rbutton_down

    # 왼쪽 버튼 클릭 시 파란색으로 붓질
    if event == cv.EVENT_LBUTTONDOWN:
        Lbutton_down = True # 좌클릭
        cv.circle(img, (x, y), brush_size, (255, 0, 0), -1) # 파란색(BGR)

    # 오른쪽 버튼 클릭 시 빨간색으로 붓질
    elif event == cv.EVENT_RBUTTONDOWN:
        Rbutton_down = True # 우클릭
        cv.circle(img, (x, y), brush_size, (0, 0, 255), -1) # 빨간색(BGR)

    # 마우스 드래그 시 연속해서 그리기
    elif event == cv.EVENT_MOUSEMOVE:
        if Lbutton_down:
            cv.circle(img, (x, y), brush_size, (255, 0, 0), -1)
        elif Rbutton_down:
            cv.circle(img, (x, y), brush_size, (0, 0, 255), -1)

    # 마우스 버튼을 떼면 그리기 상태 해제
    elif event == cv.EVENT_LBUTTONUP:
        Lbutton_down = False
    elif event == cv.EVENT_RBUTTONUP:
        Rbutton_down = False

# soccer.jpg 이미지 로드
img = cv.imread('./soccer.jpg')

# 파일 로드 실패 시 예외 처리
if img is None:
    sys.exit('이미지 파일을 찾을 수 없습니다.')

# 윈도우 창 생성 및 마우스 이벤트 연결
cv.namedWindow('Painting App')
cv.setMouseCallback('Painting App', draw)

# 메인 루프: 실시간으로 화면을 갱신하고 키 입력을 처리
while True:
    cv.imshow('Painting App', img)
    
    # 1ms 대기하며 키 입력 감지 (0이면 무한 대기이므로 반드시 1 이상 사용)
    key = cv.waitKey(1) & 0xFF

    # 'q'를 누르면 프로그램 종료
    if key == ord('q'):
        break
    
    # '+' 키 입력 시 붓 크기 1 증가
    elif key == ord('+'):
        brush_size = min(15, brush_size + 1)    # 최댓값 15
        print(f"현재 붓 크기: {brush_size}")

    # '-' 키 입력 시 붓 크기 1 감소
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)     # 최솟값 1
        print(f"현재 붓 크기: {brush_size}")    # 터미널에 현재 붓 크기 출력

# 모든 창을 닫고 프로그램 종료
cv.destroyAllWindows()