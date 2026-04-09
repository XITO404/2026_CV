import cv2 # 비디오 및 이미지 처리를 위한 OpenCV
import numpy as np # 수치 연산을 위한 넘파이
import os # 파일 경로 확인용
from sort import SORT # 직접 분리해 만든 SORT 추적 알고리즘 클래스 임포트

# COCO 데이터셋 기본 클래스 이름 리스트
COCO_NAMES = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light"]

# ─────────────────────────────────────────────────────────
# 1. YOLOv3 객체 검출기
# ─────────────────────────────────────────────────────────
class YOLOv3Detector:
    def __init__(self, cfg="yolov3.cfg", weights="yolov3.weights", names="coco.names", conf_thresh=0.4, nms_thresh=0.4):
        # 파일이 존재하면 읽어서 리스트로 만들고, 없으면 기본 COCO_NAMES 사용
        self.classes = [l.strip() for l in open(names).readlines()] if os.path.exists(names) else COCO_NAMES
        # 다크넷(Darknet) 기반의 YOLOv3 설정 및 가중치 파일을 OpenCV DNN 네트워크로 로드
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # 백엔드로 OpenCV 선택
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)      # CPU 환경에서 동작하도록 설정
        
        layer_names = self.net.getLayerNames() # 네트워크의 모든 레이어 이름 가져오기
        out_idx = self.net.getUnconnectedOutLayers() # 최종 출력 레이어의 인덱스 확인
        # OpenCV 버전에 따른 차이 처리 후 출력 레이어 이름 추출
        self.output_layers = [layer_names[i[0] - 1] if isinstance(out_idx[0], (list, np.ndarray)) else layer_names[i - 1] for i in out_idx]
        # 객체 신뢰도 임계값, NMS 임계값, YOLOv3 입력 권장 사이즈(416x416) 설정
        self.conf_thresh, self.nms_thresh, self.input_size = conf_thresh, nms_thresh, (416, 416)

    def detect(self, frame):
        H, W = frame.shape[:2] # 원본 프레임의 높이와 너비
        # 이미지를 YOLO 모델이 받아들일 수 있는 정규화된 4차원 블롭(Blob) 형태로 변환
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob) # 네트워크 입력 설정
        layer_outputs = self.net.forward(self.output_layers) # 순전파(Forward) 연산 실행하여 결과 도출

        boxes, scores, class_ids = [], [], []
        for output in layer_outputs:
            for det in output:
                confs = det[5:] # 처음 5개(x,y,w,h,물체확률) 이후의 값들이 클래스별 확률
                cid = int(np.argmax(confs)) # 가장 확률이 높은 클래스의 인덱스 추출
                conf = float(confs[cid])    # 그 클래스의 확률값 추출
                # 확률값이 임계값(0.4)보다 높은 유효한 검출만 사용
                if conf > self.conf_thresh:
                    # 0~1 사이의 비율로 나온 박스 크기를 원본 이미지 크기 픽셀값으로 복원
                    cx, cy, w, h = det[0]*W, det[1]*H, det[2]*W, det[3]*H
                    # 중심 좌표를 좌상단 좌표로 변환하여 저장
                    boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                    scores.append(conf)
                    class_ids.append(cid)

        # NMS(Non-Maximum Suppression): 같은 객체에 여러 박스가 겹쳐 쳐진 경우 가장 확률이 높은 박스 하나만 남김
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.nms_thresh)
        if len(indices) == 0: return [], [], [] # 남은 박스가 없으면 빈 리스트 반환
        # OpenCV 버전 호환성을 위해 리스트 평탄화
        indices = [i[0] for i in indices] if isinstance(indices[0], (list, np.ndarray)) else list(indices.flatten())
        
        # NMS를 통과한 최종 박스, 신뢰도, 클래스 ID만 추려서 반환
        return [boxes[i] for i in indices], [scores[i] for i in indices], [class_ids[i] for i in indices]

# ─────────────────────────────────────────────────────────
# 2. 시각화 함수
# ─────────────────────────────────────────────────────────
def draw_results(frame, results, show_traj):
    # 추적기에서 반환된 각 객체 정보(박스, ID, 클래스명, 궤적)를 순회하며 화면에 그리기
    for bbox, tid, cls_name, history in results:
        x, y, w, h = (int(v) for v in bbox) # 박스 좌표를 정수형으로 변환
        np.random.seed(tid * 7 + 13) # ID 기반으로 일정한 색상이 나오도록 랜덤 시드 고정
        color = tuple(int(c) for c in np.random.randint(80, 255, 3)) # 고유한 BGR 색상 생성

        # 프레임에 객체의 경계 상자 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # ID 번호와 클래스 이름 텍스트 생성 및 박스 위쪽에 표시
        label = f"ID:{tid} {cls_name}"
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 궤적 표시 옵션이 켜져 있고 이전 기록이 있을 경우
        if show_traj and len(history) > 1:
            # 궤적 리스트에 있는 과거 중심점들을 선으로 연결하여 이동 경로를 표시
            for k in range(1, len(history)):
                p1 = (int(history[k-1][0]+history[k-1][2]/2), int(history[k-1][1]+history[k-1][3]/2))
                p2 = (int(history[k][0]+history[k][2]/2), int(history[k][1]+history[k][3]/2))
                cv2.line(frame, p1, p2, color, 2)

# ─────────────────────────────────────────────────────────
# 3. 메인 실행 루프
# ─────────────────────────────────────────────────────────
def main():
    # 검출기와 추적기 객체 초기화 (설정값 적용)
    detector = YOLOv3Detector(cfg="yolov3.cfg", weights="yolov3.weights")
    tracker = SORT(max_age=3, min_hits=2, iou_threshold=0.3)
    
    # 분석할 비디오 파일 로드
    cap = cv2.VideoCapture("slow_traffic_small.mp4")
    show_traj = True # 궤적 그리기 모드 기본 ON

    while cap.isOpened():
        ret, frame = cap.read() # 프레임 한 장씩 읽기
        if not ret: break       # 영상이 끝나면 루프 종료

        # 1. YOLOv3를 통해 현재 프레임의 모든 객체 검출
        boxes, _, class_ids = detector.detect(frame)
        # 숫자로 된 클래스 ID를 사람이 읽을 수 있는 이름('car', 'person' 등)으로 변환
        class_names = [detector.classes[c] if 0 <= c < len(detector.classes) else "obj" for c in class_ids]

        # 2. 검출된 정보를 SORT 알고리즘에 넘겨 ID 부여 및 위치 추적 업데이트
        results = tracker.update(boxes, class_ids, class_names)

        # 3. 업데이트된 추적 결과를 영상 프레임 위에 시각화
        draw_results(frame, results, show_traj)
        # 좌상단 타이틀 텍스트 추가
        cv2.putText(frame, "SORT Tracker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 처리된 프레임을 화면에 출력
        cv2.imshow("Multi-Object Tracking", frame)

        # 키보드 입력 대기 및 제어 (q: 종료, s: 궤적 표시 토글)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'): show_traj = not show_traj

    # 메모리 해제 및 모든 창 닫기
    cap.release()
    cv2.destroyAllWindows()

# 프로그램 시작점 지정
if __name__ == "__main__":
    main()