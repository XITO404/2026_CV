import cv2 # 칼만 필터 기능 사용을 위한 OpenCV 임포트
import numpy as np # 행렬 및 수치 연산을 위한 넘파이 임포트
from scipy.optimize import linear_sum_assignment # 헝가리안 매칭 알고리즘 함수 임포트

# ─────────────────────────────────────────────────────────
# 1. IoU 계산 (Intersection over Union: 두 박스가 겹치는 비율)
# ─────────────────────────────────────────────────────────
def compute_iou_matrix(trackers, detections):
    # [x, y, w, h] 형식을 좌표 중심의 [x1, y1, x2, y2]로 변환하는 내부 함수
    def to_xyxy(b):
        return [b[0], b[1], b[0]+b[2], b[1]+b[3]]

    # 트랙(기존 객체) 개수 행, 검출(새 객체) 개수 열을 가진 빈 IoU 행렬 생성
    iou = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for i, t in enumerate(trackers):
        t = to_xyxy(t) # 트랙 박스 좌표 변환
        for j, d in enumerate(detections):
            d = to_xyxy(d) # 검출 박스 좌표 변환
            # 두 박스가 겹치는 교집합 영역의 좌상단, 우하단 좌표 계산
            xi1, yi1 = max(t[0], d[0]), max(t[1], d[1])
            xi2, yi2 = min(t[2], d[2]), min(t[3], d[3])
            # 교집합 면적 계산 (겹치지 않으면 0)
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            # 합집합 면적 = (트랙 면적) + (검출 면적) - (교집합 면적)
            union = ((t[2]-t[0])*(t[3]-t[1]) + (d[2]-d[0])*(d[3]-d[1]) - inter)
            # IoU 비율 계산 및 행렬에 저장
            iou[i, j] = inter / union if union > 0 else 0
    return iou

# ─────────────────────────────────────────────────────────
# 2. 칼만 필터 트랙 (개별 객체의 상태 관리)
# ─────────────────────────────────────────────────────────
class KalmanBoxTracker:
    count = 0 # 모든 객체에 부여될 고유 ID 카운터 (전역 변수)

    def __init__(self, bbox):
        # 8개의 상태(x,y,w,h, vx,vy,vw,vh)와 4개의 관측값(x,y,w,h)을 가지는 칼만 필터 생성
        self.kf = cv2.KalmanFilter(8, 4)
        # 상태 전이 행렬 (등속도 운동 모델: 다음 위치 = 현재 위치 + 속도)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0, 1,0,0,0], [0,1,0,0, 0,1,0,0], [0,0,1,0, 0,0,1,0], [0,0,0,1, 0,0,0,1],
            [0,0,0,0, 1,0,0,0], [0,0,0,0, 0,1,0,0], [0,0,0,0, 0,0,1,0], [0,0,0,0, 0,0,0,1]
        ], dtype=np.float32)
        # 관측 행렬 (실제 측정된 4개의 위치 및 크기 정보만 반영)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        # 시스템(예측) 노이즈 공분산 설정
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        # 측정(검출기) 노이즈 공분산 설정
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        # 초기 오차 공분산 설정 (처음엔 불확실성이 크므로 큰 값 부여)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10
        # 검출된 초기 박스 위치로 칼만 필터 상태 초기화
        self.kf.statePost = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)

        KalmanBoxTracker.count += 1      # 새 객체 생성 시 전체 ID 카운터 1 증가
        self.id = KalmanBoxTracker.count # 현재 객체에 고유 ID 부여
        # 상태 추적용 변수들 (히트 수, 연속 히트 수, 업데이트 안된 시간, 나이) 초기화
        self.hits, self.hit_streak, self.time_since_update, self.age = 1, 1, 0, 0
        self.history = []                # 객체의 이동 궤적을 저장할 리스트
        self.class_id, self.class_name = -1, "unknown" # 클래스 정보 초기화

    def predict(self):
        # 칼만 필터 수학 모델을 통해 현재 프레임의 위치 예측
        pred = self.kf.predict()
        self.age += 1                # 객체 나이 증가
        self.time_since_update += 1  # 업데이트 안된 시간 1 증가 (이후 매칭되면 0으로 리셋됨)
        bbox = pred[:4].flatten()    # 예측된 [x, y, w, h] 값 추출
        self.history.append(bbox.copy()) # 궤적 리스트에 현재 위치 저장
        # 궤적이 너무 길어지지 않도록 최근 40프레임 위치만 보존
        if len(self.history) > 40: self.history.pop(0)
        return bbox

    def update(self, bbox, class_id=-1, class_name="unknown"):
        # 실제 검출(YOLO) 결과로 칼만 필터 상태를 정밀하게 보정
        self.time_since_update = 0   # 매칭 성공했으므로 업데이트 시간 0으로 리셋
        self.hits += 1               # 총 히트 수 증가
        self.hit_streak += 1         # 연속 히트 수 증가
        self.class_id, self.class_name = class_id, class_name # 클래스 정보 갱신
        self.kf.correct(np.array(bbox, dtype=np.float32).reshape(4, 1)) # 실제 측정값으로 보정 수행

    def get_state(self):
        # 현재 추정된 가장 정확한 객체의 [x, y, w, h] 상태 반환
        return self.kf.statePost[:4].flatten()

# ─────────────────────────────────────────────────────────
# 3. SORT 알고리즘 본체 (다중 객체 관리 및 데이터 연관)
# ─────────────────────────────────────────────────────────
class SORT:
    def __init__(self, max_age=3, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age             # 객체가 검출되지 않아도 ID를 유지할 최대 프레임 수
        self.min_hits = min_hits           # 추적 결과를 화면에 표시하기 위한 최소 연속 검출 횟수
        self.iou_threshold = iou_threshold # 겹침 비율이 이 값 이상일 때만 동일 객체로 매칭 허용
        self.trackers = []                 # 현재 추적 중인 모든 칼만 필터 객체들을 담을 리스트
        self.frame_count = 0               # 전체 프레임 카운터

    def update(self, detections, class_ids=None, class_names=None):
        self.frame_count += 1
        # 클래스 정보가 없을 경우 기본값으로 채움
        if class_ids is None: class_ids = [-1] * len(detections)
        if class_names is None: class_names = ["obj"] * len(detections)

        # 1. 모든 기존 트랙에 대해 다음 위치 예측
        predicted, del_idx = [], []
        for i, tr in enumerate(self.trackers):
            p = tr.predict() # 칼만 필터 예측 실행
            # 예측값이 NaN(에러)인 경우 삭제 리스트에 추가
            if np.any(np.isnan(p)): del_idx.append(i)
            else: predicted.append(p)
        # 역순으로 에러 난 트랙 삭제 (인덱스 밀림 방지)
        for i in reversed(del_idx): self.trackers.pop(i)

        # 2. 헝가리안 매칭 알고리즘을 통해 기존 궤적과 새 검출값을 최적 매칭
        matched, unmatched_dets, unmatched_trks = self._associate(detections, predicted)

        # 3. 성공적으로 짝이 지어진 트랙들의 상태를 실제 검출값으로 보정(업데이트)
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(detections[d_idx], class_ids[d_idx], class_names[d_idx])

        # 4. 기존 트랙과 매칭되지 않은 새로운 검출값은 새로운 칼만 필터 트랙으로 생성
        for d_idx in unmatched_dets:
            tr = KalmanBoxTracker(detections[d_idx])
            tr.class_id, tr.class_name = class_ids[d_idx], class_names[d_idx]
            self.trackers.append(tr)

        # 5. 결과 반환 및 오랫동안 안 보이는 소멸 트랙 처리
        results, alive = [], []
        for tr in self.trackers:
            # 설정한 max_age 프레임 이내로 업데이트 된 트랙만 생존 처리
            if tr.time_since_update <= self.max_age:
                alive.append(tr)
                # 노이즈를 걸러내기 위해 최소 min_hits 이상 연속 검출된 확정 트랙만 결과에 포함
                if tr.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    results.append((tr.get_state(), tr.id, tr.class_name, tr.history))
        self.trackers = alive # 살아남은 트랙들로 리스트 갱신
        return results

    def _associate(self, detections, predictions):
        # 예측값이나 검출값이 아예 없으면 매칭을 건너뜀
        if not predictions: return [], list(range(len(detections))), []
        if not detections: return [], [], list(range(len(predictions)))

        # IoU 행렬 계산 (크기가 클수록 유사함)
        iou_mat = compute_iou_matrix(predictions, detections)
        # 헝가리안 알고리즘 수행 (비용 행렬을 요구하므로 1 - IoU를 입력하여 거리가 짧은 것을 찾음)
        row_ind, col_ind = linear_sum_assignment(1 - iou_mat)

        matched, used_d, used_t = [], set(), set()
        for r, c in zip(row_ind, col_ind):
            # 매칭되었더라도 IoU 값이 임계치(0.3) 이상이어야 최종 동일 객체로 승인
            if iou_mat[r, c] >= self.iou_threshold:
                matched.append((r, c))
                used_t.add(r) # 사용된 트랙 인덱스 기록
                used_d.add(c) # 사용된 검출 인덱스 기록

        # 매칭되지 못하고 남은 검출(새로운 객체)과 트랙(사라진 객체) 분류
        unmatched_dets = [i for i in range(len(detections)) if i not in used_d]
        unmatched_trks = [i for i in range(len(predictions)) if i not in used_t]
        return matched, unmatched_dets, unmatched_trks