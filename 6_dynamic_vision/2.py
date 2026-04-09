"""
Mediapipe FaceMesh 얼굴 랜드마크 추출 및 시각화
필요 패키지: pip install mediapipe opencv-python

실행: python face_landmark.py
종료: ESC 키
"""

import cv2
import mediapipe as mp

# ─────────────────────────────────────────
# 1. Mediapipe FaceMesh 초기화
# ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,          # 최대 검출 얼굴 수
    refine_landmarks=True,    # 눈동자·입술 정밀 랜드마크 포함
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────
# 2. 웹캠 열기
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] 웹캠을 열 수 없습니다.")
    exit()

print("얼굴 랜드마크 검출 시작 | ESC: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # ─────────────────────────────────────
    # 3. BGR → RGB 변환 후 FaceMesh 추론
    # ─────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False          # 성능 최적화
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    # ─────────────────────────────────────
    # 4. 랜드마크 시각화
    # ─────────────────────────────────────
    if results.multi_face_landmarks:
        for face_lm in results.multi_face_landmarks:

            # ── 468개 랜드마크를 점으로 표시 ──
            for lm in face_lm.landmark:
                # 정규화 좌표 → 픽셀 좌표 변환
                x = int(lm.x * W)
                y = int(lm.y * H)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # ── Mediapipe 기본 연결선 (테셀레이션) ──
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_lm,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles
                    .get_default_face_mesh_tesselation_style()
            )

            # ── 눈 윤곽선 ──
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_lm,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 100, 100), thickness=1)
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_lm,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(100, 100, 255), thickness=1)
            )

            # ── 입술 윤곽선 ──
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_lm,
                connections=mp_face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 200, 255), thickness=1)
            )

    # ─────────────────────────────────────
    # 5. 정보 표시
    # ─────────────────────────────────────
    n_faces = len(results.multi_face_landmarks) \
              if results.multi_face_landmarks else 0
    cv2.putText(frame, f"Faces: {n_faces}  Landmarks: {n_faces * 468}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "ESC: Quit",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imshow("FaceMesh - 468 Landmarks", frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ─────────────────────────────────────────
# 6. 자원 해제
# ─────────────────────────────────────────
cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("종료.")
