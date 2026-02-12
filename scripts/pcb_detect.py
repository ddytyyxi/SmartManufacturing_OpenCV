"""
=============================================================================
YOLO 기반 실시간 웹캠 PCB 결함 검출
=============================================================================

[전처리 구현방법]
  1단계: HSV 색상 필터로 기판 영역 판별
  2단계: CLAHE로 기판 영역 명암대비 향상
  3단계: 기판 영역만 크롭 → YOLO 검출 (conf 낮춰서 부품/결함 잘 잡기)

  ※ 기판 밖에서는 검출 안 함 → 오탐 차단
=============================================================================
"""

# 사용하는 라이브러리 정의
import cv2 #OpenCV ; 컴퓨터 비전 및 이미지/비디오 처리 라이브러리
import numpy as np # Numpy ; 수치계산 및 배열 얀신 라이브러리
from ultralytics import YOLO # 객체 탐지 모델을 위한 라이브러리
from ultralytics.utils.plotting import Colors # YOLO 시각화용 색상 
import argparse # 명령줄 인자를 파싱하기 위한 표준 라이브러리
import time # 시간 관련 라이브러리


# =============================================================================
# 1. HSV 기반 기판 영역 판별
# =============================================================================
def detect_pcb_by_hsv(frame):
    """
    HSV 색상 필터로 기판영역을 찾아서 바운딩 박스와 마스크를 반환한다.

    녹색 -> 기판 영역
    갈색 -> 납땝 같은 영역 ; 굳이 필요없을 것 같기도 함
    
    반환하는 인자 3가지
    1. 부품 바운딩 박스 → pcb_bbox: (x, y, w, h)
    2. 이진 마스크 → mask
    3. 기판 영역 비율 → mask_ratio
    """
    # 조건을 선언
    # 조건 : 반환된 값 X 이거나 빈 이미지 => 기본값을 반환
    if frame is None or frame.size == 0:
            # 기본값 
            # pcb_bbox / mask / mask_ratio
            # pcb_bbox : 바운딩 박스 X
            # mask : 0인 (100X100)픽셀, 각 픽셀 8bit로 저장
            # mask_ratio : 기판영역은 0.0% 
        return None, np.zeros((100, 100), dtype=np.uint8), 0.0
    
    # 1. BGR -> HSV로 변환 (H:색상, S:채도, V:명도)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. 기판이라고 판단되는 값 설정 → 색상을 지정해주는 것임
    # ── 녹색 기판 범위 ──
    mask_green = cv2.inRange(hsv, np.array([20, 15, 20]), np.array([100, 255, 255]))
    
    # ── 갈색 (동박/납땜) 범위 ── 
    mask_brown = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([30, 255, 200]))
    
    # 녹색 + 갈색 범위 합치기
    mask = cv2.bitwise_or(mask_green, mask_brown)
    
    if mask is None or mask.size == 0:
        return None, np.zeros(frame.shape[:2], dtype=np.uint8), 0.0
    
    # 3. 모폴로지로 정리 (작은 노이즈 제거 + 구멍 메우기)
    # CLOSE : 구멍 메우기 -> 기판 내부 끊겨 보이는거 완화
    # OPEN : 작은 점 노이즈 제거 -> 배경 잡색 제거 
    # *변경 팁 : 커널 크기 조정 (크게:거칠어짐) (작게:노이즈민감)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 구멍 메우기
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 노이즈 제거
    
    # 4. 기판의 영역이 화면에서 얼마나 차지하는지 비율계산
    # 기판 영역 비율
    mask_ratio = np.count_nonzero(mask) / mask.size * 100
    
    # 5. 외곽 컨투어를 찾아 가장 큰 덩어리를 "기판"이라고 가정
    # 컨투어 찾기 → 가장 큰 영역 = 기판 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        frame_area = frame.shape[0] * frame.shape[1]
        
        # 전체 프레임 대비 면적이 너무 작으면 기판으로 보기 어렵다고 판단하여 제외 
        # (예: 초록색 물체/노이즈)
        if area / frame_area > 0.1: # 차지하는 영역이 10% 이하이면 기판으로 인정X
            x, y, w, h = cv2.boundingRect(largest)
            
            # 마진 10% 추가 → 기판 가장자리 결함도 포함하도록
            margin = 0.10
            # 상하좌우 모두 10%씩 픽셀 확장
            x = max(0, int(x - w * margin))
            y = max(0, int(y - h * margin))
            w = min(frame.shape[1] - x, int(w * (1 + 2 * margin)))
            h = min(frame.shape[0] - y, int(h * (1 + 2 * margin)))
            
            # 기판O → 생성된 해당 값 반환
            return (x, y, w, h), mask, mask_ratio
        
    # 기판X → 기본 값 반환  
    return None, mask, mask_ratio


# =============================================================================
# 2. CLAHE 대비 향상``
# =============================================================================
def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    # frame : 입력 이미지 Numpy배열로 받음
    # clip_limt : CLAHE 대비 제한 값 → 높을수록 대비 ↑
    # grid_size : 이미지에 적용할 격자 크기
    """
    LAB 색공간의 L(밝기) 채널에만 CLAHE 적용.
    색상 왜곡 없이 대비만 향상

    BGR 로 받은 값 LAB 로 변환 할 때,
    L 값인 밝기만 조정하고 A,B인 색상 값은 그대로 보존해서 사용하도록 하는 것!
    """
    # 1. BGR -> LAB로 변환
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # 2. 채널 분리 (L값만 변환할 수 있도록)
    l, a, b = cv2.split(lab)
    
    # 3. CLAHE 객체 생성 (L값만 변환시키는 과정임)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size) # (타일단위 평활화, 과증폭 제한)
    l = clahe.apply(l)# 밝기 변환해서 명암 대비를 주는 거임
    
    # 4. 다시 LAB로 합치고 → BGR로 복원해서 반환
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# =============================================================================
# 메인
# =============================================================================
def main():
    # ArgumentParaser 객체 생성 
    parser = argparse.ArgumentParser(description='YOLO PCB 결함 검출 v2')
    # ── 생성 인자 정의 ──
    # 1. 웹캠 선택
    parser.add_argument('--source', type=int, default=0, help='웹캠 인덱스')
    # 2. YOLO 모델 파일 경로
    parser.add_argument('--model', type=str,
                        default='C:/Users/82106/OneDrive/문서/GitHub/SmartManufacturing_OpenCV/models/best.pt',
                        help='모델 경로')
    # 3. 신뢰도 임계값
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값 (기판 내부용, 낮게 설정)')
    # 4. NMS IOU 임계값 → 겹치는 바운딩박스 제거 기준 **
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU 임계값') # 전체 임계값은 높게 설정
    # 5. 입력 이미지 크기 (YOLO 처리 해상도)    
    parser.add_argument('--imgsz', type=int, default=1280, help='입력 이미지 크기') # *해상도 수정 640 → 1280
    # 6. 디버그 모드 → 기판영역 시각화
    parser.add_argument('--debug', action='store_true', help='기판 영역 시각화')
    args = parser.parse_args() # 모든 인자 파싱 후 args 객체에 저장

    # ── 모델 로드 ──
    print(f"모델 로딩: {args.model}")
    model = YOLO(args.model)
    colors = Colors()

    # ── 웹캠 연결 ──
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("※ 웹캠 실행 불가능 ※")
        return

    #웹캠 해상도 설정 → HD로 설정 (1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

    #연결된 웹캠 해상도 출력 @디버깅용
    print(f"웹캠 연결 완료: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")


    # 프로그램 시작 후 사용 가능한 키
    print()
    print("=" * 50)
    print("  [ESC]    종료")
    print("  [+/-]    신뢰도 조절")
    print("  [D]      디버그(기판 영역 표시) ON/OFF")
    print("  [SPACE]  스크린샷")
    print("=" * 50)
    print()

    # ── 상태 변수 ──
    min_conf = args.conf  # YOLO 신뢰도 임계값 (결함 검출 기준, 기본 0.25)
    debug_mode = args.debug  # 디버그 모드 ON/OFF (기판 영역 시각화 여부)
    prev_time = time.time()  # 이전 프레임 시간 (FPS 계산용)
    fps_list = []  # FPS 값 저장 리스트 (평균 계산용)
    screenshot_count = 0  # 저장된 스크린샷 개수 (파일명 번호)

    # ── 메인 루프 ──
    while True:

        # 웹캠에서 한 프레임 읽기
        ret, frame = cap.read() # ret : 프레임 읽기 성공여부 / frame : 읽어온 이미지
        if not ret: # 실패 시 종료
            break #  # 원본 보존용 복사본 (여기에 결과 그려짐)

        annotated = frame.copy() # 원본 보존용 복사본 (여기에 결과 그려짐)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # [1단계] HSV로 기판 영역 판별
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #기판 영역 검출 (바운딩 박스, 마스크, 영역 비율 반환)
        pcb_bbox, mask, mask_ratio = detect_pcb_by_hsv(frame)
        
        pcb_found = pcb_bbox is not None #기판 발견 여부 
        n_det = 0 # 검출된 부품 개수 

        if pcb_found: # 기판이 검출된 경우에만 결함 검사 진행
            px, py, pw, ph = pcb_bbox # 기판 영역 좌표 및 크기 언패킹
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # [2단계] 기판 영역 크롭 + CLAHE 대비 향상
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            pcb_crop = frame[py:py+ph, px:px+pw] # 기판 영역만 잘라내기
            pcb_crop = apply_clahe(pcb_crop) # 명압 대비 향상으로 결함 판별력 ↑

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # [3단계] 크롭된 기판 영역에서만 YOLO 검출
            #         conf를 낮춰서 부품/결함 민감하게 잡기
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            results = model.predict(
                source=pcb_crop, # 잘라낸 기판영역만 검사
                conf=min_conf, # 기판영역 내 신뢰도 임계값
                iou=args.iou, # 기판영역 내 NMS 임계값
                imgsz=args.imgsz, # 입력 이미지 크기 
                verbose=False # 로그 출력 X
            )
            
            # 검출 결과를 원본 프레임 좌표로 변환해서 그리기
            for box in results[0].boxes: # 검출된 객체에 대한 반복문
                conf = float(box.conf[0]) # 신뢰도 → 0~1
                cls = int(box.cls[0]) # 부품 ID → 클래스 인덱스 
                class_name = model.names[cls] # 부품(클래스) 명칭
                color = colors(cls, bgr=True) # 부품별 고유 색상 부여 
                
                # 잘라낸 좌표 → 원본 좌표로 오프셋 추가
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy()) # # 바운딩 박스 좌표
                x1 += px
                y1 += py
                x2 += px
                y2 += py
                
                # 부품별 바운딩박스 그리기 
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # 부품별 라벨링 하기 
                label = f"{class_name} {conf:.2f}"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - h_text - 4), (x1 + w_text, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                n_det += 1
            
            # 디버그모드: 기판 영역 표시
            if debug_mode:
                cv2.rectangle(annotated, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
                cv2.putText(annotated, f"PCB ({mask_ratio:.0f}%)", (px, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── FPS 계산 ──
        curr_time = time.time()  # 현재 시간
        fps = 1 / max(curr_time - prev_time, 0.001)  # FPS = 1 / 프레임 간 시간 (0으로 나누기 방지)
        prev_time = curr_time  # 다음 프레임 계산용 시간 업데이트
        fps_list.append(fps)  # FPS 리스트에 추가
        if len(fps_list) > 30:  # 최근 30개만 유지
            fps_list.pop(0)  # 가장 오래된 값 제거
        avg_fps = sum(fps_list) / len(fps_list)  # 평균 FPS 계산

        # ── HUD ── 정보표시 박스
        hud_h = 155 if debug_mode else 130
        cv2.rectangle(annotated, (10, 10), (350, hud_h), (0, 0, 0), -1) # 검정배경
        cv2.rectangle(annotated, (10, 10), (350, hud_h), (0, 255, 0), 1) #초록테두리
        
        #정보표시 박스 텍스트 시작 y좌표 값
        y_pos = 35

        # fps 표시
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
        
       # 기판 상태 표시
        pcb_color = (0, 255, 0) if pcb_found else (0, 0, 255)  # 발견: 초록, 미발견: 빨강
        pcb_text = f"PCB: Found ({mask_ratio:.0f}%)" if pcb_found else "PCB: Not Found"
        cv2.putText(annotated, pcb_text, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pcb_color, 2)
        y_pos += 25
        
        # 검출된 결함/부품 개수 표시
        cv2.putText(annotated, f"Detected: {n_det}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
        
        # *디버그 모드 정보 표시
        if debug_mode:
            y_pos += 25
            cv2.putText(annotated, f"Debug ON | PCB area: {mask_ratio:.1f}%",
                        (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 화면에 표시
        cv2.imshow('PCB Detection v2', annotated)

        # ── 키보드 입력 처리 ──
        key = cv2.waitKey(1) & 0xFF  # 1ms 대기하며 키 입력 받기 (& 0xFF: 8비트만 사용)
        if key == 27:  # ESC 키
            break  # 프로그램 종료
        elif key == ord('+') or key == ord('='):  # + 또는 = 키
            min_conf = min(0.95, min_conf + 0.05)  # 신뢰도 5% 증가 (최대 0.95)
            print(f"신뢰도: {min_conf:.2f}")
        elif key == ord('-') or key == ord('_'):  # - 또는 _ 키
            min_conf = max(0.05, min_conf - 0.05)  # 신뢰도 5% 감소 (최소 0.05)
            print(f"신뢰도: {min_conf:.2f}")
        elif key == ord('d') or key == ord('D'):  # D 키
            debug_mode = not debug_mode  # 디버그 모드 토글
            print(f"디버그: {'ON' if debug_mode else 'OFF'}")
        elif key == ord(' '):  # 스페이스바
            screenshot_count += 1  # 스크린샷 번호 증가
            filename = f"screenshot_{screenshot_count:03d}.jpg"  # 파일명 (예: screenshot_001.jpg)
            cv2.imwrite(filename, annotated)  # 현재 화면 저장
            print(f"스크린샷 저장: {filename}")

    # 프로그램 종료 시 리소스 해제
    cap.release()  # 웹캠 해제
    cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기