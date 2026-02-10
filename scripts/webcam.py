import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors  # ← 추가
import argparse
import time
from collections import defaultdict

def preprocess_frame(frame):
    """프레임 전처리"""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(frame, -1, kernel)
    
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    processed = cv2.merge([l, a, b])
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    
    return processed

def detect_pcb_region(frame):
    """기판 영역 자동 감지 (녹색 계열)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 녹색 범위 (기판 색상)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 윤곽선 = 기판
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 여유 공간 추가 (5% 확장)
        margin = 0.05
        x = max(0, int(x - w * margin))
        y = max(0, int(y - h * margin))
        w = int(w * (1 + 2 * margin))
        h = int(h * (1 + 2 * margin))
        
        return (x, y, w, h), mask
    
    return None, mask

def is_inside_pcb(box, pcb_region):
    """검출된 객체가 기판 내부에 있는지 확인"""
    if pcb_region is None:
        return True  # 기판 영역을 못 찾으면 모두 허용
    
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    pcb_x, pcb_y, pcb_w, pcb_h = pcb_region
    
    # 바운딩 박스의 중심점
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 중심점이 기판 영역 안에 있는지 확인
    return (pcb_x <= center_x <= pcb_x + pcb_w and 
            pcb_y <= center_y <= pcb_y + pcb_h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=int, default=0, help='웹캠 인덱스')
    parser.add_argument('--model', type=str, 
                       default='C:/Users/82106/OneDrive/문서/GitHub/SmartManufacturing_OpenCV/models/best.pt',
                       help='모델 경로')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU 임계값')
    parser.add_argument('--imgsz', type=int, default=640, help='입력 이미지 크기')
    parser.add_argument('--preprocess', action='store_true', help='전처리 활성화')
    parser.add_argument('--pcb-filter', action='store_true', help='기판 영역 필터링')
    parser.add_argument('--min-conf', type=float, default=0.5, help='최소 신뢰도 (추천: 0.5-0.7)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    args = parser.parse_args()
    
    print(f"🤖 YOLO 모델 로딩: {args.model}")
    model = YOLO(args.model)
    
    # Ultralytics 색상 팔레트 초기화
    colors = Colors()  # ← 추가
    
    print(f"🎥 웹캠 {args.source} 연결 중...")
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"❌ 웹캠을 열 수 없습니다!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ 웹캠 연결: {width}x{height}")
    print(f"⚙️  설정: conf={args.conf}, min_conf={args.min_conf}, iou={args.iou}")
    print(f"🔧 전처리: {'ON' if args.preprocess else 'OFF'}")
    print(f"🎯 기판 필터: {'ON' if args.pcb_filter else 'OFF'}")
    print(f"🐛 디버그: {'ON' if args.debug else 'OFF'}")
    print()
    print("📌 조작법:")
    print("   [ESC] - 종료")
    print("   [SPACE] - 스크린샷")
    print("   [P] - 전처리 ON/OFF")
    print("   [F] - 기판 필터 ON/OFF")
    print("   [D] - 디버그 모드 ON/OFF")
    print("   [+] - 최소 신뢰도 +0.05")
    print("   [-] - 최소 신뢰도 -0.05")
    print()
    
    # 통계
    frame_count = 0
    total_detections = 0
    filtered_detections = 0
    class_stats = defaultdict(lambda: {'count': 0, 'conf_sum': 0})
    fps_list = []
    
    preprocess_enabled = args.preprocess
    pcb_filter_enabled = args.pcb_filter
    debug_mode = args.debug
    min_confidence = args.min_conf
    screenshot_count = 0
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        original_frame = frame.copy()
        
        # 전처리
        if preprocess_enabled:
            frame = preprocess_frame(frame)
        
        # 기판 영역 감지
        pcb_region = None
        pcb_mask = None
        if pcb_filter_enabled:
            pcb_region, pcb_mask = detect_pcb_region(frame)
        
        # YOLO 검출
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False
        )
        
        # 수동으로 결과 그리기 (필터링 적용)
        annotated = frame.copy()
        
        n_det_original = len(results[0].boxes)
        n_det_filtered = 0
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Ultralytics 원본 색상 가져오기
            original_color = colors(cls, bgr=True)  # ← 추가
            
            # 필터링 조건
            passes_conf = conf >= min_confidence
            passes_pcb = is_inside_pcb(box, pcb_region) if pcb_filter_enabled else True
            
            if passes_conf and passes_pcb:
                n_det_filtered += 1
                
                # 바운딩 박스 그리기
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # 디버그 모드가 아니면 원본 색상 사용
                color = (0, 255, 0) if debug_mode else original_color  # ← 수정
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # 라벨
                label = f"{class_name} {conf:.2f}"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - h_text - 4), (x1 + w_text, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # ← 텍스트 색상 흰색으로 변경
                
                # 통계
                class_stats[class_name]['count'] += 1
                class_stats[class_name]['conf_sum'] += conf
            
            elif debug_mode:
                # 필터링된 객체는 빨간색으로 표시
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                color = (0, 0, 255)  # 빨간색 = 필터됨
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                
                # 필터링 이유 표시
                reason = []
                if not passes_conf:
                    reason.append(f"conf:{conf:.2f}<{min_confidence:.2f}")
                if not passes_pcb:
                    reason.append("outside_pcb")
                
                label = f"{class_name} [{','.join(reason)}]"
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 기판 영역 표시 (디버그 모드)
        if debug_mode and pcb_region is not None:
            x, y, w, h = pcb_region
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(annotated, "PCB Region", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        total_detections += n_det_filtered
        filtered_detections += (n_det_original - n_det_filtered)
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        
        # 통계 표시
        stats_height = 220 if debug_mode else 160
        cv2.rectangle(annotated, (10, 10), (450, stats_height), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (450, stats_height), (0, 255, 0), 2)
        
        y_offset = 35
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(annotated, f"Detected: {n_det_filtered} (filtered: {n_det_original - n_det_filtered})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(annotated, f"Min Conf: {min_confidence:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(annotated, f"Preprocess: {'ON' if preprocess_enabled else 'OFF'}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(annotated, f"PCB Filter: {'ON' if pcb_filter_enabled else 'OFF'}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if debug_mode:
            y_offset += 30
            cv2.putText(annotated, f"PCB Region: {'Found' if pcb_region else 'Not found'}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('YOLO Webcam Detection', annotated)
        
        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"📸 스크린샷: {filename}")
        elif key == ord('p') or key == ord('P'):
            preprocess_enabled = not preprocess_enabled
            print(f"🔧 전처리: {'ON' if preprocess_enabled else 'OFF'}")
        elif key == ord('f') or key == ord('F'):
            pcb_filter_enabled = not pcb_filter_enabled
            print(f"🎯 기판 필터: {'ON' if pcb_filter_enabled else 'OFF'}")
        elif key == ord('d') or key == ord('D'):
            debug_mode = not debug_mode
            print(f"🐛 디버그: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            min_confidence = min(0.95, min_confidence + 0.05)
            print(f"📈 최소 신뢰도: {min_confidence:.2f}")
        elif key == ord('-') or key == ord('_'):
            min_confidence = max(0.05, min_confidence - 0.05)
            print(f"📉 최소 신뢰도: {min_confidence:.2f}")
    
    # # 최종 통계
    # print("\n" + "="*70)
    # print("📊 최종 통계")
    # print("="*70)
    # print(f"총 프레임: {frame_count}")
    # print(f"평균 FPS: {sum(fps_list)/len(fps_list):.1f}")
    # print(f"총 검출: {total_detections}개")
    # print(f"필터링됨: {filtered_detections}개 ({filtered_detections/(total_detections+filtered_detections)*100:.1f}%)")
    # print(f"평균 검출: {total_detections/frame_count:.2f}개")
    # print()
    # print("📦 클래스별 통계:")
    # for cls_name, stats in sorted(class_stats.items()):
    #     avg_conf = stats['conf_sum'] / stats['count']
    #     print(f"  {cls_name}: {stats['count']}개 (평균 신뢰도: {avg_conf:.3f})")
    # print("="*70)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
