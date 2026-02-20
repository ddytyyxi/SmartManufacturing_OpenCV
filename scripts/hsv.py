"""
HSV 색상 범위 튜닝 도구
비디오에서 기판 색상의 최적 HSV 값을 찾기 위한 도구
"""

import cv2
import numpy as np
import argparse

def nothing(x):
    pass

def main():
    parser = argparse.ArgumentParser(description='HSV 튜닝 도구')
    parser.add_argument('--video', type=str, required=True, help='비디오 파일 경로')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {args.video}")
        return
    
    # 트랙바 윈도우 생성
    cv2.namedWindow('HSV Tuner')
    cv2.namedWindow('Original')
    cv2.namedWindow('Mask')
    
    # 초기값 설정 (녹색 기판 기준)
    cv2.createTrackbar('H_min', 'HSV Tuner', 35, 179, nothing)
    cv2.createTrackbar('H_max', 'HSV Tuner', 85, 179, nothing)
    cv2.createTrackbar('S_min', 'HSV Tuner', 40, 255, nothing)
    cv2.createTrackbar('S_max', 'HSV Tuner', 255, 255, nothing)
    cv2.createTrackbar('V_min', 'HSV Tuner', 40, 255, nothing)
    cv2.createTrackbar('V_max', 'HSV Tuner', 255, 255, nothing)
    
    # 갈색 범위 추가
    cv2.createTrackbar('Brown ON', 'HSV Tuner', 0, 1, nothing)
    cv2.createTrackbar('BH_min', 'HSV Tuner', 10, 179, nothing)
    cv2.createTrackbar('BH_max', 'HSV Tuner', 25, 179, nothing)
    cv2.createTrackbar('BS_min', 'HSV Tuner', 50, 255, nothing)
    cv2.createTrackbar('BV_min', 'HSV Tuner', 50, 255, nothing)
    
    print("\n" + "="*60)
    print("HSV 튜닝 도구")
    print("="*60)
    print("트랙바를 조정하여 기판 영역이 하얗게 표시되도록 설정하세요")
    print("기판 외 영역은 검정색이 되어야 합니다")
    print("")
    print("[SPACE] - 다음 프레임")
    print("[B] - 이전 프레임으로")
    print("[R] - 비디오 처음으로")
    print("[P] - 현재 설정값 출력")
    print("[ESC] - 종료")
    print("="*60 + "\n")
    
    frame_pos = 0
    
    while True:
        # 프레임 읽기
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        
        if not ret:
            frame_pos = 0
            continue
        
        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 트랙바 값 읽기
        h_min = cv2.getTrackbarPos('H_min', 'HSV Tuner')
        h_max = cv2.getTrackbarPos('H_max', 'HSV Tuner')
        s_min = cv2.getTrackbarPos('S_min', 'HSV Tuner')
        s_max = cv2.getTrackbarPos('S_max', 'HSV Tuner')
        v_min = cv2.getTrackbarPos('V_min', 'HSV Tuner')
        v_max = cv2.getTrackbarPos('V_max', 'HSV Tuner')
        
        brown_on = cv2.getTrackbarPos('Brown ON', 'HSV Tuner')
        bh_min = cv2.getTrackbarPos('BH_min', 'HSV Tuner')
        bh_max = cv2.getTrackbarPos('BH_max', 'HSV Tuner')
        bs_min = cv2.getTrackbarPos('BS_min', 'HSV Tuner')
        bv_min = cv2.getTrackbarPos('BV_min', 'HSV Tuner')
        
        # 녹색 마스크
        mask_green = cv2.inRange(hsv, 
                                 np.array([h_min, s_min, v_min]), 
                                 np.array([h_max, s_max, v_max]))
        
        # 갈색 마스크
        if brown_on:
            mask_brown = cv2.inRange(hsv,
                                    np.array([bh_min, bs_min, bv_min]),
                                    np.array([bh_max, 255, 200]))
            mask = cv2.bitwise_or(mask_green, mask_brown)
        else:
            mask = mask_green
        
        # 모폴로지 적용
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 마스크 비율 계산
        mask_ratio = np.count_nonzero(mask) / mask.size * 100
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 원본에 정보 표시
        info_frame = frame.copy()
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(info_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 정보 표시
        cv2.putText(info_frame, f"Frame: {frame_pos}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_frame, f"Mask Ratio: {mask_ratio:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 화면 표시
        cv2.imshow('Original', info_frame)
        cv2.imshow('Mask', mask)
        
        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # 다음 프레임
            frame_pos = min(frame_pos + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        elif key == ord('b') or key == ord('B'):  # 이전 프레임
            frame_pos = max(frame_pos - 1, 0)
        elif key == ord('r') or key == ord('R'):  # 처음으로
            frame_pos = 0
        elif key == ord('p') or key == ord('P'):  # 설정값 출력
            print("\n" + "="*60)
            print("현재 HSV 설정값:")
            print("="*60)
            print(f"녹색 범위:")
            print(f"  H: [{h_min}, {h_max}]")
            print(f"  S: [{s_min}, {s_max}]")
            print(f"  V: [{v_min}, {v_max}]")
            if brown_on:
                print(f"\n갈색 범위:")
                print(f"  H: [{bh_min}, {bh_max}]")
                print(f"  S: [{bs_min}, 255]")
                print(f"  V: [{bv_min}, 200]")
            print(f"\n코드에 적용:")
            print(f"mask_green = cv2.inRange(hsv, np.array([{h_min}, {s_min}, {v_min}]), np.array([{h_max}, {s_max}, {v_max}]))")
            if brown_on:
                print(f"mask_brown = cv2.inRange(hsv, np.array([{bh_min}, {bs_min}, {bv_min}]), np.array([{bh_max}, 255, 200]))")
            print("="*60 + "\n")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
