"""
웹캠 실시간 부품 검출 데모
"""
import cv2
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics가 설치되지 않았습니다!")
    print("설치: pip install ultralytics")
    input("Enter 키를 눌러 종료...")
    sys.exit(1)


def run_webcam(model_path='models/best.pt', conf=0.25, camera_id=0):
    """
    웹캠 실시간 검출 실행
    
    Args:
        model_path: 모델 파일 경로
        conf: 신뢰도 임계값 (0.0 ~ 1.0)
        camera_id: 카메라 ID (기본: 0)
    """
    
    # 모델 로드
    print(f"🤖 모델 로딩: {model_path}")
    try:
        model = YOLO(model_path)
        print("✅ 모델 로드 완료!\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print(f"💡 models/ 폴더에 best.pt가 있는지 확인하세요!")
        input("Enter 키를 눌러 종료...")
        sys.exit(1)
    
    # 웹캠 열기
    print(f"🎥 웹캠 시작 (카메라 ID: {camera_id})")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다!")
        print("💡 다른 카메라 ID를 시도하려면: python scripts/webcam.py --camera 1")
        input("Enter 키를 눌러 종료...")
        sys.exit(1)
    
    # 웹캠 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*50)
    print("📹 웹캠 실시간 검출 시작!")
    print("="*50)
    print("💡 종료: 'q' 키 또는 'ESC' 키")
    print("💡 일시정지: 'p' 키")
    print("="*50 + "\n")
    
    paused = False
    frame_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 프레임을 읽을 수 없습니다!")
                    break
                
                # YOLO 예측
                results = model(frame, conf=conf, verbose=False)
                
                # 결과 그리기
                annotated_frame = results[0].plot()
                
                # 검출 정보 표시
                boxes = results[0].boxes
                
                # 상단에 정보 표시
                info_text = f'Objects: {len(boxes)} | Frame: {frame_count} | Conf: {conf}'
                cv2.putText(
                    annotated_frame, 
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # 하단에 조작법 표시
                cv2.putText(
                    annotated_frame,
                    "Press 'Q' or 'ESC' to quit | 'P' to pause",
                    (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                frame_count += 1
            
            # 화면 출력
            cv2.imshow('Parts Detection - Webcam Demo', annotated_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                print("\n✅ 종료합니다...")
                break
            elif key == ord('p'):  # 'p' - 일시정지
                paused = not paused
                status = "일시정지" if paused else "재생 중"
                print(f"⏸️  {status}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자가 중단했습니다 (Ctrl+C)")
    
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
    
    finally:
        # 정리
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 통계:")
        print(f"   총 프레임: {frame_count}")
        print("\n✅ 프로그램 종료")
        input("Enter 키를 눌러 창을 닫으세요...")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='웹캠 실시간 부품 검출',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scripts/webcam.py
  python scripts/webcam.py --conf 0.3
  python scripts/webcam.py --camera 1
  python scripts/webcam.py --model models/last.pt
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/best.pt',
        help='모델 파일 경로 (기본: models/best.pt)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='신뢰도 임계값 (기본: 0.25)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='카메라 ID (기본: 0)'
    )
    
    args = parser.parse_args()
    
    # 실행
    run_webcam(
        model_path=args.model,
        conf=args.conf,
        camera_id=args.camera
    )


if __name__ == '__main__':
    main()