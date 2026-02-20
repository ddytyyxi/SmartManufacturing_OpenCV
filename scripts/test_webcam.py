"""
간단한 웹캠 테스트 스크립트
ESC 키를 누르면 종료됩니다.
"""
import cv2

print("=" * 50)
print("웹캠 테스트 시작")
print("ESC 키를 누르면 종료됩니다")
print("=" * 50)

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 에러: 웹캠을 열 수 없습니다!")
    exit()

print("✅ 웹캠 연결 성공!")
print(f"해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ 프레임을 읽을 수 없습니다")
        break

    # 화면에 텍스트 표시
    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 표시
    cv2.imshow('Webcam Test', frame)

    # ESC 키 확인
    if cv2.waitKey(1) & 0xFF == 27:
        print("프로그램 종료")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 정상 종료되었습니다")
