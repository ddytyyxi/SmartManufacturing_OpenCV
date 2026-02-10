import cv2
import time

print("카메라 연결 시도 중...")
cap = cv2.VideoCapture(1)
time.sleep(2)  # 카메라 초기화 대기

print(f"카메라 열림: {cap.isOpened()}")

if cap.isOpened():
    for i in range(10):  # 여러번 시도
        ret, frame = cap.read()
        if ret:
            print(f"프레임 성공! 크기: {frame.shape}")
            cv2.imshow('test', frame)