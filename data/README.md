# 📊 Data

## PCB Component Detection Dataset 데이터셋 정보

https://www.kaggle.com/datasets/animeshkumarnayak/pcb-fault-detection 데이터셋 사용

Kaggle의 PCB Fault Detection 데이터셋은 PCB 이미지와 각 부품 위치에 대한 바운딩 박스 라벨이 포함된 객체 탐지용 데이터셋으로, 약 9개의 전자부품 클래스를 기반으로 구성되어 있다. 이 데이터는 YOLO와 같은 딥러닝 모델을 활용한 PCB 부품 검출 및 결함 분석 연구와 실습에 활용된다.

### 디렉토리 구조

```
data/
├── test/
│   └── images/              # YOLO 학습 모델 검증용 이미지 파일
├── train/
│   ├── images/              # 학습용 이미지
│   └── labels/              # 학습용 라벨 (YOLO 형식)
├── validation/
│   ├── images/              # 검증용 이미지
│   └── labels/              # 검증용 라벨 (YOLO 형식)
└── data.yaml                # 데이터셋 설정 파일
```


