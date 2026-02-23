# 🔧 SmartManufacturing-OpenCV

스마트 제조 환경에서 생산되는 PCB 기판을 구성하는 요소를 식별하는 프로젝트 \n
YOLO 객체 탐지 모델을 활용하여 제조 초기 단계에서 PCB 부품 누락 및 결함을 자동으로 검출할 수 있도록 도와줍니다.

<br/>

## 📌 프로젝트 개요

본 프로젝트는 인쇄회로기판(PCB)의 제조 품질 관리를 자동화하기 위해 딥러닝 기반 객체 탐지 기술을 적용

### 주요 목표
- PCB 기판 상의 전자 부품 자동 인식
- 부품 누락 및 배치 오류 검출
- 제조 공정 초기 단계의 불량 예방

### 핵심 기능
- 다양한 PCB 부품 탐지
- 실시간 품질 검사 가능
- 높은 정확도의 객체 인식

<br/>

## 🗂️ 프로젝트 구조

```
SmartManufacturing-OpenCV/
├── data/                    # 데이터셋 폴더
│   ├── train/              # 학습 데이터
│   ├── validation/         # 검증 데이터
│   └── test/               # 테스트 데이터
├── models/                  # 학습된 모델 파일
│   ├── best.pt             # 최고 성능 모델
│   └── last.pt             # 마지막 에포크 모델
├── notebook/                # Colab 노트북
│   ├── train.ipynb         # 모델 학습
│   └── predict.ipynb       # 예측 및 평가
├── scripts/                 # 실행 스크립트
├── results/                 # 학습 결과 및 로그
└── requirements.txt         # 필요 패키지
```

<br/>

## 🧪 사용 데이터

- **데이터 출처:** [Dataset Ninja - PCB Component Detection](https://datasetninja.com/pcb-component-detection)
- **데이터셋 크기:** 약 2,000장 이미지
- **어노테이션 형식:** YOLO 바운딩 박스

### 주요 클래스
names: ['Cap1', 'Cap2', 'Cap3', 'Cap4', 'MOSFET', 'Mov', 'Resistor', 'Transformer']

<br/>

## 🧑‍💻 기술 스택

### Deep Learning & Computer Vision
- **YOLOv8**: 객체 탐지 모델
- **Ultralytics**: YOLO 프레임워크

### 개발 환경 및 도구
- **Python 3.10+**
- **Google Colab**: GPU 기반 학습 환경
- **Libraries**: PyTorch, OpenCV, NumPy, Pandas, Matplotlib

### 버전 관리
- **Git / GitHub**: 코드 버전 관리 및 협업

<br/>

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/SmartManufacturing-OpenCV.git
cd SmartManufacturing-OpenCV

# 필요 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터셋 준비

```bash
# data/ 폴더에 데이터셋 배치
# 구조는 data/README.md 참고
```

### 3. 모델 학습

Google Colab에서 `notebook/train.ipynb` 실행

### 4. 예측 수행

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('models/best.pt')

# 예측
results = model.predict('data/test/images/')
results[0].show()
```

<br/>

## 📊 모델 성능

| Metric | Value |
|--------|-------|
| mAP50 | 97.91% |
| mAP50-95 | 77.99% |
| Precision | 97.69% |
| Recall | 99.41% |


<br/>

## 💡 주요 인사이트

- YOLO 모델을 통해 PCB 부품 탐지의 자동화 가능성 확인
- 실시간 검사 시스템 구축을 위한 기반 기술 확보
- 제조 공정에서의 품질 관리 효율성 향상 기대
