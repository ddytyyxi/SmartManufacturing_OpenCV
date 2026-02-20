# 📓 Notebooks
이 폴더에는 PCB 부품 검출을 위한 전체 딥러닝 파이프라인이 단계별로 구성되어 있습니다.
모델은 YOLO 기반 객체 탐지를 사용하여 학습 및 추론을 수행합니다.

```
```
## 📁 Notebook 설명

### 1. [`data_preparation.ipynb]`
데이터 전처리 및 학습을 위한 데이터셋 구조를 구성합니다.
- 이미지 및 라벨 정리
- YOLO 학습용 폴더 구조 생성
- `data.yaml` 생성

---

### 2. `model_training.ipynb`
YOLO 모델 학습을 수행합니다.
- 학습 파라미터 설정
- 모델 학습 실행
- Loss 및 mAP 확인

---

### 3. `model_evaluation.ipynb`
학습된 모델 성능을 평가합니다.
- Validation dataset 평가
- Precision / Recall / mAP 분석
- 결과 시각화

---

### 4. `inference_demo.ipynb`
학습된 모델을 사용하여 PCB 이미지에서 객체 검출을 수행합니다.
- `best.pt` 모델 로드
- 테스트 이미지 입력
- Bounding Box 시각화

---

## 💻 실행 환경
- Python 3.9+
- GPU 권장 (Google Colab)

필요 라이브러리:
```bash
ultralytics
opencv-python
matplotlib
```

## 참고
- 학습된 모델 파일은 `models/` 폴더 참고
- 데이터셋 구조는 `data/README.md` 참고
