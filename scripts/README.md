# 📜 Scripts
실행 스크립트 폴더

## ⚠️ 실행 전 유의 사항

실행을 위한 패키지 설치 필수!

```bash
pip install -r requirements.txt
```


## ⚙️ PCB 결함 검출 파이프라인

```mermaid
flowchart TD
    A([📷 웹캠 입력 · 1280×720])

    A --> B["🔍 STEP 1 · 기판 영역 찾기 · detect_pcb_by_hsv()<br/>🎨 BGR → HSV 변환 후 녹색·갈색 마스크 생성 &nbsp;&nbsp; 🧹 모폴로지 연산으로 노이즈 제거 + 구멍 메우기 &nbsp;&nbsp; 📦 가장 큰 컨투어를 기판으로 판단 · 면적 20% 이상"]

    B --> C{기판 발견?}
    C -- "✅ 발견" --> D
    C -- "❌ 없음" --> E([🔄 다음 프레임으로])

    D["✨ STEP 2 · 이미지 전처리 · apply_clahe()<br/>🔄 BGR → LAB 변환 후 L채널만 분리 &nbsp;&nbsp; 💡 CLAHE 명암 대비 향상 · clipLimit=1.2 / grid=8×8 &nbsp;&nbsp; 🎯 색상 보존하면서 어두운 부품도 잘 보이게"]

    D --> F["🤖 STEP 3 · YOLO 부품 검출 · model.predict()
    <br/>✂️ 기판 영역만 크롭해서 입력 → 오탐 방지
     ⚙️ conf=0.30 / iou=0.45 / imgsz=640"]

    F --> G["🔢 STEP 4 · 부품 개수 검증 · validate_component_counts()<br/>📋 Cap1~4 · MOSFET · Mov · Resistor 각 1개 / Transformer 2개 · 총 9개 기대 &nbsp;&nbsp; ❓ 실제 검출 수와 비교 → missing / extra 분류"]

    G --> H["🕐 STEP 5 · 시간적 안정화 · TemporalSmoother.update()<br/>🪟 최근 10프레임 검출 이력 기록 &nbsp;&nbsp; 📊 60% 이상 검출 시 확정 → 깜빡임 노이즈 제거"]

    H --> I{결과}
    I -- "모두 정상" --> J([✅ ALL PARTS OK])
    I -- "미검출 있음" --> K([⚠️ MISSING 경고 · ex. MISSING: Cap1 x1])

    style A fill:#2d2d2d,color:#fff,stroke:#2d2d2d
    style B fill:#e8faf0,stroke:#1a7a45,color:#1a7a45
    style C fill:#fff,stroke:#aaa,color:#444
    style D fill:#e8f3ff,stroke:#1a4a9a,color:#1a4a9a
    style E fill:#fff0f0,stroke:#b01020,color:#b01020
    style F fill:#fff6e8,stroke:#a05a00,color:#a05a00
    style G fill:#f5eeff,stroke:#6020b0,color:#6020b0



```
## 참고

- 모델 학습 및 예측 과정은 `notebook/` 폴더의 Colab 노트북 파일 참고
    
```
