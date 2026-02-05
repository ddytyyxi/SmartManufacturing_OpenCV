# 🔧SmartManufacturing-OpenCV
스마트 제조 환경에서 기계의 센서 데이터를 기반으로 이상 징후를 분석하고,  
정비가 필요한 조건을 도출하는 프로젝트입니다.

<br/>

## 📌 프로젝트 개요
유지보수가 필요한 경우에 이상 징후 패턴 분석하고, 예측 정비 기준 조건을 수립하는 것을 목표로 합니다.


<br/>

## 🧪 사용 데이터

- **데이터 출처:** https://www.kaggle.com/datasets/animeshkumarnayak/pcb-fault-detection
- **주요 변수**
  - `temperature`, `vibration`, `humidity`, `pressure`, `energy_consumption`
  - `predicted_remaining_life`, `maintenance_required`, `machine_status`, `failure_type`

<br/>

## 🧑‍💻 기술 스택

- Python (NumPy, Pandas, Matplotlib, Seaborn, KaggleHub)
- Jupyter Notebook
- Git, GitHub

<br/>

## 📄 데이터 컬럼 설명

| 🏷️ 컬럼명 | 📘 설명 | 🧠 분석 활용 |
|-----------|---------|--------------|
| `timestamp` | 데이터가 기록된 시점 (YYYY-MM-DD HH:MM:SS) | 시간 흐름 분석, 고장 시점 확인 |
| `machine_id` | 각 기계를 식별하는 고유 ID | 기계별 이상 패턴 및 고장률 분석 |
| `temperature` 🌡️ | 기계의 온도 (℃) | 과열 여부, Overheating 고장 감지 |
| `vibration` 📈 | 진동 수치 (mm/s) | 기계 불안정성, 정비 필요성 탐지 |
| `humidity` 💧 | 습도 (%) | 주변 환경 조건 분석, 센서 오류 가능성 |
| `pressure` 🧯 | 압력 (단위 미상) | 기계 부하, 외부 환경 영향 분석 |
| `energy_consumption` ⚡ | 에너지 소비량 (kWh 등) | 작업 부하, 고장 전 과부하 탐색 |
| `predicted_remaining_life` ⏳ | 예측된 남은 수명 (단위 미상) | 정비 타이밍 결정, 고장 예측 근거 |
| `machine_status` 🛠️ | 기계 상태 코드<br>• `0`: 대기<br>• `1`: 가동<br>• `2`: 고장 | 실제 고장 여부 판단 기준 |
| `maintenance_required` 🧾 | 정비 필요 여부<br>• `0`: 필요 없음<br>• `1`: 정비 필요 | 예측 모델의 타깃 값 (정답값) |
| `failure_type` ❌ | 고장 원인 종류<br>`Overheating`, `Power Failure`, `Normal` 등 | 고장 유형별 조건 비교 분석 |
| `downtime_risk` 🚨 | 고장 시 예상 손실 위험도 (정량 수치) | 고장 우선순위 및 위험도 평가 |


## ❗ 주요 인사이트

- 기준 조건에 해당하지 않는 일부 데이터에서도 실제 고장이 발생함
- 이 경우, 주요 센서 값에 뚜렷한 이상 징후는 관찰되지 않음 → 정비 판단의 한계
- 센서 기반 감지 외에도 기계 상태 등 보조 지표의 병행 필요성 확인

<br/>
