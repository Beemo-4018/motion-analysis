# 🏃 Motion Analysis
### 물리치료 임상 경험 기반 재활운동 AI 동작평가 시스템

물리치료사 5년 임상 경험을 바탕으로 개발한 MediaPipe 기반 실시간 동작분석 시스템

---

## 📌 프로젝트 배경

5년간 물리치료사로 일하며 스쿼트 동작평가를 직접 수행했다.

> "사람마다 하체 길이가 다르기 때문에 정해진 각도 기준보다
> 개인별 패턴 변화와 좌우 비대칭을 보는 것이 더 중요하다"

이 임상 인사이트를 AI로 구현했다.

---

## ✅ 구현 기능 (현재 v1 - 스쿼트)

| 기능 | 설명 |
|------|------|
| 정면/측면 자동 감지 | 어깨 x좌표 차이로 카메라 방향 자동 인식 |
| 무릎 굴곡각도 측정 | 고관절-무릎-발목 코사인 법칙 |
| 좌우 비대칭 감지 | 15도 이상 차이 시 WARNING |
| 체간 앞쏠림 감지 | 50도 미만 시 WARNING |
| 동작 단계 자동 인식 | standing / descending / bottom / ascending |
| 횟수 자동 카운팅 | 최저점 → 기립 완료 시 1회 카운트 |
| 최저점 자동 캡쳐 | 정면/측면 구분하여 results 폴더에 자동 저장 |
| 영상 파일 분석 | iPhone 촬영 영상 입력 가능 |

## 🔜 예정 기능 (v2)

| 기능 | 설명 |
|------|------|
| Shoulder Ladder | 어깨 굴곡각도 + 체간 보상 패턴 감지 |
| SLR (Straight Leg Raise) | 다리 들기 각도 + 버티는 시간 자동 측정 |
| 재활 단계별 평가 | 초기/중기/후기 재활 운동 통합 |

---

## 📸 결과 예시

### 측면 분석 - 스쿼트 최저점
![측면 분석](results/auto_bottom_side_6rep.png)

```
Mode: side
Knee: 118.6   ← 무릎 굴곡각도
Trunk: 111.0  ← 체간 기울기
State: bottom ← 동작 단계
Count: 6      ← 횟수
```

### 정면 분석 - 스쿼트 최저점
![정면 분석](results/auto_bottom_front_4rep.png)

```
Mode: front
L Knee: 121.4  ← 왼쪽 무릎
R Knee: 118.6  ← 오른쪽 무릎
State: bottom
Count: 4
```

---

## 💡 임상 인사이트

**데이터 품질이 결과를 결정한다**

발목이 가려지면 랜드마크 추정 오류로 실제로는 없는 비대칭이 오탐된다.
입력 환경 표준화가 정확도의 핵심이다.

이는 임상에서 MRI 촬영 조건이 판독 품질에 영향을 주는 것과 동일한 원리다.

**단계별 색상 피드백**

| 단계 | 색상 | 의미 |
|------|------|------|
| standing | 초록 | 기립 상태 |
| descending | 노랑 | 내려가는 중 |
| bottom | 주황 | 최저점 |
| ascending | 하늘 | 올라오는 중 |

---

## 🗂️ 프로젝트 구조

```
motion-analysis/
├── data/
│   └── raw/               # 입력 영상 (gitignore)
├── scripts/
│   ├── pose_detection.py  # MediaPipe 기초
│   ├── angle_calculator.py # 관절각도 계산
│   ├── squat_analyzer.py  # 실시간 웹캠 분석
│   └── video_analyzer.py  # 영상 파일 분석
├── results/               # 분석 결과 이미지
│   ├── auto_bottom_front_Nrep.png
│   └── auto_bottom_side_Nrep.png
└── README.md
```

---

## 🛠️ 기술 스택

| 항목 | 내용 |
|------|------|
| Language | Python 3.10 |
| Pose Estimation | MediaPipe 0.10.9 |
| Image Processing | OpenCV |
| Numerical | NumPy |

---

## ⚙️ 실행 방법

```bash
# 환경 세팅
conda activate motion-env

# 실시간 웹캠 분석
python scripts/squat_analyzer.py

# 영상 파일 분석
python scripts/video_analyzer.py

# 단축키
# s → 현재 화면 즉시 저장
# q → 종료
```

---

## 📊 측정 항목 요약

| 모드 | 측정 항목 | 경고 기준 |
|------|-----------|-----------|
| 정면 | 좌우 무릎 각도 | 차이 15도 이상 |
| 정면 | 동작 단계 | - |
| 정면 | 횟수 카운팅 | - |
| 측면 | 무릎 굴곡각도 | - |
| 측면 | 체간 앞쏠림 | 50도 미만 |
| 측면 | 동작 단계 | - |
| 측면 | 횟수 카운팅 | - |
