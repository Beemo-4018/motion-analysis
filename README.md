# 🏃 Motion Analysis
### 물리치료 동작평가 AI 시스템

물리치료사 5년 임상 경험을 바탕으로 개발한
MediaPipe 기반 실시간 동작분석 시스템

---

## 📌 프로젝트 배경

5년간 물리치료사로 일하며 스쿼트 동작평가를 직접 수행했다.

> "사람마다 하체 길이가 다르기 때문에
> 정해진 각도 기준보다 개인별 패턴과
> 좌우 비대칭 변화를 보는 것이 더 중요하다"

이 임상 인사이트를 AI로 구현했다.

---

## 🎯 주요 기능

### v1 (현재)
| 기능 | 설명 |
|------|------|
| 정면/측면 자동 감지 | 어깨 x좌표 차이로 카메라 방향 자동 인식 |
| 무릎 굴곡각도 측정 | 고관절-무릎-발목 코사인 법칙 |
| 좌우 비대칭 감지 | 15도 이상 차이 시 WARNING |
| 체간 앞쏠림 감지 | 50도 미만 시 WARNING |
| 영상 파일 분석 | iPhone 촬영 영상 입력 가능 |

### v2 (예정)
| 기능 | 설명 |
|------|------|
| 동작 단계 인식 | 서있음/내려가는 중/최저점/올라오는 중 |
| 최저점 자동 기록 | 스쿼트 1회당 최대 굴곡각도 저장 |
| 피로 누적 감지 | 반복 동작에서 각도 변화 추적 |

---

## 📸 결과 예시

### 측면 분석
![](https://velog.velcdn.com/images/zlfktk/post/a51dffcc-275d-4e09-af59-d8e4ab9a993e/image.png)

![측면 분석](results/squat_sagital_1.png)

### 정면 분석
![](https://velog.velcdn.com/images/zlfktk/post/0820f459-c07f-4830-b3dc-53d93c3164c3/image.png)

![정면 분석](results/squat_frontal.png)

---

## 🗂️ 프로젝트 구조
```
motion-analysis/
├── data/
│   └── raw/          # 입력 영상
├── scripts/
│   ├── pose_detection.py    # MediaPipe 기초
│   ├── angle_calculator.py  # 관절각도 계산
│   ├── squat_analyzer.py    # 실시간 웹캠 분석
│   └── video_analyzer.py    # 영상 파일 분석
├── results/          # 분석 결과 이미지
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

## 💡 핵심 인사이트

**데이터 품질이 결과를 결정한다**

발목이 가려지면 랜드마크 추정 오류로
실제로는 없는 비대칭이 오탐된다.
입력 환경 표준화가 정확도의 핵심이다.

이는 임상에서 MRI 촬영 조건이
판독 품질에 영향을 주는 것과 동일한 원리다.

---

## ⚙️ 실행 방법
```bash
# 환경 세팅
conda activate motion-env

# 실시간 웹캠 분석
python scripts/squat_analyzer.py

# 영상 파일 분석
python scripts/video_analyzer.py
```
