
# IJmond Smoke Segmentation 프로젝트 개발 방향
## TMS 프로젝트의 공통 기술을 활용한 U-Net 기반 연기 분석 Agent 구축

---

# 1. 프로젝트 개발 방향

## 1.1 프로젝트 목적

이번 IJmond 연기 분석 프로젝트는 산업시설 이미지를 분석하여 연기 영역을 자동으로 분할하고, 연기의 면적과 불투명도 등을 정량적으로 분석하는 AI 시스템을 구축하는 것이다.

프로젝트를 진행할 때 모든 기능을 새롭게 개발할 필요는 없다.

현재 진행 중인 TMS AI 프로젝트에서 개발하는 Agent Orchestration, 데이터 저장 및 관리, API 연계 등의 공통 기술을 재사용할 수 있다.

다만 연기 영역을 실제로 분석하는 U-Net 모델과 이미지 처리 기능은 IJmond 프로젝트에서 직접 개발해야 한다.

따라서 두 프로젝트의 개발 범위를 다음과 같이 구분한다.

- **TMS 프로젝트:** AI Agent 실행 및 데이터 처리에 필요한 공통 기반 개발
- **IJmond 프로젝트:** U-Net 모델 학습 및 연기 이미지 분석 기능 개발

이렇게 역할을 구분하면 멘티들은 TMS 프로젝트에서 학습한 기술을 활용하면서 IJmond 프로젝트의 핵심 기술인 Image Segmentation에 집중할 수 있다.

## 1.2 핵심 개발 목표

1. IJmond 데이터셋을 활용한 U-Net 기반 Smoke Segmentation 모델 개발
2. 연기 영역을 Background / Low Opacity / High Opacity로 구분
3. 연기 영역의 면적, 비율, 불투명도 및 경계 정보 추출
4. TMS 프로젝트의 Agent Orchestration 구조를 활용한 Smoke Segmentation 전담 Agent 구현
5. TMS에서 등록한 사진과 사건 발생일자를 Smoke Segmentation Agent에 전달하는 연계 구조 구현
6. TMS 프로젝트의 데이터 저장 및 관리 구조를 활용한 분석 결과 저장·조회 기능 구현
7. 사건 발생일자 기준으로 연기 분석 결과의 시간적 변화 및 사건별 비교 분석 구현

---

# 2. TMS 프로젝트와 IJmond 프로젝트의 역할 구분

## 2.1 왜 역할을 구분해야 할까?

두 프로젝트는 서로 다른 목적을 가지고 있지만, 일부 기술과 개발 구조는 공통으로 활용할 수 있다.

예를 들어 AI Agent가 작업을 관리하고 Python Tool을 실행하는 구조는 두 프로젝트 모두에서 필요하다.

반면 U-Net을 활용한 연기 이미지 분석은 IJmond 프로젝트에서만 필요한 기능이다.

따라서 공통으로 사용할 수 있는 기술은 TMS 프로젝트에서 개발한 구조를 활용하고, 연기 분석에 특화된 기능은 IJmond 프로젝트에서 직접 구현한다.

## 2.2 프로젝트별 개발 범위

| 개발 항목 | TMS 프로젝트 | IJmond 프로젝트 |
|---|---|---|
| Agent Orchestration | 공통 구조 개발 | 재사용 |
| Agent 실행 및 상태 관리 | 공통 구조 개발 | 재사용 |
| Agent Tool 호출 구조 | 공통 구조 개발 | 이미지 분석 Tool 연결 |
| API 연계 | 공통 구조 개발 | 기존 구조 활용 |
| 데이터 저장 및 관리 | 공통 구조 개발 | 기존 구조 활용 |
| U-Net 모델 구성 | 해당 없음 | 직접 개발 |
| IJmond 데이터셋 분석 | 해당 없음 | 직접 수행 |
| 이미지 전처리 및 증강 | 해당 없음 | 직접 수행 |
| U-Net 학습 및 Fine-tuning | 해당 없음 | 직접 수행 |
| Smoke Segmentation | 해당 없음 | 직접 구현 |
| Boundary 추출 | 해당 없음 | 직접 구현 |
| 연기 분석 지표 계산 | 해당 없음 | 직접 구현 |
| Smoke Segmentation Agent | 공통 기반 제공 | 전용 Agent 구성 |
| 사건별·시간별 분석 | 공통 데이터 처리 기반 활용 | 분석 기능 구현 |

핵심은 TMS 프로젝트에서 개발한 Agent가 U-Net 모델까지 대신 개발해 주는 것은 아니라는 점이다.

TMS 프로젝트에서는 Agent가 Tool을 실행하고 데이터를 관리하는 공통 구조를 만들고, IJmond 프로젝트에서는 그 구조에 연기 분석 기능을 연결한다.

---

# 3. IJmond 프로젝트에서 직접 개발해야 하는 영역

## 3.1 U-Net 기반 Smoke Segmentation 모델

<badge color="success">직접 개발</badge>

이번 프로젝트의 가장 중요한 AI 개발 과제다.

U-Net은 이미지의 각 픽셀을 분석하여 해당 픽셀이 어떤 영역에 속하는지 분류하는 Semantic Segmentation 모델이다.

이번 프로젝트에서는 연기를 다음 3개 클래스로 구분한다.

| Class | 의미 |
|---|---|
| 0 | Background |
| 1 | Low Opacity Smoke |
| 2 | High Opacity Smoke |

예를 들어 산업시설 사진이 입력되면 U-Net은 이미지의 각 픽셀을 분석하여 연기가 아닌 영역, 저불투명도 연기, 고불투명도 연기로 구분한다.

### 권장 모델 구성

처음부터 U-Net 모델을 직접 설계하고 학습하는 것은 시간이 많이 필요하다.

따라서 이미 학습된 이미지 특징 추출 모델을 활용하는 방식을 권장한다.

```text
U-Net
 └── ResNet34 Encoder
       └── ImageNet Pretrained
             │
             ▼
        U-Net Decoder
             │
             ▼
       3-Class Output
             │
             ▼
       Smoke Segmentation
```

- U-Net: 이미지의 픽셀 단위 영역 분할
- ResNet34: 이미지의 특징 추출
- ImageNet Pretrained: 사전 학습된 가중치 활용
- Fine-tuning: IJmond 이미지와 Mask를 사용하여 연기 분할에 맞게 학습

주의할 점은 ImageNet Pretrained 모델이 이미 연기를 인식하도록 학습된 것은 아니라는 것이다.

일반적인 이미지 특징을 추출하는 능력을 활용하고, IJmond 데이터셋으로 별도 학습해야 한다.

## 3.2 U-Net 학습 진행 순서

```text
IJmond 데이터셋 다운로드
          │
          ▼
이미지와 Mask 구조 분석
          │
          ▼
Train / Validation / Test 분리
          │
          ▼
이미지 전처리
          │
          ▼
Train 데이터 증강
          │
          ▼
U-Net + ResNet34 구성
          │
          ▼
IJmond 데이터로 Fine-tuning
          │
          ▼
Validation 성능 평가
          │
          ▼
최종 모델 저장
```

### 주요 학습 작업

| 순서 | 작업 | 설명 |
|---|---|---|
| 1 | Dataset 분석 | 이미지와 Mask의 대응 관계 확인 |
| 2 | 데이터 분리 | Train / Validation / Test 분리 |
| 3 | 전처리 | 이미지 크기 조정 및 정규화 |
| 4 | 데이터 증강 | Train 데이터에 다양한 변형 적용 |
| 5 | 모델 구성 | U-Net + ResNet34 |
| 6 | Fine-tuning | IJmond 데이터로 모델 학습 |
| 7 | 성능 평가 | IoU, Dice, Precision, Recall 확인 |
| 8 | 모델 저장 | Agent에서 사용할 학습 모델 저장 |

### 학습 시 중요한 사항

- 원본 이미지 기준으로 데이터를 분리한다.
- 같은 원본 이미지에서 생성된 Crop이 Train과 Test에 동시에 포함되지 않도록 한다.
- 데이터 증강은 Train에만 적용한다.
- 이미지와 Mask에는 동일한 기하학적 변환을 적용한다.
- Mask는 Nearest Neighbor 보간법을 사용하여 클래스 값이 유지되도록 한다.
- Accuracy만 보지 말고 IoU와 Dice를 함께 확인한다.

## 3.3 Segmentation 및 Boundary 추출

<badge color="success">직접 개발</badge>

U-Net이 연기 영역을 분할하면 후처리를 통해 연기의 경계를 추출한다.

이번 프로젝트에서는 별도의 Boundary AI 모델을 학습하지 않고 OpenCV를 활용한다.

```text
원본 이미지
     │
     ▼
U-Net 추론
     │
     ▼
Segmentation Mask
     │
     ├── Background
     ├── Low Opacity Smoke
     └── High Opacity Smoke
             │
             ▼
       Smoke Mask 생성
             │
             ▼
      OpenCV Contour
             │
             ▼
        Boundary 추출
```

### 주요 분석 항목

| 분석 항목 | 설명 |
|---|---|
| Smoke Ratio | 전체 이미지에서 연기가 차지하는 비율 |
| High Opacity Ratio | 전체 이미지에서 고불투명도 연기가 차지하는 비율 |
| High Opacity Share | 전체 연기 중 고불투명도 연기의 비율 |
| Contour Area | 연기 영역의 픽셀 면적 |
| Perimeter | 연기 영역의 외곽선 길이 |
| Bounding Box | 연기 영역을 포함하는 사각형 |

이러한 분석 지표는 LLM이 추정하는 것이 아니라 Python 코드로 계산해야 한다.

---

# 4. TMS 프로젝트에서 재사용할 영역

## 4.1 Agent Orchestration

<badge color="success">TMS 구조 재사용</badge>

TMS 프로젝트에서 개발하는 Multi-Agent 및 Orchestration 구조를 활용한다.

Orchestration은 여러 Agent와 Tool의 실행 순서를 관리하는 역할이다.

IJmond 프로젝트에서는 기존 구조를 활용하여 Smoke Segmentation Agent를 실행한다.

```text
       TMS Agent Orchestration
                 │
                 ▼
       Smoke Segmentation Agent
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   이미지      U-Net      분석 지표
   전처리      추론       계산
       │         │         │
       └─────────┼─────────┘
                 ▼
             결과 저장
                 │
                 ▼
             TMS 전달
```

### 재사용할 기능

- Agent 실행 흐름 관리
- Agent 간 작업 전달
- Tool 호출 및 실행
- 실행 상태 관리
- 오류 처리 및 재시도
- 분석 결과 전달

다만 TMS 프로젝트에서 개발한 구조가 IJmond 프로젝트에 그대로 연결되는 것은 아니다.

실제 코드와 실행 환경이 다르다면 API, Tool 인터페이스 및 데이터 형식을 맞추는 작업이 필요하다.

## 4.2 Smoke Segmentation 전담 Agent

<badge color="success">공통 기반 재사용 + 전용 Tool 개발</badge>

Smoke Segmentation Agent는 이미지 분석 업무를 전담하는 Agent다.

기존 TMS Agent 실행 구조를 활용하되, 이미지 분석에 필요한 전용 Tool을 새롭게 연결한다.

| Tool | 기능 | 개발 방식 |
|---|---|---|
| validate_input | 이미지 및 사건 정보 검증 | IJmond 전용 |
| preprocess_image | 이미지 전처리 | IJmond 전용 |
| run_unet_inference | U-Net 모델 실행 | IJmond 전용 |
| extract_smoke_mask | 연기 Mask 생성 | IJmond 전용 |
| extract_boundary | Boundary 추출 | IJmond 전용 |
| calculate_smoke_metrics | 연기 분석 지표 계산 | IJmond 전용 |
| save_analysis_result | 분석 결과 저장 | TMS 공통 구조 활용 |
| update_tms_result | TMS에 분석 결과 전달 | TMS 연계 구조 활용 |

### Agent의 역할

Agent는 모든 수치를 직접 판단하는 것이 아니다.

- Agent: 분석 작업의 전체 실행 순서 관리
- U-Net: 이미지의 연기 영역 분할
- OpenCV: Boundary 추출
- Python: 분석 지표 계산
- Storage: 결과 저장
- LLM: 계산 결과를 자연어로 설명

이처럼 각 기능의 역할을 명확하게 분리해야 분석 결과의 신뢰성을 확보할 수 있다.

---

# 5. TMS와 IJmond Agent 연계

## 5.1 TMS에서 사진 및 사건 발생일자 등록

<badge color="success">TMS 공통 구조 활용</badge>

사용자가 TMS 화면에서 산업시설 사진과 사건 발생일자를 등록한다.

TMS는 해당 정보를 바탕으로 Smoke Segmentation Agent에 분석을 요청한다.

### 주요 입력 정보

| 항목 | 설명 |
|---|---|
| event_id | 사건 식별자 |
| facility_id | 사업장 식별자 |
| image_uri | 원본 이미지 저장 경로 |
| event_datetime | 사건 발생일자 및 시간 |
| captured_at | 실제 이미지 촬영일시 |
| camera_id | 촬영 카메라 식별자 |

사건 발생일자와 이미지 촬영일자는 서로 다를 수 있으므로 별도로 관리한다.

## 5.2 Agent 분석 요청

이미지 자체를 Agent에 직접 전달하는 방법도 있지만, Object Storage에 이미지를 저장한 후 URI를 전달하는 방식을 권장한다.

```json
{
  "event_id": "EVT-20261120-001",
  "facility_id": "FACILITY-001",
  "event_datetime": "2026-11-20T09:00:00",
  "captured_at": "2026-11-20T08:58:00",
  "camera_id": "CAM-001",
  "image_uri": "storage/events/EVT-20261120-001/image.jpg"
}
```

## 5.3 전체 연계 흐름

```text
┌────────────────────────────┐
│            TMS             │
│                            │
│  사진 등록                  │
│  사건 발생일자 등록         │
│  분석 요청                  │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│     Agent Orchestration    │
│       (TMS 공통 구조)       │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│  Smoke Segmentation Agent  │
│                            │
│  이미지 검증                │
│  이미지 전처리              │
│  U-Net 추론                │
│  Boundary 추출              │
│  분석 지표 계산             │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│      Analysis Storage      │
│                            │
│  Mask 저장                  │
│  Boundary 저장              │
│  분석 결과 저장             │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│            TMS             │
│                            │
│  분석 결과 수신 및 조회     │
└────────────────────────────┘
```

이미지 분석은 처리 시간이 필요하므로 비동기 방식으로 진행하는 것이 적합하다.

TMS는 분석 요청을 접수한 뒤 결과가 완료되면 전달받도록 구성한다.

---

# 6. 분석 결과 저장 및 시계열 분석

## 6.1 TMS 데이터 저장 구조 활용

<badge color="success">TMS 공통 구조 활용</badge>

TMS 프로젝트에서 개발하는 데이터 저장 및 관리 구조를 활용한다.

다만 NOx, SO2 측정 데이터와 이미지 분석 데이터는 데이터 특성이 다르므로 별도 데이터 영역으로 관리한다.

기존 Bronze / Silver / Gold 구조를 적용하면 다음과 같이 구성할 수 있다.

| 데이터 영역 | IJmond 적용 내용 |
|---|---|
| Bronze | 원본 이미지, 사건 정보, 원본 메타데이터 |
| Silver | Segmentation Mask, Boundary, 분석 결과 |
| Gold | 사건별 분석 요약, 시간별 연기 분석 통계 |

### 데이터 흐름

```text
Bronze
 ├── Original Image
 └── Event Metadata
          │
          ▼
Silver
 ├── Segmentation Mask
 ├── Boundary Image
 └── Analysis Metrics
          │
          ▼
Gold
 ├── Event Analysis Summary
 ├── Daily Smoke Statistics
 └── Time-series Analysis
```

## 6.2 사건 발생일자 기준 시계열 분석

<badge color="success">TMS 데이터 처리 구조 활용</badge>

분석 결과에 사건 발생일자와 촬영일자를 함께 저장하면 시간 순서에 따라 연기 분석 결과를 비교할 수 있다.

예를 들어 다음과 같은 분석이 가능하다.

| 사건 발생일자 | Smoke Ratio | High Opacity Ratio |
|---|---:|---:|
| 09:00 | 5.2% | 1.1% |
| 10:00 | 8.7% | 2.5% |
| 11:00 | 12.3% | 4.8% |
| 12:00 | 7.1% | 2.0% |

이를 통해 다음 항목을 분석할 수 있다.

- 시간대별 Smoke Ratio 변화
- 시간대별 High Opacity Ratio 변화
- 사건별 연기 분석 결과 비교
- 카메라별 연기 분석 결과 비교
- 특정 기간의 연기 발생 패턴

주의할 점은 IJmond 데이터셋이 연속 동영상이 아니라 시간 정보가 있는 이미지 데이터라는 것이다.

따라서 시간대별 이미지 분석 결과의 변화를 비교하는 것이며, 실제 연기의 지속시간이나 이동 경로를 측정하는 것은 아니다.

---

# 7. 최종 개발 아키텍처

두 프로젝트의 기술을 결합하면 다음과 같은 구조가 된다.

```text
┌──────────────────────────────────────────────┐
│                TMS PROJECT                   │
│                                              │
│  Agent Orchestration                         │
│  Agent 실행 및 상태 관리                      │
│  API 연계                                    │
│  데이터 저장 및 관리                          │
│  분석 결과 조회                               │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              IJMOND PROJECT                  │
│                                              │
│       Smoke Segmentation Agent               │
│                       │                      │
│          ┌────────────┼────────────┐         │
│          ▼            ▼            ▼         │
│     Image Tool    U-Net Tool   Analysis Tool │
│          │            │            │         │
│          ▼            ▼            ▼         │
│     Preprocessing  Segmentation  Metrics     │
│                       │                      │
│                       ▼                      │
│                 Boundary 추출                │
│                       │                      │
│                       ▼                      │
│                  결과 저장                    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                 TMS RESULT                   │
│                                              │
│  사건별 분석 결과                             │
│  시간별 연기 변화                             │
│  분석 이력 및 모델 버전                       │
└──────────────────────────────────────────────┘
```

---

# 8. 개발 우선순위 및 진행 방법

학습 시간이 제한되어 있으므로 모든 기능을 동일한 수준으로 개발할 필요는 없다.

특히 IJmond 프로젝트에서는 U-Net 모델 학습과 실제 이미지 분석 기능 구현에 집중하는 것이 좋다.

| 순서 | 개발 항목 | 개발 방식 | 우선순위 |
|---|---|---|---|
| 1 | IJmond 데이터셋 분석 | 직접 수행 | 필수 |
| 2 | U-Net 모델 구성 | Pretrained 활용 | 필수 |
| 3 | 모델 Fine-tuning | 직접 수행 | 필수 |
| 4 | Segmentation 결과 평가 | 직접 수행 | 필수 |
| 5 | Boundary 추출 | OpenCV 활용 | 필수 |
| 6 | 연기 분석 지표 계산 | Python 구현 | 필수 |
| 7 | Smoke Segmentation Agent | TMS 공통 구조 재사용 | 필수 |
| 8 | 사진 및 사건 정보 연계 | TMS API 구조 재사용 | 필수 |
| 9 | 분석 결과 저장 | TMS 데이터 구조 재사용 | 필수 |
| 10 | 사건별·시간별 분석 | Python 및 SQL 활용 | 필수 |
| 11 | YOLO Segmentation 비교 | 별도 실험 | 선택 |
| 12 | Boundary 모델 고도화 | 추가 연구 | 선택 |

## 8.1 꼭 직접 해야 하는 부분

다음 기능은 IJmond 프로젝트의 핵심 학습 과제이므로 직접 구현하는 것을 권장한다.

1. IJmond 데이터셋 구조 이해
2. 이미지와 Mask 전처리
3. 데이터 증강
4. U-Net 모델 구성
5. IJmond 데이터로 Fine-tuning
6. Segmentation 성능 평가
7. Boundary 추출
8. 연기 분석 지표 계산

## 8.2 TMS 프로젝트에서 재사용할 부분

다음 기능은 TMS 프로젝트에서 개발하는 공통 구조를 활용한다.

1. Agent Orchestration
2. Agent 실행 및 상태 관리
3. Tool 호출 구조
4. API 연계
5. 데이터 저장 및 메타데이터 관리
6. 분석 결과 조회 및 시계열 데이터 처리

단, 기존 코드와 데이터 구조가 실제로 재사용 가능한지 확인하고 필요한 인터페이스를 맞춰야 한다.

---

# 9. 멘티들이 반드시 이해해야 할 핵심 사항

이번 프로젝트에서 가장 중요한 것은 두 프로젝트의 개발 역할을 명확하게 구분하는 것이다.

| 질문 | 답변 |
|---|---|
| U-Net 모델은 어디서 개발하나요? | IJmond 프로젝트에서 직접 개발 |
| U-Net 학습도 직접 하나요? | 네. IJmond 데이터로 Fine-tuning 수행 |
| Agent 구조도 새롭게 개발하나요? | TMS 프로젝트의 공통 구조 재사용 |
| Smoke Segmentation Agent는 개발하나요? | 네. 기존 Agent 구조에 이미지 분석 Tool 연결 |
| 이미지 분석 Tool은 재사용하나요? | 아니요. IJmond 전용 기능으로 개발 |
| TMS에서 사진을 등록할 수 있나요? | TMS 연계 기능으로 구현 |
| 분석 결과는 어디에 저장하나요? | TMS 프로젝트의 데이터 저장 구조 활용 |
| 시간별 분석도 가능한가요? | 네. 사건 발생일자와 촬영일자를 기준으로 비교 |

## 최종 정리

**TMS 프로젝트에서는 AI Agent와 데이터 처리에 필요한 공통 기반을 개발하고, IJmond 프로젝트에서는 U-Net 기반 연기 이미지 분석 기능을 직접 개발한다.**

이렇게 하면 멘티들은 TMS 프로젝트에서 학습한 Agent와 데이터 엔지니어링 기술을 실제 이미지 분석 서비스에 적용하는 경험을 할 수 있다.

최종적으로는 다음과 같은 흐름을 구현하는 것이 목표다.

```text
TMS에서 사진 및 사건 발생일자 등록
                │
                ▼
       Smoke Segmentation Agent
                │
                ▼
       U-Net 모델로 연기 분할
                │
                ▼
       Boundary 및 분석 지표 계산
                │
                ▼
          분석 결과 저장
                │
                ▼
       TMS에서 분석 결과 조회
                │
                ▼
        사건별·시간별 비교 분석
```

**핵심은 모델 학습은 IJmond 프로젝트에서 직접 수행하고, Agent 및 데이터 처리 기반은 TMS 프로젝트에서 재사용하여 개발 시간을 줄이는 것이다.**
