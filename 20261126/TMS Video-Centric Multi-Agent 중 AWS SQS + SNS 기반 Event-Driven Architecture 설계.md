# TMS Video-Centric Multi-Agent 시스템
## AWS SQS + SNS 기반 Event-Driven Architecture 설계 및 구현 가이드

## 1. 프로젝트 개요

본 프로젝트는 굴뚝 및 배출시설에서 촬영된 영상을 등록하면 AI가 영상을 분석하고, 연기 발생 여부와 지속 시간 등을 종합적으로 판단하여 최종 상태를 결정하는 Video-Centric Multi-Agent 시스템을 구축하는 것을 목표로 한다.

여기서 AWS SQS와 SNS를 함께 활용하여 영상 분석 요청, AI 분석 작업, 최종 판정 이벤트 및 후속 시스템 처리를 비동기 방식으로 연결한다.

핵심 설계 방향은 다음과 같다.

- SQS를 이용하여 영상 분석 작업을 안정적으로 전달한다.
- Video Analysis Agent가 영상 분석 Workflow를 수행한다.
- AI Inference Layer가 여러 분석 결과를 종합하여 최종 판정한다.
- SNS를 이용하여 최종 판정 이벤트를 여러 시스템에 전달한다.
- 각 Consumer는 SQS를 통해 독립적으로 후속 작업을 수행한다.

**핵심 설계 원칙**

> SQS는 작업을 전달하고, SNS는 이벤트를 전파한다. AI Inference Layer는 분석 결과를 종합하여 최종 판정을 통제한다.

---

## 2. AWS SQS와 SNS의 역할

### 2.1 AWS SQS (Simple Queue Service)

SQS는 AWS에서 제공하는 메시지 큐 서비스다.

시스템 간 작업 요청을 메시지 형태로 전달하고, Consumer가 처리할 수 있을 때 메시지를 가져가도록 한다.

**주요 기능**

| 기능 | 설명 |
|---|---|
| Message Queue | 작업 요청 메시지 저장 |
| 비동기 처리 | Producer와 Consumer 실행 시점 분리 |
| 작업 분산 | 여러 Worker가 작업을 나누어 처리 |
| Visibility Timeout | 처리 중인 메시지를 다른 Consumer에게 일시적으로 숨김 |
| Retry | 실패한 메시지 재처리 |
| Dead Letter Queue | 반복 실패한 메시지 별도 관리 |

**TMS 적용 예시**

영상이 등록되면 SQS에 분석 요청을 등록한다.

Video Analysis Agent는 SQS에서 메시지를 가져와 영상 분석을 수행한다.

```text
[영상 등록 시스템]
        |
        v
   [AWS SQS]
        |
        v
[Video Analysis Agent]
        |
        v
     [영상 분석]
```

### 2.2 AWS SNS (Simple Notification Service)

SNS는 메시지 발행 및 구독(Publish/Subscribe) 방식의 메시징 서비스다.

하나의 이벤트를 여러 시스템에 동시에 전달할 수 있다.

**주요 기능**

| 기능 | 설명 |
|---|---|
| Topic | 이벤트를 발행하는 논리적 채널 |
| Publisher | 이벤트를 SNS Topic에 발행 |
| Subscriber | Topic을 구독하는 시스템 |
| Fan-out | 하나의 이벤트를 여러 구독자에게 전달 |
| Message Filtering | 구독자별로 필요한 이벤트만 전달 |
| 재시도 | 지원되는 구독 유형에 대한 전달 재시도 |

**TMS 적용 예시**

AI Inference Layer에서 WARNING 판정이 발생하면 SNS Topic에 이벤트를 발행한다.

SNS는 연결된 여러 SQS Queue에 이벤트를 전달한다.

```text
       [SNS Topic]
             |
       +-----+-----+
       |     |     |
       v     v     v
     SQS-1 SQS-2 SQS-3
       |     |     |
       v     v     v
   Dashboard Slack Observability
```

### 2.3 SQS와 SNS 비교

| 비교 항목 | AWS SQS | AWS SNS |
|---|---|---|
| 핵심 역할 | 작업 대기열 관리 | 이벤트 발행 및 전파 |
| 통신 방식 | Queue | Publish / Subscribe |
| 메시지 처리 | Consumer가 가져감 | 구독자에게 전달 |
| 주요 목적 | 작업의 안정적인 처리 | 여러 시스템에 이벤트 전달 |
| 대표 기능 | Visibility Timeout, DLQ | Topic, Fan-out, Filter |
| TMS 활용 | 영상 분석 요청 | AI 최종 판정 이벤트 |

쉽게 이해하면 다음과 같다.

- SQS: 처리해야 할 작업을 줄 세운다.
- SNS: 이벤트가 발생했다는 사실을 여러 시스템에 알린다.

---

## 3. 전체 시스템 아키텍처

### 3.1 전체 데이터 흐름

```text
                  [영상 등록]
                       |
                       v
                  [AWS SQS]
                       |
                       v
             [Video Analysis Agent]
                       |
             +---------+---------+
             |         |         |
             v         v         v
           YOLO   Segmentation Temporal
             |         |         |
             +---------+---------+
                       |
                       v
              [Analysis Result]
                       |
                       v
              [AI Inference Layer]
                       |
                       v
                  [최종 판정]
                       |
              +--------+--------+
              |        |        |
              v        v        v
            정상      주의      경고
                       |
                       v
                  [AWS SNS]
                       |
              +--------+---------+
              |        |         |
              v        v         v
           SQS-1     SQS-2     SQS-3
              |        |         |
              v        v         v
          Dashboard  Slack   Observability
           Worker   Worker     Worker
```

### 3.2 구성 요소별 역할

| 구성 요소 | 주요 역할 |
|---|---|
| 영상 등록 | 영상 등록 및 분석 요청 생성 |
| AWS SQS | 영상 분석 요청 메시지 전달 |
| Video Analysis Agent | 영상 분석 Workflow 관리 |
| YOLO Detection | 연기 객체 탐지 |
| Segmentation | 연기 영역을 픽셀 단위로 추출 |
| Temporal Analysis | 시간별 연기 발생 및 지속 상태 분석 |
| AI Inference Layer | 분석 결과 종합 및 최종 판정 |
| AWS SNS | 최종 판정 이벤트 발행 |
| Dashboard SQS | 대시보드 갱신 작업 전달 |
| Slack SQS | 알림 작업 전달 |
| Observability SQS | Agent 실행 및 판정 이력 수집 |

---

## 4. 단계별 처리 과정

### 4.1 영상 등록 및 분석 요청

사용자가 영상을 등록하면 Backend API가 분석 요청을 생성한다.

영상 파일 자체를 SQS에 넣는 것이 아니라 영상 식별자와 영상 접근 정보를 메시지로 전달한다.

**분석 요청 메시지 예시**

```json
{
  "event_id": "TMS-20261125-001",
  "video_id": "VIDEO-001",
  "stack_code": "STACK-001",
  "event_type": "VIDEO_ANALYSIS_REQUEST",
  "video_uri": "video-storage/video001.mp4",
  "analysis_type": [
    "YOLO_DETECTION",
    "SEGMENTATION",
    "TEMPORAL_ANALYSIS"
  ],
  "requested_at": "2026-11-25T07:00:00+09:00"
}
```

`video_uri`는 예시이며 실제 운영 환경에서는 모든 Worker가 접근할 수 있는 공유 저장소 경로나 인증된 접근 정보를 사용해야 한다.

### 4.2 SQS를 통한 영상 분석 작업 전달

SQS는 영상 분석 요청을 보관하고 Video Analysis Agent가 처리할 수 있도록 전달한다.

**처리 흐름**

```text
[영상 등록 API]
       |
       v
[SQS 분석 요청 Queue]
       |
       v
[Video Analysis Agent]
       |
       v
[영상 분석 Workflow 실행]
```

이 구조를 사용하면 영상 등록 시스템은 AI 분석이 끝날 때까지 기다릴 필요가 없다.

또한 여러 Worker를 운영하여 영상 분석 요청을 분산 처리할 수 있다.

### 4.3 Video Analysis Agent

Video Analysis Agent는 영상 분석 Workflow를 관리한다.

주요 작업은 다음과 같다.

| 분석 모듈 | 기능 |
|---|---|
| YOLO Detection | 연기 객체 탐지 |
| Segmentation | 연기 영역 추출 |
| Temporal Analysis | 시간에 따른 연기 발생 및 변화 분석 |

**내부 Workflow 예시**

```text
[SQS 메시지 수신]
       |
       v
[Video Analysis Agent]
       |
       v
[영상 데이터 접근]
       |
       v
[프레임 추출]
       |
       +----------------------+
       |                      |
       v                      v
[YOLO Detection]       [Segmentation]
       |                      |
       v                      v
[연기 객체 탐지]       [연기 영역 분할]
       |                      |
       +----------+-----------+
                  |
                  v
          [Temporal Analysis]
                  |
                  v
          [Analysis Result]
```

분석 결과에는 탐지 신뢰도, 연기 면적, 지속 시간, 탐지 빈도 등이 포함된다.

### 4.4 AI Inference Layer

AI Inference Layer는 각 분석 모듈에서 전달된 결과를 종합하여 최종 상태를 결정한다.

**주요 입력 데이터**

| 입력 항목 | 설명 |
|---|---|
| Detection Confidence | 연기 탐지 신뢰도 |
| Smoke Area Ratio | 연기 영역 비율 |
| Smoke Duration | 연기 지속 시간 |
| Detection Frequency | 연기 탐지 빈도 |
| Area Growth Rate | 연기 면적 증가율 |
| Temporal Pattern | 시간에 따른 연기 발생 패턴 |
| Analysis Quality | 분석 결과의 품질 및 유효성 |

**최종 판정 상태**

| 상태 | 의미 | 판정 예시 |
|---|---|---|
| NORMAL | 정상 | 유의미한 연기 탐지 없음 |
| CAUTION | 주의 | 일시적이거나 지속성이 낮은 연기 탐지 |
| WARNING | 경고 | 지속적이고 뚜렷한 연기 발생 |

실제 판정 임계값은 실험 데이터와 검증 결과를 기반으로 결정한다.

영상 분석 실패나 데이터 부족은 정상으로 처리하지 않고 별도 상태로 관리한다.

예를 들어 다음 상태를 사용할 수 있다.

- ANALYSIS_FAILED
- INSUFFICIENT_DATA

### 4.5 SNS를 통한 최종 판정 이벤트 발행

AI Inference Layer가 최종 판정을 완료하면 SNS Topic에 이벤트를 발행한다.

**SNS 이벤트 메시지 예시**

```json
{
  "event_id": "TMS-EVENT-001",
  "video_id": "VIDEO-001",
  "stack_code": "STACK-001",
  "event_type": "SMOKE_WARNING",
  "status": "WARNING",
  "smoke_duration_sec": 12.5,
  "smoke_area_ratio": 0.28,
  "reason_codes": [
    "PERSISTENT_SMOKE",
    "INCREASING_SMOKE_AREA"
  ],
  "created_at": "2026-11-25T07:00:00+09:00"
}
```

SNS Topic은 최종 판정 이벤트를 구독 중인 여러 시스템에 전달한다.

### 4.6 SNS + SQS Fan-out 구조

SNS Topic 하나에 여러 SQS Queue를 연결한다.

이를 Fan-out 패턴이라고 한다.

**구독 Queue 구성**

| Queue | Consumer | 역할 |
|---|---|---|
| TMS-Dashboard-Queue | Dashboard Worker | 분석 결과 및 상태 갱신 |
| TMS-Slack-Queue | Slack Worker | 담당자 경고 알림 |
| TMS-Observability-Queue | Observability Worker | Agent 실행 및 판정 이력 기록 |

**처리 흐름**

```text
                  [AI Inference Layer]
                           |
                           v
                    [SNS Topic]
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
       [Dashboard SQS] [Slack SQS] [Observability SQS]
             |             |             |
             v             v             v
       Dashboard       Slack 알림     실행 이력 저장
        Worker          Worker          Worker
```

각 Consumer는 독립적으로 메시지를 처리한다.

예를 들어 Slack 알림이 실패하더라도 Dashboard 갱신과 Observability 처리는 계속 진행할 수 있다.

---

## 5. 메시지 및 이벤트 설계

### 5.1 이벤트 유형 정의

| 이벤트 유형 | 발생 시점 | 전달 대상 |
|---|---|---|
| VIDEO_ANALYSIS_REQUEST | 영상 등록 완료 | Video Analysis Agent |
| VIDEO_ANALYSIS_STARTED | 분석 시작 | Observability |
| VIDEO_ANALYSIS_COMPLETED | 영상 분석 완료 | 후속 판정 처리 |
| INFERENCE_COMPLETED | 최종 판정 완료 | Dashboard / Observability |
| SMOKE_WARNING | 경고 판정 발생 | Slack / Dashboard |
| ANALYSIS_FAILED | 분석 실패 | Observability / 운영 담당자 |

이벤트 유형을 명확히 정의하면 새로운 Agent나 후속 시스템을 추가할 때 기존 구조를 크게 변경하지 않아도 된다.

### 5.2 처리 상태와 판정 상태 구분

영상 분석 작업의 처리 상태와 AI의 최종 판정 상태는 별도로 관리한다.

**작업 처리 상태**

| 상태 | 설명 |
|---|---|
| REQUESTED | 분석 요청 생성 |
| QUEUED | SQS 대기 중 |
| PROCESSING | 분석 진행 중 |
| COMPLETED | 분석 완료 |
| FAILED | 분석 실패 |
| RETRYING | 재처리 중 |

**AI 최종 판정 상태**

| 상태 | 설명 |
|---|---|
| NORMAL | 정상 |
| CAUTION | 주의 |
| WARNING | 경고 |
| INSUFFICIENT_DATA | 판정 데이터 부족 |

예를 들어 다음과 같이 관리한다.

```json
{
  "video_id": "VIDEO-001",
  "processing_status": "COMPLETED",
  "inference_status": "WARNING"
}
```

---

## 6. 장애 처리 및 안정성 확보

### 6.1 SQS Dead Letter Queue

SQS Consumer에서 동일한 메시지가 반복적으로 실패하면 DLQ로 이동시킨다.

```text
[SQS Main Queue]
        |
        v
[Video Analysis Agent]
        |
        v
    [분석 실패]
        |
        v
     [재시도]
        |
        v
 [최대 수신 횟수 초과]
        |
        v
       [DLQ]
        |
        v
  [오류 분석 및 재처리]
```

### 6.2 주요 안정성 설계

| 항목 | 설계 방향 |
|---|---|
| 중복 메시지 | event_id 기반 멱등성 처리 |
| 처리 시간 초과 | Visibility Timeout 연장 |
| 반복 실패 | DLQ 분리 |
| 메시지 유실 방지 | 결과 영구 저장 후 메시지 삭제 |
| 이벤트 중복 발행 | 이벤트 ID 및 중복 방지 처리 |
| Consumer 장애 | Queue에 메시지를 보관하고 복구 후 처리 |

**중요 사항**

SQS Standard Queue는 최소 한 번 전달 방식이므로 중복 메시지가 발생할 수 있다.

따라서 동일한 영상 분석 요청이 여러 번 전달되어도 최종 결과가 중복 생성되지 않도록 멱등성을 확보해야 한다.

또한 SNS와 SQS를 연결할 때 각 SQS Queue에 SNS Topic의 메시지 전달 권한을 부여해야 한다.

---

## 7. AWS 서비스 구성

| AWS 서비스 | 적용 영역 | 주요 역할 |
|---|---|---|
| Amazon SQS | 영상 분석 요청 | 비동기 작업 대기열 |
| Amazon SQS | 후속 작업 처리 | Dashboard, Slack, Observability |
| Amazon SNS | 판정 이벤트 발행 | 이벤트 Fan-out |
| Amazon SQS DLQ | 장애 대응 | 반복 실패 메시지 보관 |
| AWS IAM | 보안 | 서비스별 최소 권한 관리 |
| Amazon CloudWatch | 모니터링 | Queue 적체, 처리량, 오류 모니터링 |

---

## 8. 멘티 구현 권장 순서

처음부터 모든 기능을 구현하기보다는 단계별로 확장한다.

| 단계 | 구현 항목 | 주요 결과물 |
|---|---|---|
| 1단계 | SQS Queue 생성 | 영상 분석 요청 Queue |
| 2단계 | 영상 등록 API 연동 | SQS 메시지 생성 |
| 3단계 | Video Analysis Agent 연동 | 메시지 수신 및 Workflow 실행 |
| 4단계 | YOLO Detection | 연기 탐지 결과 |
| 5단계 | Segmentation | 연기 영역 Mask |
| 6단계 | Temporal Analysis | 시간별 연기 분석 |
| 7단계 | AI Inference Layer | 최종 판정 엔진 |
| 8단계 | SNS Topic 생성 | 판정 이벤트 발행 |
| 9단계 | SNS + SQS Fan-out | 여러 Consumer 연결 |
| 10단계 | Dashboard / Slack 연동 | 후속 작업 처리 |
| 11단계 | DLQ 및 재처리 | 장애 대응 |
| 12단계 | CloudWatch 모니터링 | 메시지 처리 상태 관찰 |

---

## 9. 구현 시 반드시 확인할 사항

### SQS 관련

- 메시지 크기 제한을 고려하여 영상 파일 자체를 메시지에 넣지 않는다.
- Worker가 접근할 수 있는 영상 경로를 전달한다.
- Visibility Timeout을 영상 분석 예상 시간에 맞게 설정한다.
- 중복 메시지 처리를 고려한다.
- 실패 메시지에 대한 DLQ를 구성한다.

### SNS 관련

- 최종 판정 이벤트를 SNS Topic으로 발행한다.
- 각 후속 시스템에 독립적인 SQS Queue를 연결한다.
- 구독별 Message Filtering을 활용할 수 있다.
- SNS Topic과 SQS Queue 간 IAM 권한을 설정한다.
- 이벤트 중복 수신을 고려하여 Consumer 멱등성을 확보한다.

### AI Inference Layer 관련

- 모델별 분석 결과와 최종 판정 결과를 분리한다.
- 판정 기준을 코드 및 설정으로 명확하게 관리한다.
- 판정 결과에 Reason Code를 포함한다.
- 분석 실패와 정상 판정을 구분한다.
- 판정 결과를 영구 저장한 뒤 이벤트를 발행한다.

---

## 10. 최종 정리

이번 프로젝트에서 AWS SQS와 SNS는 서로 보완적인 역할을 담당한다.

| 구성 요소 | 핵심 책임 |
|---|---|
| SQS | 영상 분석 작업 전달 및 대기열 관리 |
| Video Analysis Agent | 영상 분석 Workflow 실행 |
| YOLO | 연기 객체 탐지 |
| Segmentation | 연기 영역 추출 |
| Temporal Analysis | 시간에 따른 연기 변화 분석 |
| AI Inference Layer | 분석 결과 종합 및 최종 판정 |
| SNS | 최종 판정 이벤트 발행 및 전파 |
| 후속 SQS Queue | 각 시스템의 독립적인 작업 처리 |
| Dashboard / Slack / Observability | 결과 표시, 알림 및 실행 이력 관리 |

### 최종 아키텍처 설계 원칙

1. 영상 분석 요청은 SQS를 통해 비동기 처리한다.
2. Video Analysis Agent는 영상 분석 Workflow를 관리한다.
3. YOLO, Segmentation, Temporal Analysis는 각각의 분석 책임을 수행한다.
4. AI Inference Layer는 분석 결과를 종합하여 최종 판정을 통제한다.
5. SNS는 최종 판정 이벤트를 여러 시스템에 전파한다.
6. 각 후속 시스템은 독립적인 SQS Queue를 통해 작업을 처리한다.
7. DLQ와 CloudWatch를 활용하여 장애 및 처리 상태를 관리한다.

**핵심 메시지**

> SQS로 Agent의 작업 실행을 관리하고, SNS로 시스템 간 이벤트를 전파한다. AI Inference Layer에서 최종 판정을 통제하는 Event-Driven Multi-Agent 아키텍처를 구현한다.
