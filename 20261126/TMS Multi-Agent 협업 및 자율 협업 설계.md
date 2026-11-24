# TMS Multi-Agent 협업 및 자율 협업 설계

## 1. 개요

TMS Multi-Agent 시스템은 여러 전문 Agent가 각자의 업무를 수행하고, 분석 결과를 공유하면서 협력하는 구조입니다.

Agent 협업 방식은 업무를 누가 결정하고, 어떻게 분배하며, 실행 과정에서 얼마나 자율적으로 판단하는지에 따라 다음 세 가지로 구분할 수 있습니다.

1. **위임형 협업(Delegation-based Collaboration)**
2. **자율형 협업(Autonomous Collaboration)**
3. **하이브리드 협업(Hybrid Collaboration)**

TMS에서는 위임형 협업을 기본으로 구축하고, 자율형 협업을 단계적으로 추가하여 최종적으로 하이브리드 Multi-Agent 구조를 구현하는 것을 권장합니다.

---

## 2. 위임형 협업 (Delegation-based Collaboration)

### 2.1 개념

Manager Agent가 전체 업무를 분석하고 세부 작업으로 분해한 후, 적합한 전문 Agent에게 업무를 위임하는 방식입니다.

각 전문 Agent는 위임받은 업무를 수행하고 결과를 Manager Agent에게 전달합니다.

주요 특징은 다음과 같습니다.

- Manager Agent가 전체 업무와 실행 순서를 결정합니다.
- 전문 Agent는 위임받은 업무를 수행합니다.
- 사전에 정의된 Workflow에 따라 실행합니다.
- 작업 결과를 Manager Agent가 통합합니다.

### 2.2 아키텍처

```text
                  [Manager Agent]
                  업무 분해 및 할당
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
     [Data Agent]  [Inference Agent] [Video Agent]
     데이터 수집     ML 추론 및 분석    영상 분석
          |              |              |
          +--------------+--------------+
                         |
                         v
                    [QA Agent]
                    결과 검증
                         |
                         v
                  [Report Agent]
                   보고서 작성
```

### 2.3 TMS 적용 사례

운영자가 다음과 같이 요청합니다.

> A공장의 최근 7일 NOx 이상 현상을 분석해 줘.

1. Manager Agent가 분석 업무를 분해합니다.
2. Data Agent가 NOx 측정 데이터와 품질을 확인합니다.
3. Inference Agent가 ML 및 Rule Engine 결과를 확인합니다.
4. QA Agent가 분석 결과를 검증합니다.
5. Report Agent가 최종 보고서를 작성합니다.

이 방식은 실행 흐름이 명확하고 결과를 추적하기 쉬워 TMS의 기본 협업 구조로 적합합니다.

---

## 3. 자율형 협업 (Autonomous Collaboration)

### 3.1 개념

각 Agent가 자신의 작업 결과를 평가하고, 추가 작업이 필요하다고 판단하면 다른 Agent와 동적으로 협력하는 방식입니다.

위임형 협업과 달리 최초에 정의된 업무 흐름에만 의존하지 않습니다.

주요 특징은 다음과 같습니다.

- Agent가 추가 분석 필요성을 판단합니다.
- 새로운 Task를 생성하거나 Manager Agent에 추가 업무를 요청합니다.
- 상황에 따라 다른 전문 Agent를 선택합니다.
- 작업 실패 시 다른 Agent로 재할당할 수 있습니다.
- 실행 결과에 따라 협업 흐름이 변경됩니다.

### 3.2 아키텍처

```text
                 [Manager Agent]
                        |
                        v
                  [Video Agent]
                        |
                        v
                 연기 Detection
                        |
                        v
                  결과 자체 평가
                        |
               추가 분석이 필요한가?
                        |
                        v
                 [Manager Agent]
                        |
              추가 Task 생성 및 할당
                        |
          +-------------+-------------+
          |                           |
          v                           v
     [Data Agent]              [Inference Agent]
     NOx 변화 분석              ML 및 Rule 재확인
          |                           |
          +-------------+-------------+
                        |
                        v
                    [QA Agent]
                    교차 검증
                        |
                        v
                  [Report Agent]
                   결과 보고
```

### 3.3 TMS 적용 사례

영상에서 연기가 감지되었지만 Detection 신뢰도가 낮고, 같은 시간대 NOx 측정값도 상승했다고 가정합니다.

1. Video Agent가 연기 Detection 결과를 확인합니다.
2. Video Agent가 추가 분석이 필요하다고 판단합니다.
3. Manager Agent에 추가 분석을 요청합니다.
4. Manager Agent가 Data Agent와 Inference Agent에 새로운 Task를 할당합니다.
5. Data Agent가 동일 시간대 NOx 변화 추이를 분석합니다.
6. Inference Agent가 ML 및 Rule Engine 결과를 재확인합니다.
7. QA Agent가 영상 분석과 측정 데이터 결과를 교차 검증합니다.
8. Report Agent가 통합 분석 결과를 작성합니다.

이처럼 최초 계획에 없던 추가 업무를 생성하고, 분석 결과에 따라 협업 대상을 변경할 수 있습니다.

### 3.4 자율 협업의 핵심 기능

| 기능 | 설명 |
|---|---|
| Dynamic Routing | 상황에 따라 적합한 Agent 선택 |
| Task Delegation | 추가 업무 생성 및 할당 |
| Agent Reassignment | 작업 실패 시 다른 Agent로 재할당 |
| Self-Evaluation | 자신의 작업 결과 평가 |
| Feedback Loop | 결과가 부족하면 추가 분석 수행 |

---

## 4. 하이브리드 협업 (Hybrid Collaboration)

### 4.1 개념

위임형 협업을 기본으로 하면서 필요한 경우 자율형 협업을 수행하는 방식입니다.

일반적인 업무는 정해진 Workflow에 따라 실행하고, 분석 결과가 부족하거나 추가 검증이 필요한 경우 Agent가 동적으로 협업합니다.

주요 특징은 다음과 같습니다.

- 기본 업무는 Manager Agent가 계획하고 위임합니다.
- 전문 Agent는 정해진 업무를 수행합니다.
- 추가 분석이 필요하면 새로운 Task를 생성합니다.
- Manager Agent가 추가 협업 범위와 실행을 관리합니다.
- 최종 결과는 QA 및 판정 엔진을 통해 검증합니다.

### 4.2 아키텍처

```text
                 [Manager Agent]
                        |
                        v
                  업무 계획 수립
                        |
                        v
                 [위임형 협업]
                        |
              +---------+---------+
              |         |         |
              v         v         v
           Data    Inference    Video
           Agent     Agent     Agent
              |         |         |
              +---------+---------+
                        |
                        v
                    결과 평가
                        |
                 추가 분석이 필요한가?
                   /          \
                 아니오         예
                  |             |
                  |             v
                  |       [자율형 협업]
                  |             |
                  |       추가 Task 생성
                  |             |
                  |       Agent 동적 호출
                  |             |
                  |       추가 분석 수행
                  |             |
                  +------+------+
                         |
                         v
                      [QA Agent]
                         |
                         v
                [Rule Engine + ML]
                         |
                         v
                     최종 판정
```

### 4.3 TMS 적용 사례

- 일반적인 NOx 이상 분석은 위임형 협업으로 처리합니다.
- 영상에서 연기가 감지되거나 측정 데이터와 ML 결과가 일치하지 않으면 추가 분석을 수행합니다.
- Manager Agent가 필요한 전문 Agent를 동적으로 호출합니다.
- QA Agent가 추가 분석 결과를 검증합니다.
- Rule Engine과 ML 기반 판정 엔진이 최종 이상 여부를 판단합니다.

TMS에서는 이 방식이 가장 현실적인 Multi-Agent 운영 구조입니다.

---

## 5. 협업 유형 비교

| 구분 | 위임형 협업 | 자율형 협업 | 하이브리드 협업 |
|---|---|---|---|
| 업무 결정 | Manager 중심 | Agent의 동적 판단 | Manager + Agent |
| 업무 분배 | 사전에 정의 | 상황에 따라 변경 | 기본 위임 + 동적 변경 |
| 실행 흐름 | 정형 Workflow | 동적 Workflow | 정형 + 동적 |
| Agent 호출 | Manager 중심 | 추가 호출 및 재할당 | 두 방식 결합 |
| 오류 대응 | 정해진 재시도 | 상황별 대응 경로 선택 | 정책 기반 대응 |
| 통제 수준 | 높음 | 상대적으로 낮음 | 정책 기반 통제 |
| 구현 난이도 | 낮음 | 높음 | 중간~높음 |

---

## 6. TMS 적용 기술

| 기술 | 주요 역할 |
|---|---|
| LangGraph Supervisor | Manager 및 전문 Agent 관리 |
| Conditional Edge | 실행 결과에 따른 다음 Agent 선택 |
| Send API | 동적 Agent 호출 및 병렬 실행 |
| Checkpointer | 작업 상태 저장 및 재개 |
| AWS SNS/SQS | 비동기 작업 전달 및 분배 |
| PostgreSQL | Task 상태 및 실행 이력 관리 |
| MCP | 외부 시스템 및 도구 연동 |

특히 LangGraph의 Supervisor, Conditional Routing, Send API를 활용하면 위임형과 자율형 협업을 구현할 수 있습니다.

---

## 7. 단계별 개발 방향

| 단계 | 협업 방식 | 주요 개발 내용 |
|---|---|---|
| 1단계 | 위임형 | Manager 및 전문 Agent 구성 |
| 2단계 | 위임형 확장 | Sequential / Parallel 실행 |
| 3단계 | 자율형 | 추가 Task 생성 및 Agent 동적 호출 |
| 4단계 | 자율형 확장 | Agent 재할당 및 결과 기반 재분석 |
| 5단계 | 하이브리드 | 위임형과 자율형 통합 운영 |

### 7.1 자율 협업 통제 원칙

자율 협업은 Agent가 자유롭게 서로를 호출하는 구조가 아닙니다. 안정적인 운영을 위해 다음과 같은 통제 정책이 필요합니다.

- Agent의 최대 반복 횟수 제한
- Task별 실행 시간 제한
- 등록된 Agent만 호출하도록 제한
- Agent별 데이터 및 도구 접근 권한 관리
- API 호출 및 Token 사용량 관리
- 오류 유형별 재시도 정책 적용
- 중요 운영 조치에 Human-in-the-Loop 승인 적용
- 최종 이상 판정은 Rule Engine과 ML 기반 판정 엔진에서 수행

---

## 8. 핵심 정리

Multi-Agent 협업은 다음과 같이 이해하면 됩니다.

| 협업 방식 | 핵심 개념 |
|---|---|
| 위임형 | Manager가 업무를 나누고 Agent가 수행한다. |
| 자율형 | Agent가 작업 결과를 평가하고 필요한 업무를 찾아 협력한다. |
| 하이브리드 | 기본적으로 Manager가 관리하되 필요한 경우 Agent가 자율적으로 협력한다. |

**TMS 최종 권장 방향은 하이브리드 Multi-Agent 구조입니다.**

처음부터 복잡한 자율형 협업을 구현하기보다는 위임형 협업으로 안정적인 업무 흐름을 구축하고, 이후 추가 분석과 동적 Agent 호출 기능을 확장하는 것이 좋습니다.

> **핵심 설계 원칙: Agent에게는 업무 수행의 자율성을 부여하되, 최종 판정과 중요한 운영 조치는 반드시 검증하고 통제해야 합니다.**
