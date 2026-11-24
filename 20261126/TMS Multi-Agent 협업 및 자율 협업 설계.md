# TMS Multi-Agent 협업 및 자율 협업 설계

## 1. 개요

TMS Multi-Agent 시스템은 여러 전문 Agent가 각자의 업무를 수행하고, 분석 결과를 공유하면서 협력하는 구조입니다.

Agent 협업은 크게 두 가지 방식으로 구분합니다.

- **Agent 간 협업(Agent Collaboration):** 사전에 정의된 업무 흐름에 따라 여러 Agent가 협력하는 방식
- **자율 협업(Autonomous Agent Collaboration):** Agent가 작업 결과를 평가하고, 추가 업무의 필요성을 판단하여 다른 Agent와 동적으로 협력하는 방식

TMS에서는 일반 협업을 기본 구조로 구축하고, 이후 자율 협업 기능을 단계적으로 확장하는 것을 권장합니다.

---

## 2. Agent 간 협업 (Agent Collaboration)

### 2.1 개념

각 Agent가 독립적인 전문 업무를 수행하고, 작업 결과를 서로 전달하면서 협력하는 방식입니다.

Manager Agent가 전체 업무를 분배하고, 전문 Agent들이 담당 업무를 수행한 후 결과를 전달합니다.

### 2.2 기본 아키텍처

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

### 2.3 협업 방식

| 방식 | 설명 | TMS 적용 사례 |
|---|---|---|
| Sequential | Agent가 순차적으로 실행 | Data → Inference → QA |
| Parallel | 여러 Agent가 동시에 실행 | 데이터 분석 + 영상 분석 |
| Hierarchical | 상위 Agent가 하위 Agent를 관리 | Manager → 전문 Agent |

### 2.4 TMS 적용 사례

운영자가 다음과 같이 요청합니다.

> A공장의 최근 7일 NOx 이상 현상을 분석해 줘.

1. Manager Agent가 분석 업무를 분해합니다.
2. Data Agent가 NOx 측정 데이터와 데이터 품질을 확인합니다.
3. Inference Agent가 ML 및 Rule Engine 결과를 확인합니다.
4. QA Agent가 분석 결과를 검증합니다.
5. Report Agent가 최종 보고서를 작성합니다.

각 Agent는 구조화된 결과를 전달하고 Manager Agent가 전체 결과를 통합합니다.

---

## 3. 자율 협업 (Autonomous Agent Collaboration)

### 3.1 개념

자율 협업은 Agent가 자신의 작업 결과를 평가하고, 추가로 필요한 업무를 판단하여 다른 Agent에게 작업을 요청하는 방식입니다.

일반 협업이 미리 정해진 업무 흐름을 따르는 것이라면, 자율 협업은 **실행 결과에 따라 다음 업무와 협업 대상을 동적으로 결정**한다는 차이가 있습니다.

### 3.2 TMS 적용 사례

영상에서 연기가 감지되었지만 Detection 신뢰도가 낮고, 같은 시간대 NOx 측정값도 상승했다고 가정합니다.

```text
[Video Agent]
     |
     v
연기 Detection 결과 확인
     |
     v
추가 분석 필요 판단
     |
     v
[Manager Agent]
     |
     +--------------------------+
     |                          |
     v                          v
[Data Agent]              [Inference Agent]
NOx 변화 분석              ML 및 Rule 결과 재확인
     |                          |
     +------------+-------------+
                  |
                  v
              [QA Agent]
             교차 검증
                  |
                  v
            [Report Agent]
             결과 보고
```

### 3.3 자율 협업의 핵심 기능

| 기능 | 설명 |
|---|---|
| Dynamic Routing | 상황에 따라 적합한 Agent 선택 |
| Task Delegation | 추가 업무 생성 및 할당 |
| Agent Reassignment | 작업 실패 시 다른 Agent로 재할당 |
| Self-Evaluation | 자신의 작업 결과 평가 |
| Feedback Loop | 결과가 부족하면 추가 분석 수행 |

자율 협업에서는 최초 계획에 없던 추가 업무도 생성할 수 있습니다.

단, 최종 이상 판정은 LLM이 단독으로 결정하지 않고 기존 Rule Engine과 ML 기반 판정 엔진이 담당해야 합니다.

---

## 4. 일반 협업과 자율 협업 비교

| 구분 | 일반 협업 | 자율 협업 |
|---|---|---|
| 업무 분배 | 사전에 정의 | 상황에 따라 변경 |
| 실행 순서 | 정해진 흐름 | 결과에 따라 변경 |
| Agent 호출 | Manager 중심 | 추가 호출 및 재할당 |
| 오류 대응 | 정해진 재시도 | 상황별 대응 경로 선택 |
| 업무 확장 | 사전 정의된 범위 | 새로운 Task 생성 |
| 구현 난이도 | 상대적으로 낮음 | 상대적으로 높음 |

---

## 5. TMS 적용 기술

| 기술 | 주요 역할 |
|---|---|
| LangGraph Supervisor | Manager 및 전문 Agent 관리 |
| Conditional Edge | 결과에 따른 다음 Agent 선택 |
| Send API | 동적 Agent 호출 및 병렬 실행 |
| Checkpointer | 작업 상태 저장 및 재개 |
| AWS SNS/SQS | 비동기 작업 전달 |
| PostgreSQL | Task 상태 및 실행 이력 관리 |
| MCP | 외부 시스템 및 도구 연동 |

LangGraph의 Supervisor, Conditional Routing, Send API를 활용하면 일반 협업과 자율 협업을 모두 구현할 수 있습니다.

---

## 6. 최종 추천: Hybrid Multi-Agent Collaboration

TMS에는 일반 협업과 자율 협업을 결합한 하이브리드 구조를 추천합니다.

```text
                 [Manager Agent]
                        |
                        v
              [1단계: 일반 협업]
              Workflow-based Collaboration
                        |
              +---------+---------+
              |                   |
              v                   v
         [정형 업무 분배]     [병렬 업무 실행]
              |                   |
              +---------+---------+
                        |
                        v
              [2단계: 자율 협업]
             Autonomous Collaboration
                        |
              +---------+---------+
              |         |         |
              v         v         v
          추가 분석   Agent 선택  업무 재할당
              |         |         |
              +---------+---------+
                        |
                        v
                [QA 및 검증]
                        |
                        v
              [Human-in-the-Loop]
                        |
                        v
                 [최종 결과]
```

### 6.1 단계별 개발 우선순위

1. Manager-Specialist 기반 Agent 협업 구조 구축
2. Sequential / Parallel 협업 구현
3. Agent 간 구조화된 데이터 전달
4. Conditional Routing 구현
5. 분석 결과에 따른 Agent 재호출
6. 동적 Agent 선택 및 업무 재할당

### 6.2 자율 협업 통제 원칙

자율 협업을 구현할 때는 다음과 같은 통제 정책이 필요합니다.

- Agent의 최대 반복 횟수 제한
- 작업별 실행 시간 제한
- 등록된 Agent만 호출하도록 제한
- Agent별 데이터 및 도구 접근 권한 관리
- API 호출 및 Token 사용량 관리
- 중요 운영 조치는 Human-in-the-Loop 승인 적용
- 최종 이상 판정은 Rule Engine과 ML 기반 판정 엔진에서 수행

---

## 7. 핵심 정리

일반적인 Multi-Agent는 여러 AI가 각자 맡은 업무를 수행하는 구조입니다.

자율 협업은 여기서 한 단계 발전하여, AI가 작업 결과를 평가하고 다음에 필요한 업무를 스스로 결정하는 구조입니다.

TMS에서는 먼저 정형화된 협업을 구현하고, 이후 자율 협업 기능을 추가하는 것이 현실적인 접근입니다.

**핵심 설계 원칙**

> Agent에게는 업무 수행의 자율성을 부여하되, 최종 판정과 중요한 운영 조치는 반드시 검증하고 통제해야 합니다.
