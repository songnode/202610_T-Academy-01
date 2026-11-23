# TMS AI Inference Layer 구현 가이드
## ML Model + Rule Engine 기반 Hybrid Decision

## 1. AI Inference Layer란?

### 1.1 개념

AI Inference Layer는 학습이 완료된 AI 모델과 사전에 정의한 규칙을 이용하여 새로운 데이터의 위험도를 판단하는 영역이다.

쉽게 말하면, **수집된 TMS 데이터를 분석하여 현재 굴뚝의 배출 상태가 정상인지, 주의가 필요한지, 경보를 발생시켜야 하는지를 결정하는 영역**이다.

TMS 프로젝트에서는 ML Model 하나만으로 최종 경보를 결정하지 않는다.

ML Model과 Rule Engine을 함께 사용하고, 두 결과를 Hybrid Decision에서 통합하는 구조를 적용한다.

예를 들어 다음과 같은 상황을 생각해 보자.

- 현재 NOx 기준대비율은 77%이다.
- 최근 30분 동안 농도가 12%p 상승했다.
- 최근 측정값의 변동성도 높다.
- ML Model은 미래 초과 가능성을 낮게 예측했다.

ML Model만 사용하면 정상으로 판단할 수 있다.

하지만 Rule Engine은 농도가 빠르게 상승하고 있다는 사실을 감지할 수 있다.

이 두 결과를 함께 고려하면 ML Model이 놓칠 수 있는 급상승 위험을 보완할 수 있다.

### 1.2 AI Inference Layer의 전체 구조

```text
                 AI-Ready Data
                       |
                       v
                 Feature Service
                       |
                       v
                AI Inference Layer
                       |
             +---------+---------+
             |                   |
             v                   v
         ML Inference        Rule Engine
             |                   |
             v                   v
         ML Prediction       Rule Result
             |                   |
             +---------+---------+
                       |
                       v
                 Hybrid Decision
                       |
                       v
                  Final Decision
                       |
                       v
                  Alert Manager
                       |
                       v
                Alert / History
                       |
                       v
                Agent Observability
```

각 구성 요소의 역할은 다음과 같다.

| 구성 요소 | 주요 역할 |
|---|---|
| Feature Service | 원천 데이터를 모델과 규칙이 사용할 수 있는 Feature로 변환 |
| ML Inference | 학습된 모델을 이용하여 미래 초과 가능성 예측 |
| Rule Engine | 명시적인 조건으로 현재 위험 신호와 예외 상황 탐지 |
| Hybrid Decision | ML과 Rule 결과를 결합하여 최종 판정 |
| Alert Manager | 경보 중복 방지, 상태 관리, 경보 발생 및 해제 |
| Decision History | 판정 결과와 판단 근거 저장 |
| Agent Observability | 모델, 규칙, 최종 판정 및 경보 이력 추적 |

핵심은 ML Model과 Rule Engine이 각각 판단한 후, Hybrid Decision에서 최종 결과를 결정한다는 것이다.

---

## 2. 먼저 이해해야 할 판정 상태값

AI Inference Layer를 구현하기 전에 가장 먼저 통일해야 할 것이 판정 상태값이다.

각 모듈에서 서로 다른 상태값을 사용하면 최종 판정과 경보 이력을 연결하기 어렵다.

따라서 프로젝트 전체에서 공통으로 사용할 상태값을 정의한다.

### 2.1 최종 판정 상태값

TMS 프로젝트에서는 다음 6개 상태값을 최종 판정의 표준으로 사용한다.

| 상태값 | 한글 명칭 | 의미 |
|---|---|---|
| `NORMAL` | 정상 | 확인된 위험 신호가 없음 |
| `CAUTION` | 주의 | 일부 위험 신호가 확인되어 지속적인 관찰 필요 |
| `WARNING` | 경고 | 검증된 경보 조건을 충족하여 경보 발생 필요 |
| `WARNING_CANDIDATE` | 경고 후보 | 위험 신호는 발견했으나 운영 경보로 확정하기 전 검증 필요 |
| `EXCEEDED` | 현재 기준 초과 | 현재 기준대비율이 90% 이상인 상태 |
| `UNKNOWN` | 판단 보류 | 데이터 품질 문제 등으로 신뢰할 수 있는 판정 불가 |

각 상태값의 차이를 정확하게 이해해야 한다.

#### NORMAL

ML Model과 Rule Engine에서 특별한 위험 신호를 발견하지 못한 상태이다.

정상이라고 해서 미래에 초과가 발생하지 않는다는 의미는 아니다.

#### CAUTION

현재 농도나 최근 상승 추세에서 주의할 만한 신호가 확인된 상태이다.

아직 운영 경보를 발생시킬 수준은 아니지만 지속적인 관찰이 필요하다.

#### WARNING

검증된 ML 예측 또는 운영 정책상 승인된 Rule 조건을 충족한 상태이다.

실제 경보 발생 대상으로 전달한다.

#### WARNING_CANDIDATE

새롭게 추가한 규칙이 위험 신호를 발견했지만 아직 운영 경보로 확정하지 않은 상태이다.

예를 들어 저농도 급상승 Rule을 개발한 초기에는 이 상태로 결과를 기록하고 Offline Replay를 통해 효과를 검증한다.

검증 결과가 충분하면 해당 Rule을 정식 경보 정책으로 승격할 수 있다.

#### EXCEEDED

현재 측정값이 기준대비율 90% 이상인 상태이다.

미래 초과 가능성을 예측한 결과가 아니라 현재 측정값이 해당 기준에 도달했음을 의미한다.

이 문서에서 90%는 프로젝트의 분석 및 경보 정책에서 사용하는 기준이다. 법적 배출허용기준 자체와 혼동하지 않아야 한다.

#### UNKNOWN

데이터가 누락되었거나 시간 간격이 맞지 않는 등 판단에 필요한 조건이 충족되지 않은 상태이다.

`UNKNOWN`을 `NORMAL`로 처리하면 안 된다.

데이터 품질 문제를 별도로 관리해야 한다.

### 2.2 판정 상태와 경보 운영 상태의 구분

여기서 중요한 점은 AI가 판단한 결과와 실제 경보가 운영되는 상태는 서로 다르다는 것이다.

예를 들어 AI가 `WARNING`을 반환했다고 해서 매번 새로운 경보를 발생시키는 것은 아니다.

이미 해당 굴뚝에 경보가 발생한 상태라면 중복 경보를 방지해야 한다.

따라서 다음과 같이 구분한다.

| 구분 | 관리 대상 | 상태값 |
|---|---|---|
| Decision Status | AI의 최종 판정 결과 | `NORMAL`, `CAUTION`, `WARNING`, `WARNING_CANDIDATE`, `EXCEEDED`, `UNKNOWN` |
| Alert Status | 실제 경보의 운영 상태 | `NORMAL`, `CAUTION`, `WARNING`, `EXCEEDED`, `RECOVERY` |

`RECOVERY`는 AI의 최종 판정 상태가 아니다.

Alert Manager에서 기존 경보가 해제되는 과정에 사용하는 운영 상태이다.

### 2.3 단계별 상태값 사용 기준

| 구성 요소 | 사용 상태값 | 설명 |
|---|---|---|
| ML Inference | `NORMAL`, `WARNING` | 모델 예측 결과 |
| Rule Engine | `NORMAL`, `CAUTION`, `WARNING_CANDIDATE`, `EXCEEDED`, `UNKNOWN` | 규칙 평가 결과 |
| Hybrid Decision | 최종 판정 6개 상태 | ML과 Rule 결과 통합 |
| Alert Manager | `NORMAL`, `CAUTION`, `WARNING`, `EXCEEDED`, `RECOVERY` | 경보 상태 관리 |

ML Inference의 `WARNING`은 모델의 예측 점수가 설정된 임계값을 충족했다는 의미이다.

Rule Engine의 `WARNING_CANDIDATE`는 새롭게 실험 중인 규칙이 위험 신호를 발견했다는 의미이다.

두 결과를 구분해야 최종 판정의 근거를 추적할 수 있다.

---

## 3. ML Inference: AI 모델을 이용한 위험 예측

### 3.1 ML Inference란?

ML Inference는 학습이 완료된 모델에 새로운 데이터를 입력하여 예측 결과를 생성하는 과정이다.

머신러닝 프로젝트는 크게 두 단계로 나뉜다.

| 구분 | Training | Inference |
|---|---|---|
| 목적 | 모델 학습 | 새로운 데이터 예측 |
| 입력 | 학습 데이터와 정답(Label) | 새로운 Feature 데이터 |
| 처리 | 모델 파라미터 학습 | 학습된 모델로 예측 |
| 결과 | 학습된 모델 파일 | 예측 점수 및 위험도 |
| 수행 시점 | 모델 개발 및 재학습 시 | 실제 데이터가 들어올 때 |

예를 들어 TMS 프로젝트에서 XGBoost 모델을 학습했다면 학습이 완료된 모델을 저장한다.

이후 새로운 NOx 측정 데이터가 들어올 때마다 저장된 모델을 호출하여 미래 초과 가능성을 예측한다.

### 3.2 TMS에서 ML Model이 예측하는 것

우리 프로젝트의 ML Model은 현재 농도 자체를 판단하는 것과 미래 기준 초과 가능성을 예측하는 것을 구분해야 한다.

이번 Hybrid Decision에서 ML Model이 수행하는 역할은 다음과 같다.

> 현재까지의 측정 이력을 바탕으로 향후 30분 또는 60분 내 배출 기준 초과 가능성을 예측한다.

예를 들어 다음과 같은 Feature를 사용할 수 있다.

| 입력 Feature | 값 |
|---|---:|
| 현재 기준대비율 | 77% |
| 최근 6슬롯 최댓값 | 79% |
| 최근 6슬롯 표준편차 | 9.5 |
| 최근 30분 상승폭 | 12%p |
| 과거 6슬롯 90% 이상 관측 여부 | False |

위 Feature를 학습된 ML Model에 입력하면 다음과 같은 결과가 나올 수 있다.

```python
{
    "ml_score": 0.08,
    "ml_risk_level": "NORMAL"
}
```

여기서 `ml_score`는 모델이 산출한 예측 점수이다.

이 점수가 실제 발생 확률 8%를 의미하는지는 별도의 확률 보정과 검증이 필요하다.

또한 학습 Label이 30분 초과인지, 60분 초과인지에 따라 예측 결과의 의미도 달라진다.

### 3.3 ML Inference 구현 예시

```python
import joblib
import pandas as pd


class TMSMLInference:

    def __init__(self, model_path, threshold=0.5):
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def predict(self, features):

        X = pd.DataFrame([features])

        score = float(
            self.model.predict_proba(X)[0][1]
        )

        risk_level = (
            "WARNING"
            if score >= self.threshold
            else "NORMAL"
        )

        return {
            "ml_score": score,
            "ml_risk_level": risk_level,
            "inference_status": "SUCCESS"
        }
```

위 코드는 ML Inference의 기본 구조를 설명하기 위한 예시이다.

실제 프로젝트에서는 다음 사항을 확인해야 한다.

- 학습 당시 Feature 이름과 순서 일치
- 학습 당시 전처리 과정 재현
- 결측값 처리 방식 일치
- 모델 버전 관리
- 예측 점수 임계값 검증
- 30분 및 60분 예측 목표 구분
- 모델 호출 실패 및 예외 처리

특히 `0.5`는 설명을 위한 예시일 뿐 실제 운영 임계값으로 확정된 값이 아니다.

ML 예측 결과를 생성할 수 없는 경우에는 `inference_status`를 `FAILED`로 기록하고, 이를 정상 예측으로 처리해서는 안 된다.

---

## 4. Rule Engine: 명시적인 규칙으로 위험 신호 탐지

### 4.1 Rule Engine이란?

Rule Engine은 사람이 사전에 정의한 조건을 기준으로 데이터를 판단하는 시스템이다.

ML Model이 데이터에서 학습한 패턴을 이용한다면 Rule Engine은 개발자가 정의한 명확한 조건을 이용한다.

예를 들어 다음과 같은 규칙을 생각할 수 있다.

- 현재 농도가 기준 대비 90% 이상이면 현재 기준 초과 상태로 판단한다.
- 현재 농도가 70~80%이고 최근 상승폭이 크면 주의 대상으로 판단한다.
- 최근 측정값의 변동성이 높으면 급격한 변화 가능성을 확인한다.
- 데이터가 결측되거나 시간 간격이 맞지 않으면 판단을 보류한다.

Rule Engine은 ML Model이 놓칠 수 있는 특정 위험 상황을 보완하는 역할을 한다.

### 4.2 TMS 프로젝트에서 적용할 Rule 유형

#### Rule A. 현재 농도 구간 판단

현재 기준대비율에 따라 위험 구간을 구분한다.

| 현재 기준대비율 | 구분 | 의미 |
|---|---|---|
| 70% 미만 | LOW | 상대적으로 낮은 농도 |
| 70~80% | WATCH | 상승 추세 확인 필요 |
| 80~90% | CAUTION | 기준 초과에 가까운 구간 |
| 90% 이상 | EXCEEDED | 현재 기준 초과 상태 |

위 구간은 현재 진단에서 사용한 분석 구간을 바탕으로 한 예시이다.

실제 운영 임계값은 대만 Train/Validation 데이터와 국내 독립 평가를 통해 검증해야 한다.

#### Rule B. 저농도 급상승 탐지

현재 TMS 진단에서 중요한 문제 중 하나는 저농도 상태에서 갑자기 농도가 상승하는 사건이다.

기존 ML Model은 높은 농도나 과거 초과 경험이 있는 상황에서는 상대적으로 강했지만, 낮은 농도에서 급격하게 상승하는 일부 사건에서는 탐지율이 낮았다.

이를 보완하기 위해 상승폭과 변동성을 활용하는 Rule을 실험할 수 있다.

| Feature | 설명 |
|---|---|
| `current_ratio` | 현재 기준대비율 |
| `delta_30m` | 최근 30분 상승폭 |
| `std_6slot` | 최근 6개 측정값의 표준편차 |
| `recent_max_6slot` | 최근 6개 측정값의 최댓값 |

예를 들어 다음과 같은 조건을 검토할 수 있다.

**상승폭 Rule**

```text
IF
    현재 기준대비율이 70~80%이고
    최근 30분 상승폭이 크면
THEN
    급상승 주의 후보
```

**상승폭 + 변동성 Rule**

```text
IF
    현재 기준대비율이 70~80%이고
    최근 30분 상승폭이 크고
    최근 6개 측정값의 변동성도 높으면
THEN
    급상승 경고 후보
```

상승폭만 사용하는 규칙과 상승폭 및 변동성을 함께 사용하는 규칙은 별도로 실험해야 한다.

어떤 규칙이 더 효과적인지는 실제 데이터로 비교해야 한다.

#### Rule C. 과거 초과 이력 확인

과거 6개 측정 슬롯 중 90% 이상 관측값이 있었는지 확인한다.

| 유형 | 정의 |
|---|---|
| A형 | 기준 시점 이전 과거 6개 슬롯에 90% 이상 관측값이 한 번도 없음 |
| B형 | 기준 시점 이전 과거 6개 슬롯 중 한 번 이상 90% 이상 관측값이 있음 |

이 정보는 과거 초과 이력을 구분하는 Feature로 활용할 수 있다.

다만 B형이라고 해서 무조건 경보를 발생시키거나 A형이라고 해서 위험이 낮다고 판단해서는 안 된다.

과거 초과 이력은 여러 Feature 중 하나로 사용해야 한다.

#### Rule D. 데이터 품질 확인

데이터 품질은 모든 위험 판단보다 먼저 확인해야 한다.

| 검사 항목 | 확인 내용 |
|---|---|
| 결측값 | 필수 측정값 누락 여부 |
| 중복 데이터 | 동일 굴뚝, 항목, 측정 시각 중복 |
| 시간 간격 | 정상적인 측정 주기 준수 여부 |
| 비정상 값 | 허용 범위를 벗어난 값 |
| 극단값 | 200% 초과 등 극단적인 관측값 확인 |

특히 200% 초과 데이터를 무조건 센서 오류로 처리하면 안 된다.

실제 급등일 가능성도 있으므로 데이터 품질 문제와 실제 위험 신호를 구분해야 한다.

데이터 품질을 확인할 수 없는 경우에는 `UNKNOWN`으로 처리한다.

### 4.3 Rule Engine 구현 예시

```python
class TMSRuleEngine:

    def evaluate(self, features):

        rules = []
        risk_level = "NORMAL"

        # Rule D: 데이터 품질 확인
        if not features["data_quality"]:
            return {
                "rule_risk_level": "UNKNOWN",
                "rule_ids": ["DQ_001"],
                "decision_reason": "데이터 품질 확인 필요"
            }

        current = features["current_ratio"]
        delta = features["delta_30m"]
        volatility = features["std_6slot"]

        # Rule A: 현재 기준 초과 상태
        if current >= 90:
            return {
                "rule_risk_level": "EXCEEDED",
                "rule_ids": ["R_CURRENT_90"],
                "decision_reason": "현재 기준대비율 90% 이상"
            }

        # Rule B: 저농도 급상승
        if 70 <= current < 80 and delta >= 10:
            rules.append("R_RAPID_RISE")
            risk_level = "CAUTION"

        # Rule B-2: 상승폭 + 변동성
        if (
            70 <= current < 80
            and delta >= 10
            and volatility >= 8
        ):
            rules.append("R_HIGH_VOLATILITY")
            risk_level = "WARNING_CANDIDATE"

        return {
            "rule_risk_level": risk_level,
            "rule_ids": rules,
            "decision_reason": "규칙 조건 평가 완료"
        }
```

위 코드의 `delta >= 10`, `volatility >= 8`은 동작을 설명하기 위한 가상 임계값이다.

실제 TMS 운영 기준으로 사용하면 안 된다.

대만 Train/Validation 데이터를 이용해 임계값을 검토하고, 국내 데이터는 독립 평가에 활용해야 한다.

또한 실제 구현에서는 결측, 측정 간격, Feature 생성 실패, 비정상 데이터 등의 검증을 보강해야 한다.

---

## 5. Hybrid Decision: ML과 Rule 결과를 결합하는 방법

### 5.1 Hybrid Decision이란?

Hybrid Decision은 ML Model의 예측 결과와 Rule Engine의 판단 결과를 결합하여 최종 위험도를 결정하는 방식이다.

두 방식의 장점은 서로 다르다.

| 구분 | ML Model | Rule Engine |
|---|---|---|
| 판단 기준 | 데이터에서 학습한 패턴 | 사전에 정의한 조건 |
| 장점 | 복잡한 패턴 탐지 | 명확한 조건과 판단 근거 |
| 단점 | 학습 데이터에 없는 상황에 취약할 수 있음 | 규칙에 없는 상황은 탐지하기 어려움 |
| 역할 | 미래 초과 가능성 예측 | 현재 위험 신호 및 예외 상황 확인 |

두 결과를 결합하면 ML Model의 예측 성능을 활용하면서 Rule Engine으로 특정 위험 상황을 보완할 수 있다.

### 5.2 초기 Hybrid Decision 권장 방식

처음부터 ML 점수와 Rule 점수를 가중합하는 방식보다는 위험 수준의 우선순위를 비교하는 방식을 권장한다.

예를 들어 다음과 같이 판단할 수 있다.

| ML 결과 | Rule 결과 | Hybrid Decision |
|---|---|---|
| NORMAL | NORMAL | NORMAL |
| NORMAL | CAUTION | CAUTION |
| NORMAL | WARNING_CANDIDATE | WARNING_CANDIDATE |
| WARNING | NORMAL | WARNING |
| WARNING | CAUTION | WARNING |
| WARNING | WARNING_CANDIDATE | WARNING |
| 정상/경고 | EXCEEDED | EXCEEDED |
| 임의 결과 | UNKNOWN | UNKNOWN |

여기서 `WARNING_CANDIDATE`는 검증되지 않은 새로운 Rule이 발견한 위험 신호이다.

이를 바로 `WARNING`으로 승격하지 않고 Offline Replay를 통해 실제 초과 사건을 얼마나 추가로 탐지하는지, 정상 상황에서 오탐이 얼마나 발생하는지 확인해야 한다.

검증된 Rule은 별도의 정책 설정을 통해 `WARNING`으로 승격할 수 있다.

### 5.3 Hybrid Decision 판정 우선순위

최종 판정은 다음 순서로 처리한다.

1. 데이터 품질 확인
2. 현재 기준 초과 여부 확인
3. ML 및 Rule 위험 수준 결합
4. 최종 상태 결정

| 우선순위 | 조건 | 최종 상태 |
|---:|---|---|
| 1 | 데이터 품질 확인 불가 | `UNKNOWN` |
| 2 | 현재 기준대비율 90% 이상 | `EXCEEDED` |
| 3 | ML 또는 승인된 Rule에서 경보 조건 충족 | `WARNING` |
| 4 | 검증 전 Rule에서 경고 후보 조건 충족 | `WARNING_CANDIDATE` |
| 5 | 주의 조건 충족 | `CAUTION` |
| 6 | 위험 신호 없음 | `NORMAL` |

데이터 품질 검사는 현재 초과 여부보다 먼저 수행한다.

측정값을 신뢰할 수 없다면 초과 여부도 확정할 수 없기 때문이다.

단, ML Model 호출이 실패했다고 해서 정상으로 처리해서는 안 된다.

이 경우에는 ML 결과의 실패 상태를 별도로 기록하고, 시스템 정책에 따라 Rule 기반 판단만 수행할지 또는 전체 판정을 `UNKNOWN`으로 처리할지 결정해야 한다.

초기 프로젝트에서는 ML 예측 실패 시 `UNKNOWN`으로 처리하는 보수적인 정책을 권장한다.

### 5.4 Hybrid Decision 구현 예시

```python
from enum import Enum


class DecisionStatus(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    WARNING_CANDIDATE = "WARNING_CANDIDATE"
    EXCEEDED = "EXCEEDED"
    UNKNOWN = "UNKNOWN"


def hybrid_decision(
    ml_result,
    rule_result
):

    ml_level = ml_result["ml_risk_level"]
    rule_level = rule_result["rule_risk_level"]

    # 1. 데이터 품질 확인
    if rule_level == DecisionStatus.UNKNOWN.value:
        return {
            "final_decision": DecisionStatus.UNKNOWN.value,
            "decision_reason": "데이터 품질 확인 필요"
        }

    # 2. ML Inference 실행 여부 확인
    if ml_result.get("inference_status") != "SUCCESS":
        return {
            "final_decision": DecisionStatus.UNKNOWN.value,
            "decision_reason": "ML Inference 실행 실패"
        }

    # 3. 현재 기준 초과 상태
    if rule_level == DecisionStatus.EXCEEDED.value:
        return {
            "final_decision": DecisionStatus.EXCEEDED.value,
            "decision_reason": "현재 기준대비율 90% 이상"
        }

    # 4. 위험 수준 우선순위
    priority = {
        "NORMAL": 0,
        "CAUTION": 1,
        "WARNING_CANDIDATE": 2,
        "WARNING": 3
    }

    if ml_level not in priority or rule_level not in priority:
        return {
            "final_decision": DecisionStatus.UNKNOWN.value,
            "decision_reason": "정의되지 않은 판정 상태"
        }

    final_level = max(
        [ml_level, rule_level],
        key=lambda x: priority[x]
    )

    return {
        "final_decision": final_level,
        "decision_reason": (
            f"ML={ml_level}, Rule={rule_level}"
        )
    }
```

위 코드는 초기 Hybrid Decision의 기본 원리를 설명하기 위한 예시이다.

실제 구현에서는 다음 사항을 보강해야 한다.

- 승인된 Rule과 실험 중인 Rule 구분
- Rule 버전 및 임계값 관리
- ML 예측 실패 시 대체 정책
- 현재 초과 상태와 미래 위험 예측 구분
- 판정 근거 및 사용된 Rule ID 기록

특히 위험 수준을 단순히 숫자로 비교하는 방식은 초기 구현에 적합하다.

향후에는 위험 유형별 우선순위, 경보 정책, 데이터 품질 상태 등을 별도 정책으로 관리할 수 있다.

---

## 6. AI Inference Layer 전체 실행 예시

지금까지 구현한 기능을 하나의 흐름으로 연결해 보자.

### 6.1 입력 데이터

다음은 설명을 위한 가상의 NOx 측정 데이터이다.

| Feature | 값 |
|---|---:|
| 현재 기준대비율 | 77% |
| 최근 30분 상승폭 | 12%p |
| 최근 6슬롯 표준편차 | 9.5 |
| 최근 6슬롯 최댓값 | 79% |
| 과거 6슬롯 90% 이상 관측 여부 | False |
| 데이터 품질 | 정상 |

### 6.2 ML Model 실행

```python
ml_result = {
    "ml_score": 0.08,
    "ml_risk_level": "NORMAL",
    "inference_status": "SUCCESS"
}
```

ML Model은 학습된 패턴을 기준으로 위험도가 낮다고 판단했다.

### 6.3 Rule Engine 실행

```python
rule_result = {
    "rule_risk_level": "WARNING_CANDIDATE",
    "rule_ids": [
        "R_RAPID_RISE",
        "R_HIGH_VOLATILITY"
    ],
    "decision_reason": "저농도 급상승 및 변동성 확인"
}
```

Rule Engine은 현재 농도는 낮지만 최근 상승폭과 변동성이 높다는 점을 확인했다.

### 6.4 Hybrid Decision 실행

```python
final_result = {
    "final_decision": "WARNING_CANDIDATE",
    "decision_reason": (
        "ML은 정상으로 판단했으나 "
        "Rule Engine에서 급상승 위험 신호를 확인"
    )
}
```

### 6.5 결과 해석

이 사례에서 ML Model은 정상으로 판단했지만 Rule Engine은 급상승 위험 신호를 발견했다.

따라서 Hybrid Decision은 이를 `WARNING_CANDIDATE`로 분류한다.

이 결과를 바로 운영 경보로 사용하기보다는 과거 데이터 Replay를 통해 실제 초과 사건을 얼마나 추가로 탐지하는지, 정상 상황에서 오탐이 얼마나 발생하는지 확인해야 한다.

검증이 완료되면 해당 Rule을 정식 경보 정책으로 승격할 수 있다.

---

## 7. Alert Manager: 최종 판정을 실제 경보로 관리

### 7.1 Alert Manager란?

AI Inference Layer에서 최종 판정이 완료되었다고 해서 매번 새로운 경보를 발생시키면 안 된다.

예를 들어 동일한 굴뚝에서 30분마다 WARNING 결과가 발생하면 같은 내용의 경보가 반복해서 전달될 수 있다.

이를 방지하고 경보의 시작, 유지, 승격, 해제를 관리하는 것이 Alert Manager이다.

| 구분 | AI Inference Layer | Alert Manager |
|---|---|---|
| 주요 목적 | 위험도 판단 | 경보 상태 관리 |
| 입력 | Feature, ML 결과, Rule 결과 | Hybrid Decision 결과 |
| 처리 | ML + Rule + Hybrid Decision | 중복 방지, 상태 전이, 해제 |
| 결과 | 최종 판정 | 실제 경보 이벤트 |

### 7.2 Alert Manager 상태 관리

Alert Manager는 다음과 같은 운영 상태를 관리한다.

| Alert Status | 의미 |
|---|---|
| `NORMAL` | 활성 경보 없음 |
| `CAUTION` | 주의 상태 |
| `WARNING` | 경고 발생 및 유지 |
| `EXCEEDED` | 현재 기준 초과 경보 유지 |
| `RECOVERY` | 정상 회복 여부 확인 중 |

상태 전이 예시:

```text
                  NORMAL
                     |
                     | 위험 신호 감지
                     v
                  CAUTION
                     |
                     | 경보 조건 충족
                     v
                  WARNING
                     |
                     | 현재 기준 초과
                     v
                 EXCEEDED
                     |
                     | 안정 조건 확인 시작
                     v
                 RECOVERY
                     |
                     | 연속 정상 조건 충족
                     v
                   NORMAL
```

이 흐름은 개념적인 예시이다.

실제 구현에서는 `WARNING`에서 바로 `RECOVERY`로 이동할 수 있고, `EXCEEDED`에서 안정 상태를 확인한 뒤 회복할 수도 있다.

다음 사항을 고려해야 한다.

- 동일 굴뚝의 중복 경보 방지
- 기존 경보보다 위험도가 높아지면 경보 승격
- 한 번의 정상 측정값으로 경보 해제하지 않기
- 일정 시간 안정 상태가 유지되면 경보 해제
- 경보 발생 및 해제 이력 저장
- `UNKNOWN` 상태에서는 기존 경보를 임의로 해제하지 않기

특히 `UNKNOWN`은 정상 회복을 의미하지 않는다. 데이터가 다시 정상적으로 수집된 이후 별도의 회복 조건을 확인해야 한다.

---

## 8. AI Inference Layer 권장 Python 프로젝트 구조

멘티들이 실제 구현할 때는 기능별로 Python 파일을 분리하는 것을 권장한다.

```text
tms-ai/
│
├── models/
│   └── nox_xgboost_v1.joblib
│
├── inference/
│   ├── feature_service.py
│   ├── ml_inference.py
│   ├── rule_engine.py
│   ├── hybrid_decision.py
│   └── inference_service.py
│
├── alert/
│   └── alert_manager.py
│
├── repository/
│   └── decision_repository.py
│
├── config/
│   └── rules.yaml
│
├── common/
│   └── status.py
│
└── main.py
```

각 파일의 역할은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `feature_service.py` | 측정 데이터를 모델 및 규칙용 Feature로 변환 |
| `ml_inference.py` | 저장된 ML Model 호출 및 예측 |
| `rule_engine.py` | 규칙 정의 및 위험 신호 탐지 |
| `hybrid_decision.py` | ML과 Rule 결과 결합 |
| `inference_service.py` | AI Inference 전체 실행 흐름 관리 |
| `alert_manager.py` | 경보 상태 및 중복 경보 관리 |
| `decision_repository.py` | 최종 판정 및 근거 저장 |
| `rules.yaml` | 규칙과 임계값 외부 설정 관리 |
| `status.py` | 공통 상태값 정의 |

### 8.1 공통 상태값 정의

각 모듈에서 문자열을 직접 입력하지 않고 공통 Enum을 사용한다.

`common/status.py`

```python
from enum import Enum


class DecisionStatus(str, Enum):

    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    WARNING_CANDIDATE = "WARNING_CANDIDATE"
    EXCEEDED = "EXCEEDED"
    UNKNOWN = "UNKNOWN"


class AlertStatus(str, Enum):

    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    EXCEEDED = "EXCEEDED"
    RECOVERY = "RECOVERY"
```

이렇게 관리하면 ML Inference, Rule Engine, Hybrid Decision에서 동일한 상태값을 사용할 수 있다.

### 8.2 Rule 설정 외부화

규칙의 임계값을 Python 코드에 직접 작성하면 규칙을 변경할 때마다 소스 코드를 수정해야 한다.

따라서 초기에는 YAML 파일로 규칙을 분리하는 것을 권장한다.

`config/rules.yaml`

```yaml
rule_version: "1.0"

current_ratio:
  caution: 80
  exceeded: 90

rapid_rise:
  enabled: true
  min_ratio: 70
  max_ratio: 80
  delta_30m_threshold: 10

volatility:
  enabled: true
  std_6slot_threshold: 8
```

위 임계값은 교육용 예시이다.

실제 운영에서는 검증된 임계값을 사용해야 한다.

Rule 버전을 함께 저장하면 규칙을 변경한 이후에도 과거 판정 결과를 재현하고 비교할 수 있다.

---

## 9. AI Inference Layer 전체 실행 서비스

앞에서 구현한 기능을 하나의 서비스로 연결한다.

### 9.1 전체 실행 흐름

```text
1. TMS 측정 데이터 수신
          |
          v
2. Feature Service
          |
          v
3. ML Inference
          |
          v
4. Rule Engine
          |
          v
5. Hybrid Decision
          |
          v
6. Decision History 저장
          |
          v
7. Alert Manager
          |
          v
8. 경보 발생 및 상태 관리
```

### 9.2 실행 서비스 예시

```python
class TMSInferenceService:

    def __init__(
        self,
        feature_service,
        ml_inference,
        rule_engine,
        decision_repository,
        alert_manager
    ):
        self.feature_service = feature_service
        self.ml_inference = ml_inference
        self.rule_engine = rule_engine
        self.decision_repository = decision_repository
        self.alert_manager = alert_manager

    def run(self, raw_data):

        # 1. Feature 생성
        features = self.feature_service.transform(
            raw_data
        )

        # 2. ML 예측
        ml_result = self.ml_inference.predict(
            features
        )

        # 3. Rule 평가
        rule_result = self.rule_engine.evaluate(
            features
        )

        # 4. Hybrid Decision
        final_result = hybrid_decision(
            ml_result,
            rule_result
        )

        # 5. 판정 결과 저장
        self.decision_repository.save(
            features,
            ml_result,
            rule_result,
            final_result
        )

        # 6. Alert Manager 처리
        alert_result = self.alert_manager.process(
            final_result
        )

        return {
            "ml_result": ml_result,
            "rule_result": rule_result,
            "final_result": final_result,
            "alert_result": alert_result
        }
```

위 코드는 전체 모듈 간 호출 관계를 보여주는 구조 예시이다.

실제 실행을 위해서는 각 모듈의 인터페이스, 예외 처리, 데이터 검증, 트랜잭션 및 저장 로직을 추가해야 한다.

---

## 10. TMS 데이터 처리 시 고려사항

AI Inference Layer는 모델의 예측 정확도뿐 아니라 입력 데이터의 시간 정합성과 품질도 중요하다.

### 10.1 30분 단위 데이터 처리

우리 프로젝트는 30분 단위 측정 데이터를 기본으로 사용한다.

따라서 다음 항목을 정확하게 관리해야 한다.

- 굴뚝별 측정 데이터 시간 정렬
- 동일 시각 중복 데이터 제거 또는 격리
- 누락된 측정 슬롯 확인
- 늦게 도착한 데이터 처리
- Feature 생성 시 과거 데이터만 사용
- 모델 입력과 Rule Engine 입력의 기준 시각 일치

특히 `recent_max_6slot`, `std_6slot`, `delta_30m` 등의 Feature는 기준 시점 이전 데이터만 사용해야 한다.

미래 30분 또는 60분 측정값이 Feature에 포함되면 Data Leakage가 발생한다.

### 10.2 데이터 품질과 위험 판정 분리

다음 두 가지는 서로 다른 문제이다.

| 구분 | 의미 |
|---|---|
| 데이터 품질 문제 | 입력 데이터가 신뢰할 수 있는지 확인 |
| 배출 위험 문제 | 실제 배출 농도가 상승하거나 기준에 접근하는지 확인 |

예를 들어 200% 초과 데이터가 발생했다고 가정해 보자.

이 값이 실제 급등인지 센서 오류인지는 별도로 검증해야 한다.

따라서 데이터 품질 검사를 수행하되 실제 위험 가능성을 무조건 제거하지 않도록 설계해야 한다.

---

## 11. AI Inference Layer 성능 검증

Hybrid Decision을 구현했다고 해서 기존 ML Model보다 성능이 좋아졌다고 볼 수는 없다.

반드시 동일한 데이터와 평가 조건으로 비교해야 한다.

### 11.1 비교 실험

| 실험 | 적용 방식 | 검증 목적 |
|---|---|---|
| 실험 1 | 기존 ML Model | 기준 성능 확인 |
| 실험 2 | ML + 상승폭 Rule | 급상승 탐지 효과 |
| 실험 3 | ML + 변동성 Rule | 변동성 활용 효과 |
| 실험 4 | ML + 상승폭 + 변동성 Rule | 두 규칙 결합 효과 |

초기에는 기존 모델의 저장 예측 결과를 재사용하여 Rule 정책의 추가 효과를 평가한다.

새로운 모델을 학습하기 전에 기존 예측 결과에 Rule을 추가했을 때 얼마나 개선되는지 확인하는 것이 우선이다.

### 11.2 주요 평가 지표

| 평가 지표 | 의미 |
|---|---|
| Event Recall | 실제 초과 사건 중 탐지한 사건의 비율 |
| False Alert | 실제 초과 사건이 아닌데 발생한 경보 |
| False Alerts / Stack / Day | 굴뚝당 하루 평균 오탐 경보 수 |
| Lead Time | 실제 초과 발생 전 경고한 시간 |
| A형 Recall | 과거 6개 슬롯에 초과 관측이 없었던 사건의 탐지율 |
| B형 Recall | 과거 6개 슬롯 중 초과 관측이 있었던 사건의 탐지율 |
| Low-concentration Recall | 저농도 출발 사건의 탐지율 |

특히 이번 TMS 진단에서 확인한 문제는 저농도 급상승 사건의 탐지율이 낮다는 것이다.

따라서 전체 Recall만 비교하지 말고 저농도 급상승 사건을 별도로 평가해야 한다.

또한 `WARNING_CANDIDATE`를 추가 탐지 성과로 계산하더라도 실제 운영 경보 성능과 구분해서 보고해야 한다.

### 11.3 평가 데이터 분리 원칙

현재 프로젝트는 대만 데이터 약 36개월, 국내 데이터 약 11일을 확보하고 있다.

따라서 다음 평가 원칙을 권장한다.

1. 대만 데이터를 시간 순서에 따라 Train / Validation으로 분리한다.
2. 대만 Validation에서 Rule 임계값과 Hybrid 정책을 결정한다.
3. 국내 데이터는 독립 평가 데이터로 활용한다.
4. 국내 Test 결과를 반복 확인하면서 임계값을 최적화하지 않는다.

또한 미래 30분 또는 60분 관측값은 Label 생성과 평가에만 사용해야 하며 Inference 입력에 포함하면 안 된다.

---

## 12. Agent Observability와의 연계

AI Inference Layer가 최종 판정을 내리면 그 결과와 판단 근거를 저장해야 한다.

이 데이터는 나중에 모델 성능 분석, Rule 개선, 경보 원인 분석에 활용할 수 있다.

### 12.1 Decision History 권장 저장 항목

| 컬럼 | 설명 |
|---|---|
| decision_id | 판정 고유 ID |
| stack_code | 굴뚝 코드 |
| item_code | 측정 항목 |
| measure_time | 측정 시각 |
| ml_score | ML 예측 점수 |
| ml_risk_level | ML 위험도 |
| inference_status | ML 실행 성공/실패 |
| rule_ids | 실행된 Rule ID |
| rule_risk_level | Rule 위험도 |
| final_decision | 최종 판정 |
| decision_reason | 판정 근거 |
| model_version | ML Model 버전 |
| rule_version | Rule 설정 버전 |
| data_quality | 데이터 품질 상태 |
| created_at | 판정 생성 시각 |

### 12.2 저장해야 하는 이유

이러한 기록을 통해 다음 질문에 답할 수 있다.

- 어떤 ML Model이 경보를 발생시켰는가?
- 어떤 Rule이 추가 위험 신호를 탐지했는가?
- ML은 정상으로 판단했지만 Rule이 위험을 발견한 사건은 무엇인가?
- Rule을 변경한 이후 오탐이 얼마나 증가했는가?
- 특정 모델 버전에서 저농도 급상승 탐지율이 개선되었는가?
- `UNKNOWN`이 발생한 원인은 무엇인가?

이를 기반으로 Agent Observability에서 AI 판단 과정을 추적하고 분석할 수 있다.

### 12.3 LLM Agent의 역할

LLM Agent는 AI Inference Layer의 판정 결과를 대신 결정하지 않는다.

LLM Agent의 역할은 다음과 같다.

- 판정 이력 조회
- ML 및 Rule 판단 근거 설명
- 특정 굴뚝의 과거 위험 이력 분석
- 경보 발생 원인 요약
- 운영자가 요청한 조건에 따른 데이터 검색
- 성능 분석 결과 설명

수치 판정과 최종 경보 결정은 Python 기반의 Inference 및 Decision 로직이 수행해야 한다.

즉, **Python은 판단하고 LLM Agent는 판단 결과를 설명한다.**

---

## 13. 멘티 구현 순서 및 학습 목표

처음부터 모든 기능을 동시에 구현하지 말고 다음 순서로 진행한다.

| 단계 | 구현 내용 | 완료 기준 |
|---|---|---|
| 1단계 | Feature Service | ML과 Rule이 공통으로 사용할 Feature 생성 |
| 2단계 | ML Inference | 저장된 모델을 불러와 예측 결과 생성 |
| 3단계 | Rule Engine | 현재 농도, 상승폭, 변동성, 데이터 품질 규칙 평가 |
| 4단계 | Hybrid Decision | ML과 Rule 결과를 결합하여 최종 판정 |
| 5단계 | Alert Manager | 중복 방지, 상태 전이 및 경보 해제 |
| 6단계 | Decision History | 판정 결과 및 판단 근거 저장 |
| 7단계 | Offline Replay | 기존 ML과 Hybrid 정책의 성능 비교 |
| 8단계 | Agent Observability | 판정 이력 조회 및 분석 기능 구현 |

### 13.1 멘티가 반드시 이해해야 할 질문

**질문 1. ML Model은 무엇을 하는가?**

과거 데이터를 학습하여 미래 초과 가능성을 예측한다.

**질문 2. Rule Engine은 무엇을 하는가?**

사전에 정의한 조건을 이용하여 현재 위험 신호와 예외 상황을 탐지한다.

**질문 3. Hybrid Decision은 무엇을 하는가?**

ML과 Rule의 결과를 결합하여 최종 위험도를 결정한다.

**질문 4. WARNING_CANDIDATE와 WARNING은 무엇이 다른가?**

`WARNING_CANDIDATE`는 검증 전 위험 후보이고, `WARNING`은 검증된 경보 조건을 충족한 상태이다.

**질문 5. UNKNOWN과 NORMAL은 무엇이 다른가?**

`NORMAL`은 판단 결과 위험 신호가 없는 상태이고, `UNKNOWN`은 데이터 문제 등으로 판단할 수 없는 상태이다.

**질문 6. EXCEEDED와 WARNING은 무엇이 다른가?**

`EXCEEDED`는 현재 기준대비율이 90% 이상인 상태이고, `WARNING`은 미래 초과 가능성 또는 검증된 경보 조건에 따른 경고 상태이다.

**질문 7. Alert Manager는 왜 필요한가?**

동일한 위험 상황에서 경보가 반복 발생하지 않도록 경보의 시작, 유지, 승격, 회복 및 해제를 관리하기 위해 필요하다.

---

## 14. 최종 정리

TMS 프로젝트에서 구현하려는 AI Inference Layer의 핵심은 ML Model 하나에 모든 판단을 맡기지 않는 것이다.

각 구성 요소가 명확한 역할을 수행하도록 설계해야 한다.

| 구성 요소 | 핵심 역할 |
|---|---|
| Feature Service | 판단에 필요한 데이터 생성 |
| ML Model | 학습된 패턴으로 미래 위험 예측 |
| Rule Engine | 명시적인 조건으로 위험 신호 탐지 |
| Hybrid Decision | ML과 Rule 결과를 결합하여 최종 판정 |
| Alert Manager | 경보 발생, 유지, 승격 및 해제 |
| Agent Observability | AI 판단 과정과 결과 추적 |

최종적으로 기억해야 할 핵심은 다음과 같다.

> **ML Model이 예측하고, Rule Engine이 보완하며, Hybrid Decision이 최종 판단하고, Alert Manager가 경보를 관리한다.**

그리고 모든 판정 결과는 기록되어야 하며, 나중에 동일한 입력과 동일한 모델 및 규칙 버전으로 판단 과정을 추적할 수 있어야 한다.

이 구조의 궁극적인 목표는 AI가 판단하고 실행하더라도 최종 결과와 판단 근거를 추적하고 통제할 수 있는 TMS AI 운영 체계를 만드는 것이다.
