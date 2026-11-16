
# TMS 분류 모델 Upgrade 및 성능 개선 방안

## 1. 모델 개선 방향

파라미터 튜닝과 SMOTE는 모델 성능을 개선하는 대표적인 방법이다. 하지만 모델을 Upgrade하는 방법이 이것만 있는 것은 아니다.

이번 TMS 프로젝트에서는 국내 모델의 FPR(오경보율)이 낮은 반면 Recall(재현율)도 낮게 나타났다는 점에 주목할 필요가 있다.

따라서 무조건 데이터를 늘리거나 복잡한 모델을 적용하기보다는 다음 순서로 접근하는 것이 좋다.

1. 현재 모델의 판단 기준(Threshold)이 적절한지 확인
2. 위험 데이터의 불균형 문제 개선
3. 입력 Feature 개선
4. 모델의 하이퍼파라미터 최적화
5. 다양한 알고리즘 비교
6. 최종 운영 목적에 적합한 모델 선정

**핵심은 오경보를 크게 늘리지 않으면서 실제 위험 탐지율(Recall)을 높이는 방향으로 모델을 개선하는 것이다.**

---

## 2. 모델 성능 개선 방법

모델을 개선하는 방법은 크게 6가지로 구분할 수 있다.

| 구분 | 개선 방법 | 목적 | 우선순위 |
|---|---|---|---|
| ① 판단 기준 최적화 | Threshold 조정 | 오경보와 미탐의 균형 조정 | 최우선 |
| ② 데이터 불균형 개선 | Class Weight, SMOTE | 적은 위험 데이터의 학습 강화 | 높음 |
| ③ Feature Engineering | 입력 변수 추가·변경 | 위험 상황을 구분할 정보 강화 | 높음 |
| ④ 하이퍼파라미터 탐색 | ValidationCurve, GridSearchCV, Optuna 등 | 최적 파라미터 탐색 | 높음 |
| ⑤ 알고리즘 개선 | XGBoost, LightGBM, LSTM 등 | 데이터에 적합한 모델 탐색 | 중간 |
| ⑥ 실행 최적화 | Parallel Processing | 실험 및 학습 시간 단축 | 병행 적용 |

### 2.1 개선 방법의 차이

- **Threshold 조정**: 이미 학습된 모델의 예측 점수 중 어디부터 위험으로 판단할지 결정한다.
- **Class Weight / SMOTE**: 위험 데이터의 불균형을 개선하여 학습한다.
- **Feature Engineering**: 모델이 위험 상황을 더 잘 구분할 수 있도록 입력 데이터를 개선한다.
- **하이퍼파라미터 튜닝**: 모델의 학습 설정값을 최적화한다.
- **알고리즘 개선**: 다른 모델 구조를 적용하여 성능을 비교한다.
- **Parallel Processing**: 여러 실험을 동시에 실행하여 탐색 시간을 단축한다.

따라서 먼저 Threshold 문제인지 확인하고, 이후 데이터와 Feature, 모델 자체를 개선하는 것이 효율적이다.

---

## 3. 방법 ① Threshold 최적화

### 3.1 현재 모델의 문제점

현재 국내 NOx 분류 모델의 분석 결과는 다음과 같다.

| 지표 | 대만 | 국내 |
|---|---:|---:|
| FPR | 5.09% | 1.00% |
| Recall | 66.9% | 46.9% |

국내 모델은 FPR과 Recall이 모두 낮다.

이것이 모델의 위험 구분 능력 문제인지, 위험 판정 기준인 Threshold가 보수적으로 설정된 문제인지 먼저 확인해야 한다.

### 3.2 Threshold에 따른 변화

모델이 위험 확률을 예측한다고 가정하자.

| 실제 데이터 | 예측 점수 |
|---|---:|
| 위험 데이터 A | 0.95 |
| 위험 데이터 B | 0.87 |
| 위험 데이터 C | 0.78 |
| 정상 데이터 D | 0.65 |
| 정상 데이터 E | 0.32 |

Threshold가 0.8이면 0.8 이상을 위험으로 판정한다.

Threshold를 0.7로 낮추면 더 많은 위험 데이터를 탐지할 수 있지만, 정상 데이터도 위험으로 잘못 판단할 가능성이 커진다.

| Threshold 변화 | FPR | Recall | 의미 |
|---|---|---|---|
| 높임 | 감소 가능 | 감소 가능 | 경보를 보수적으로 발생 |
| 낮춤 | 증가 가능 | 증가 가능 | 위험 탐지를 적극적으로 수행 |

일반적인 경향이며, 실제 변화 폭은 모델의 예측 점수 분포에 따라 달라진다.

### 3.3 권장 실험

- FPR 목표를 1%, 2%, 5%로 설정
- Validation 데이터에서 각 목표에 맞는 Threshold 결정
- 결정한 Threshold를 고정하여 Test 데이터 평가
- 각 조건에서 Recall, Precision, FPR 비교

이 결과를 통해 국내 모델의 Recall을 어느 정도까지 개선할 수 있는지 확인한다.

**주의사항**

- Threshold는 Validation 데이터에서 결정한다.
- Test 데이터는 최종 성능 평가에만 사용한다.
- Test 결과를 보고 Threshold를 다시 조정하면 평가 결과가 왜곡된다.
- 대만과 국내는 데이터 분포가 다를 수 있으므로 각 데이터의 Validation에서 Threshold를 별도로 결정한다.

---

## 4. 방법 ② 데이터 불균형 개선

TMS 데이터는 정상 데이터에 비해 위험 데이터가 적을 수 있다.

이러한 데이터에서는 모델이 정상 데이터의 패턴을 주로 학습하고, 상대적으로 적은 위험 데이터를 충분히 학습하지 못할 가능성이 있다.

이 문제를 개선하는 대표적인 방법이 Class Weight와 SMOTE이다.

### 4.1 Class Weight

위험 데이터에 더 높은 가중치를 주어 학습시키는 방법이다.

예를 들어 다음과 같은 데이터가 있다고 가정한다.

| 데이터 | 건수 | 비율 |
|---|---:|---:|
| 정상 | 9,000 | 90% |
| 위험 | 1,000 | 10% |

정상 데이터가 훨씬 많기 때문에 모델이 정상 분류에 치우칠 수 있다.

Class Weight를 적용하면 위험 데이터를 잘못 분류했을 때 더 큰 패널티를 주도록 학습할 수 있다.

**장점**

- 실제 데이터를 복제하거나 생성하지 않는다.
- 적용이 비교적 간단하다.
- XGBoost 등의 분류 모델에서 지원한다.

**TMS 프로젝트 적용**

- XGBoost의 `scale_pos_weight` 활용
- 위험 데이터에 적용하는 가중치를 여러 수준으로 비교
- 가중치에 따른 Recall과 FPR 변화 확인

위험 데이터 비율을 참고해 초기 가중치를 설정할 수 있지만, 실제 최적값은 Validation 성능으로 결정해야 한다.

### 4.2 SMOTE

SMOTE(Synthetic Minority Over-sampling Technique)는 위험 데이터가 부족할 때 기존 위험 데이터의 특성을 이용해 새로운 위험 데이터를 합성하는 방법이다.

예를 들어 위험 데이터 A와 B가 있을 때 두 데이터 사이의 특성을 보간하여 새로운 위험 데이터를 생성한다.

**장점**

- 부족한 위험 데이터의 학습 비중을 높일 수 있다.
- 기존 데이터를 단순 복제하는 방식보다 다양한 학습 샘플을 제공할 수 있다.

**주의사항**

- 실제로 발생하지 않은 위험 패턴을 생성할 수 있다.
- 시간에 따른 농도 변화와 설비 운전 상태의 연속성을 훼손할 수 있다.
- 정상과 위험의 경계에 있는 데이터가 합성되면 오히려 오분류가 증가할 수 있다.

특히 TMS 데이터는 시간 순서와 설비별 특성이 중요하므로 SMOTE를 무조건 적용하는 것은 권장하지 않는다.

### 4.3 Class Weight와 SMOTE 비교

| 비교 항목 | Class Weight | SMOTE |
|---|---|---|
| 데이터 생성 | 없음 | 합성 데이터 생성 |
| 위험 데이터 학습 비중 | 가중치로 증가 | 샘플 수로 증가 |
| 시간적 특성 보존 | 원본 데이터 유지 | 합성 과정에서 훼손 가능 |
| 구현 난이도 | 낮음 | 상대적으로 높음 |
| 권장 순서 | 먼저 적용 | 비교 실험으로 적용 |

**권장 실험 순서**

1. 기존 모델(Baseline)
2. Class Weight 적용
3. SMOTE 적용
4. Class Weight와 SMOTE 조합

네 가지 실험을 동일한 데이터 분할과 평가 기준으로 비교한다.

**SMOTE 적용 시 주의사항**

- Train, Validation, Test 데이터를 먼저 분리한다.
- SMOTE는 Train 데이터에만 적용한다.
- Validation과 Test에는 합성 데이터를 적용하지 않는다.
- 시간 순서와 설비별 특성이 훼손되지 않는지 확인한다.
- 시계열 데이터의 경우 일반 SMOTE 외에 시간적 특성을 고려한 샘플링 방법도 검토한다.

---

## 5. 방법 ③ Feature Engineering

개인적으로 이번 TMS 프로젝트에서 중요하게 보고 싶은 개선 방법이다.

모델 알고리즘을 바꾸는 것보다 **위험 상황을 구분할 수 있는 입력 데이터를 잘 설계하는 것이 더 큰 효과를 낼 수도 있기 때문이다.**

현재 TMS 데이터에는 농도와 과거 시점의 데이터가 포함되어 있다. 여기에 시간에 따른 변화 특성을 더 구체적으로 반영할 수 있다.

### 5.1 추천 Feature

| Feature | 의미 | 기대 효과 |
|---|---|---|
| 현재 농도 | 현재 배출 농도 | 현재 위험 수준 파악 |
| 30분 전 농도 | 직전 측정 농도 | 단기 변화 확인 |
| 1시간 전 농도 | 과거 농도 | 변화 추세 확인 |
| 3시간 평균 | 최근 3시간 평균 농도 | 단기적인 변동 완화 |
| 6시간 평균 | 최근 6시간 평균 농도 | 중기적인 변화 확인 |
| 최근 1시간 변화량 | 현재 농도 - 1시간 전 농도 | 상승·하락 추세 파악 |
| 최근 3시간 최대값 | 최근 최대 농도 | 일시적 고농도 경험 반영 |
| 최근 24시간 표준편차 | 농도 변동성 | 불안정한 배출 패턴 파악 |
| 기준치 대비 현재 비율 | 현재 농도 / 기준치 | 위험 수준의 상대적 위치 파악 |

### 5.2 Feature Engineering 예시

현재 농도가 80%인 두 설비가 있다고 가정한다.

- A 설비: 3시간 동안 80% 수준 유지
- B 설비: 3시간 전 40%에서 현재 80%까지 상승

현재 농도만 보면 두 설비는 동일하지만, B 설비는 빠르게 상승하고 있다.

이러한 차이를 모델이 학습하도록 하는 것이 Feature Engineering이다.

### 5.3 기존 24시간 Feature 실험과 연결

기존 분석에서는 24시간 이력 Feature를 추가했을 때 다음과 같은 결과가 있었다.

- 대만 NOx: AP 개선
- 국내 NOx: AP 하락
- SOx: 대만과 국내 모두 AP 개선

따라서 24시간 이력 Feature가 항상 좋은 것은 아니다.

**추천하는 방법은 Feature를 한 번에 많이 추가하는 것이 아니라, 하나씩 추가하고 제거하면서 성능 변화를 확인하는 것이다.**

각 Feature 조합마다 모델을 다시 학습해야 하며, 동일한 데이터 분할과 평가 조건을 유지해야 한다.

### 5.4 Feature Ablation 실험

Feature가 실제로 모델 성능 개선에 기여하는지 확인하기 위한 실험이다.

| 실험 | Feature 구성 | 확인 목적 |
|---|---|---|
| A | 기존 Feature | Baseline |
| B | 기존 + 최근 1시간 변화량 | 단기 변화 효과 |
| C | 기존 + 이동평균 | 평균적인 추세 효과 |
| D | 기존 + 변동성 | 농도 불안정성 효과 |
| E | 기존 + 24시간 이력 | 장기 이력 효과 |
| F | 성능이 좋은 Feature 조합 | 조합 효과 |

각 실험은 Feature 구성을 변경한 뒤 모델을 다시 학습해야 한다. 학습된 모델에서 입력 변수를 단순히 제거하는 방식만으로는 올바른 Feature Ablation 결과를 얻기 어렵다.

---

## 6. 방법 ④ 하이퍼파라미터 탐색 및 최적화

하이퍼파라미터는 모델이 학습하는 과정과 복잡도를 결정하는 설정값이다.

예를 들어 XGBoost에서는 트리의 깊이, 학습률, 트리 개수 등을 조정하여 성능을 개선할 수 있다.

하지만 파라미터를 임의로 변경하는 것보다 체계적인 탐색 방법을 사용하는 것이 좋다.

### 6.1 주요 탐색 방법

| 방법 | 주요 기능 | 장점 | 단점 |
|---|---|---|---|
| ValidationCurve | 특정 파라미터 변화에 따른 성능 분석 | 파라미터 영향 파악이 쉬움 | 여러 파라미터 조합 탐색은 어려움 |
| ParameterGrid | 탐색할 파라미터 조합 생성 | 실험 조합을 명확하게 관리 | 자체적으로 성능 평가하지 않음 |
| GridSearchCV | 모든 파라미터 조합을 교차 검증 | 체계적이고 재현성 높음 | 조합이 많으면 느림 |
| RandomizedSearchCV | 무작위로 파라미터 조합 탐색 | 넓은 범위를 효율적으로 탐색 | 최적 조합을 보장하지 않음 |
| Bayesian Optimization | 이전 실험 결과를 활용해 유망한 조합 탐색 | 적은 실험으로 좋은 조합 탐색 가능 | 탐색 과정과 설정이 상대적으로 복잡 |
| Optuna | 다양한 탐색 알고리즘과 실험 관리 지원 | 유연한 탐색 및 중간 결과 관리 | 탐색 설정과 평가 설계 필요 |

여기서 Parallel Processing은 탐색 알고리즘과 별도로 적용할 수 있는 실행 최적화 방법이다.

### 6.2 ValidationCurve

특정 파라미터 하나를 변화시키면서 모델 성능이 어떻게 달라지는지 확인하는 방법이다.

예를 들어 XGBoost의 `max_depth`를 변경하면서 성능을 비교한다.

```python
from sklearn.model_selection import validation_curve
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

param_range = [2, 3, 4, 5, 6, 7, 8]

train_scores, valid_scores = validation_curve(
    model,
    X_train,
    y_train,
    param_name="max_depth",
    param_range=param_range,
    cv=3,
    scoring="average_precision",
    n_jobs=-1
)

print("Train:", train_scores.mean(axis=1))
print("Valid:", valid_scores.mean(axis=1))
```

**결과 해석**

- Train과 Validation 점수가 모두 낮음: 과소적합 가능성
- Train 점수는 높지만 Validation 점수가 낮음: 과적합 가능성
- Validation 점수가 가장 높은 구간: 추가 탐색 후보

이번 프로젝트에서는 다음 파라미터를 하나씩 분석할 수 있다.

- `max_depth`
- `learning_rate`
- `min_child_weight`
- `n_estimators`

ValidationCurve는 특정 파라미터의 영향을 파악하는 데 적합하며, 여러 파라미터 조합을 동시에 탐색하는 용도는 아니다.

### 6.3 ParameterGrid

여러 파라미터의 조합을 미리 정의하고 생성하는 방법이다.

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500]
}

grid = ParameterGrid(param_grid)

print("전체 실험 조합:", len(grid))

for params in grid:
    print(params)
```

위 설정에서는 다음과 같이 총 27개 조합이 생성된다.

\[
3 \times 3 \times 3 = 27
\]

ParameterGrid는 파라미터 조합을 생성하는 기능이지, 그 자체가 최적 파라미터를 찾아주는 알고리즘은 아니다.

### 6.4 GridSearchCV

ParameterGrid에서 생성한 조합을 교차 검증으로 평가하고 가장 좋은 조합을 찾는 방법이다.

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

model = XGBClassifier(
    random_state=42,
    eval_metric="aucpr"
)

param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="average_precision",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best AP:", grid_search.best_score_)
```

27개 조합에 3-Fold 교차 검증을 적용하면 총 81회의 모델 학습이 수행된다.

**장점**

- 모든 조합을 빠짐없이 비교할 수 있다.
- 결과를 재현하고 설명하기 쉽다.
- 기준 모델의 파라미터 탐색에 적합하다.

**단점**

- 탐색 조합이 많아지면 학습 시간이 급격히 증가한다.
- 모든 조합이 좋은 후보인 것은 아니므로 비효율적인 실험이 발생할 수 있다.

따라서 처음부터 파라미터를 너무 넓게 설정하기보다는 ValidationCurve 등으로 탐색 범위를 좁힌 다음 적용하는 것이 좋다.

### 6.5 RandomizedSearchCV

GridSearchCV는 사전에 지정한 모든 조합을 탐색하지만, RandomizedSearchCV는 지정한 범위에서 일부 조합을 무작위로 선택하여 평가한다.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

model = XGBClassifier(
    random_state=42,
    eval_metric="aucpr"
)

param_distributions = {
    "max_depth": randint(2, 9),
    "learning_rate": uniform(0.01, 0.19),
    "n_estimators": randint(100, 801),
    "min_child_weight": randint(1, 11)
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring="average_precision",
    cv=3,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best AP:", search.best_score_)
```

**장점**

- 탐색 조합이 많을 때 효율적이다.
- 연속형 파라미터 탐색에 적합하다.
- GridSearchCV보다 적은 횟수로 넓은 범위를 탐색할 수 있다.

GridSearchCV와 Bayesian Optimization 사이에서 중간 단계의 탐색 방법으로 활용할 수 있다.

### 6.6 Bayesian Optimization

Bayesian Optimization은 이전 실험 결과를 활용하여 다음에 평가할 유망한 파라미터 조합을 선택하는 방법이다.

GridSearchCV는 미리 정한 모든 조합을 평가하지만, Bayesian Optimization은 이전 결과를 바탕으로 다음 실험 대상을 결정한다.

**GridSearch와 Bayesian Optimization의 차이**

| 비교 | GridSearch | Bayesian Optimization |
|---|---|---|
| 탐색 방식 | 사전에 정한 모든 조합 평가 | 이전 결과를 이용해 다음 조합 선택 |
| 탐색 순서 | 사전 정의된 순서 | 이전 평가 결과에 따라 결정 |
| 실험 횟수 | 조합 수에 따라 증가 | 제한된 횟수로 유망한 조합 탐색 |
| 장점 | 단순하고 재현하기 쉬움 | 탐색 효율이 높을 수 있음 |
| 단점 | 조합이 많으면 느림 | 탐색 설정과 과정이 상대적으로 복잡 |

대표적인 구현 도구는 다음과 같다.

- Optuna: TPE 등 다양한 탐색 알고리즘 지원
- scikit-optimize: Bayesian 탐색 및 `BayesSearchCV` 제공
- BayesianOptimization: 별도 Python 라이브러리

Optuna의 기본 TPE Sampler는 이전 실험 결과를 이용해 유망한 파라미터 조합을 탐색하는 방식이다. 엄밀히 말해 Gaussian Process 기반 탐색과 구현 방식은 다르지만, 실무에서는 Bayesian Optimization 계열의 순차적 최적화 방법으로 함께 활용된다.

### 6.7 Optuna 적용 예시

```python
import optuna

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

def objective(trial):

    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 800
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "eval_metric": "aucpr",
        "random_state": 42
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_valid)[:, 1]

    return average_precision_score(y_valid, probabilities)


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

print("Best Parameters:", study.best_params)
print("Best AP:", study.best_value)
```

위 코드는 AP(Average Precision)를 최적화하는 예시다.

탐색 횟수는 50회로 설정했지만, 데이터 규모와 학습 시간에 따라 조정할 수 있다.

**주의사항**

- Optuna도 Validation 데이터만 사용해 최적 파라미터를 결정해야 한다.
- Test 데이터는 탐색 과정에 사용하지 않는다.
- 동일한 데이터 분할과 평가 조건으로 GridSearchCV와 비교한다.
- 탐색 결과가 특정 데이터 기간에만 유효하지 않은지 확인한다.

---

## 7. 방법 ⑤ Parallel Processing

여러 파라미터 조합이나 모델 학습을 동시에 실행하는 방법이다.

예를 들어 GridSearchCV에서 다음과 같이 설정할 수 있다.

```python
GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="average_precision",
    n_jobs=-1
)
```

- `n_jobs=1`: 순차 실행
- `n_jobs=2`: 2개 작업 병렬 실행
- `n_jobs=-1`: 사용 가능한 CPU 작업자 최대 활용

**장점**

- 전체 탐색 시간 단축
- 여러 파라미터 조합을 동시에 평가
- 대규모 실험에 유리

**주의사항**

- CPU 사용량 증가
- 메모리 사용량 증가
- XGBoost 자체도 멀티스레드를 사용하므로 과도한 병렬 실행 주의
- 여러 실험을 동시에 수행할 때 자원 경합이 발생할 수 있음

예를 들어 GridSearchCV가 여러 조합을 병렬로 실행한다면 XGBoost의 `n_jobs`를 1~2 정도로 제한하는 방식도 고려할 수 있다.

**중요한 점은 Parallel Processing은 별도의 모델 최적화 알고리즘이 아니라 탐색 및 학습 시간을 단축하는 실행 방식이라는 것이다.**

---

## 8. TMS 프로젝트에 권장하는 하이퍼파라미터 탐색 전략

모든 방법을 한꺼번에 사용하는 것보다 단계별로 적용하는 것이 좋다.

| 단계 | 적용 방법 | 수행 목적 |
|---|---|---|
| 1단계 | ValidationCurve | 주요 파라미터의 영향 분석 |
| 2단계 | ParameterGrid | 탐색 범위와 조합 정의 |
| 3단계 | GridSearchCV | 기본 파라미터 조합 비교 |
| 4단계 | RandomizedSearchCV | 넓은 탐색 범위에서 유망한 조합 탐색 |
| 5단계 | Bayesian Optimization(Optuna) | 유망한 파라미터 조합을 효율적으로 탐색 |
| 6단계 | Parallel Processing | 탐색 시간 단축 |
| 7단계 | Threshold 최적화 | 목표 FPR에서 Recall 최대화 |

Parallel Processing은 1~6단계에서 필요한 만큼 병행 적용한다.

### 8.1 XGBoost 주요 파라미터

| 파라미터 | 의미 | 조정 목적 |
|---|---|---|
| `max_depth` | 트리의 최대 깊이 | 모델 복잡도 조정 |
| `learning_rate` | 학습률 | 학습 속도와 일반화 조정 |
| `n_estimators` | 트리 개수 | 학습 반복 횟수 조정 |
| `min_child_weight` | 자식 노드의 최소 가중치 | 과적합 방지 |
| `subsample` | 학습 데이터 샘플 비율 | 일반화 성능 조정 |
| `colsample_bytree` | 트리별 Feature 사용 비율 | Feature 편향 완화 |
| `reg_alpha` | L1 규제 | 불필요한 변수 영향 억제 |
| `reg_lambda` | L2 규제 | 모델 복잡도 억제 |

### 8.2 탐색 범위 설정 예시

| 파라미터 | 탐색 범위 예시 |
|---|---|
| `max_depth` | 2 ~ 8 |
| `learning_rate` | 0.01 ~ 0.2 |
| `n_estimators` | 100 ~ 800 |
| `min_child_weight` | 1 ~ 10 |
| `subsample` | 0.7 ~ 1.0 |
| `colsample_bytree` | 0.7 ~ 1.0 |
| `reg_alpha` | 0 ~ 10 |
| `reg_lambda` | 0.1 ~ 10 |

위 범위는 초기 탐색을 위한 예시이며, 기존 Baseline 모델의 설정값과 ValidationCurve 결과를 참고해 조정한다.

---

## 9. 방법 ⑥ 모델 알고리즘 자체를 개선

현재 모델이 Random Forest 또는 XGBoost 중심이라면 다른 알고리즘과 비교할 수 있다.

| 모델 | 특징 | TMS 적용 관점 |
|---|---|---|
| Random Forest | 여러 결정 트리의 결과 결합 | 기본 비교 모델 |
| XGBoost | 성능이 우수한 부스팅 계열 | 현재 모델과 비교 기준 |
| LightGBM | 빠른 학습과 대규모 데이터 처리 | 데이터 규모가 큰 TMS에 적합 |
| CatBoost | 범주형 데이터 처리에 강점 | 설비·사업장 관련 범주형 변수 활용 시 검토 |
| LSTM | 시간 순서와 과거 패턴 학습 | 농도 변화 시계열 학습 |

이번 프로젝트는 데이터 규모가 크고 설비별 시간 흐름이 중요하므로 다음 비교를 권장한다.

1. 기존 Random Forest
2. 기존 XGBoost
3. LightGBM
4. LSTM

단, 모델을 많이 만드는 것이 목적은 아니다.

기존 모델보다 실제로 성능이 개선되는지, 학습 및 추론 비용은 적절한지, 운영 환경에서 재현 가능한지를 함께 판단해야 한다.

---

## 10. 회귀 모델도 별도로 개선할 필요가 있다

이번 내용은 FPR과 Recall 중심의 분류 모델에 대한 것이지만, TMS 프로젝트에는 회귀 모델도 있으므로 구분해서 접근해야 한다.

현재 회귀 모델은 정확히 60분 후의 농도를 예측하는 구조다.

기존 분석에서 고농도 구간의 예측값이 실제보다 낮게 나오는 과소 예측 문제가 확인되었다면 다음 방법을 검토할 수 있다.

| 개선 방법 | 설명 | 기대 효과 |
|---|---|---|
| Delta Prediction | 미래 농도 자체 대신 현재 대비 변화량 예측 | 상승·하락 패턴 학습 |
| Sample Weight | 고농도 데이터에 더 높은 가중치 부여 | 고농도 예측 강화 |
| Quantile Regression | 평균이 아닌 특정 분위수 예측 | 위험 수준을 보수적으로 예측 |
| Feature Engineering | 최근 변화량·이동평균 등 추가 | 미래 농도 변화 설명력 향상 |

### 10.1 Delta Prediction

미래 농도 자체가 아니라 현재 대비 변화량을 예측하는 방법이다.

\[
\Delta C = C_{t+60} - C_t
\]

\[
\hat{C}_{t+60} = C_t + \widehat{\Delta C}
\]

현재 농도에서 60분 동안 얼마나 변화할지를 예측하는 방식이다.

회귀 모델의 개선 결과는 MAE, RMSE뿐 아니라 고농도 구간의 과소 예측률도 함께 평가해야 한다.

---

## 11. 대만 데이터 활용을 통한 성능 개선

이번 프로젝트는 대만 데이터와 국내 데이터를 모두 분석하고 있으므로, 단순히 SMOTE를 적용하는 것 외에 데이터 활용 전략도 검토할 수 있다.

다음 세 가지 실험을 비교한다.

| 실험 | 학습 데이터 | 검증·평가 데이터 | 확인 목적 |
|---|---|---|---|
| A. 국내 단독 학습 | 국내 데이터 | 국내 데이터 | 기준 성능 |
| B. 대만 + 국내 학습 | 대만 + 국내 데이터 | 국내 데이터 | 대만 데이터 추가 효과 |
| C. 대만 사전 학습 후 국내 보완 학습 | 대만 데이터 → 국내 데이터 | 국내 데이터 | 데이터 도메인 차이 대응 |

여기서 중요한 것은 **최종 평가를 국내 실제 데이터로 통일하는 것**이다.

대만 데이터는 학습에 활용하되, 국내 Validation과 Test 데이터는 학습에 섞지 않아야 한다.

또한 대만과 국내 데이터는 측정 환경, 설비 특성, 농도 분포가 다를 수 있으므로 단순 결합이 항상 성능 개선으로 이어지는 것은 아니다.

SMOTE는 학습 데이터 안에서만 적용하고, 시간 순서와 설비별 특성이 훼손되지 않는지 확인해야 한다.

---

## 12. 모델 최적화 시 데이터 분할 및 평가 기준

파라미터 탐색을 아무리 잘하더라도 데이터 분할이 잘못되면 평가 결과를 신뢰하기 어렵다.

특히 TMS 데이터는 30분 단위의 시계열 데이터이므로 일반적인 무작위 분할은 주의해야 한다.

### 12.1 시간 순서 기반 데이터 분할

| 구분 | 데이터 | 용도 |
|---|---|---|
| Train | 과거 구간 | 모델 학습 |
| Validation | Train 이후 구간 | 파라미터 및 Threshold 결정 |
| Test | 가장 최근 구간 | 최종 성능 평가 |

일반적인 `GridSearchCV(cv=3)`은 시간 순서에 따른 분할을 자동으로 보장하지 않는다.

따라서 `TimeSeriesSplit` 또는 날짜·설비 단위의 사용자 정의 분할을 검토해야 한다.

동일 설비의 연속된 관측값이 학습과 평가에 무분별하게 섞이지 않도록 데이터 분할 기준을 명확히 해야 한다.

또한 미래 시점의 데이터가 과거 시점의 학습 Feature에 포함되지 않도록 Feature 생성 과정에서도 데이터 누수를 확인해야 한다.

### 12.2 TMS 데이터의 추가 고려사항

- 동일 설비의 시간 순서 유지
- Train과 Validation 사이의 시간적 누수 방지
- Feature 생성 시 미래 데이터 참조 금지
- 분류 라벨의 예측 기간과 회귀 Target의 예측 기간 구분
- 결측 및 NULL 라벨 처리 기준 유지
- 모든 모델이 동일한 평가 대상 데이터에서 비교되도록 관리

현재 분류 모델은 향후 30분 또는 60분 내 위험 수준 도달 여부를 판단하는 구조이고, 회귀 모델은 정확히 60분 후의 농도를 예측한다.

따라서 두 모델의 예측 대상과 평가 기준을 혼동하지 않도록 해야 한다.

### 12.3 최적화 목표를 AP 하나로 한정하지 않기

이번 프로젝트의 목적은 AP만 높이는 것이 아니다.

최종 운영 목표는 허용 가능한 FPR 수준에서 Recall을 최대한 높이는 것이다.

| 평가 단계 | 주요 지표 | 목적 |
|---|---|---|
| 모델 탐색 | AP | 위험·정상 데이터 구분 능력 비교 |
| Threshold 결정 | FPR, Recall | 오경보와 미탐의 균형 조정 |
| 최종 평가 | AP, FPR, Recall, Precision | 모델의 종합 성능 확인 |

AP를 기준으로 후보 모델을 탐색하되, 최종 선정은 Validation에서 결정한 Threshold를 Test에 적용하여 판단하는 방식이 적절하다.

---

## 13. TMS 분류 모델 Upgrade 실험 로드맵

위의 방법을 모두 한꺼번에 적용하면 어떤 방법이 성능 개선에 기여했는지 알기 어렵다.

따라서 다음과 같이 단계별 실험을 진행하는 것을 권장한다.

### 1단계. 기존 모델 진단

- 기존 모델의 Threshold별 FPR·Recall 측정
- Validation에서 FPR 1%, 2%, 5%에 해당하는 Threshold 확인
- AP, Precision, Recall, FPR 비교
- 예측 점수 분포 확인
- 과적합 여부 및 데이터 분포 확인

### 2단계. 데이터 불균형 개선

- 기존 모델(Baseline)
- Class Weight 적용
- SMOTE 적용
- Class Weight + SMOTE 적용

네 가지 실험 결과를 비교한다.

### 3단계. Feature 개선

- 최근 1시간·3시간 농도 변화량
- 이동평균 및 최대값
- 농도 변동성
- 기존 24시간 Feature의 효과 재검증
- Feature Ablation 실험

Feature 조합별로 모델을 다시 학습한다.

### 4단계. 하이퍼파라미터 탐색

- ValidationCurve로 주요 파라미터 영향 분석
- ParameterGrid로 탐색 조합 정의
- GridSearchCV로 기본 조합 비교
- RandomizedSearchCV로 넓은 범위 탐색
- Optuna를 이용한 Bayesian 계열 탐색
- Parallel Processing으로 실험 시간 단축

### 5단계. 모델 알고리즘 비교

- Random Forest
- XGBoost
- LightGBM
- 필요시 LSTM

동일한 데이터와 평가 기준으로 성능을 비교한다.

### 6단계. 최종 모델 선정

- 동일한 Test 데이터에서 평가
- 목표 FPR 조건에서 Recall 비교
- Precision 및 AP 확인
- 추론시간과 운영 비용 확인
- 데이터 기간이 달라져도 성능이 유지되는지 확인

---

## 14. 실험 결과 관리 기준

모든 실험은 동일한 데이터 분할과 평가 기준으로 비교해야 한다.

### 14.1 모델 개선 실험 비교표

| 실험 모델 | AP | Recall | FPR | Precision | 학습 시간 |
|---|---:|---:|---:|---:|---:|
| Baseline | 측정 | 측정 | 측정 | 측정 | 측정 |
| Class Weight | 측정 | 측정 | 측정 | 측정 | 측정 |
| SMOTE | 측정 | 측정 | 측정 | 측정 | 측정 |
| Feature 개선 | 측정 | 측정 | 측정 | 측정 | 측정 |
| GridSearchCV | 측정 | 측정 | 측정 | 측정 | 측정 |
| Optuna | 측정 | 측정 | 측정 | 측정 | 측정 |
| LightGBM | 측정 | 측정 | 측정 | 측정 | 측정 |

### 14.2 파라미터 탐색 방법 비교표

| 탐색 방법 | 탐색 횟수 | 최적 AP | Recall | FPR | 소요 시간 |
|---|---:|---:|---:|---:|---:|
| GridSearchCV | 측정 | 측정 | 측정 | 측정 | 측정 |
| RandomizedSearchCV | 측정 | 측정 | 측정 | 측정 | 측정 |
| Optuna | 측정 | 측정 | 측정 | 측정 | 측정 |
| Optuna + Parallel | 측정 | 측정 | 측정 | 측정 | 측정 |

이 표를 통해 단순히 어떤 탐색 방법이 높은 AP를 찾았는지뿐 아니라, 동일한 시간과 자원에서 얼마나 효율적으로 좋은 모델을 찾았는지도 비교할 수 있다.

**실험 결과 관리 시 함께 기록할 항목**

- 데이터 Snapshot ID
- Train / Validation / Test 기간
- 데이터 건수 및 위험 데이터 비율
- Feature 구성
- 모델 및 파라미터 설정
- Random Seed
- Threshold
- 평가 지표
- 학습 및 추론 시간

특히 국내 K-TMS 평가 데이터가 약 13일 수준이므로, 이 기간의 성능만으로 모델 개선 효과를 확정하기에는 한계가 있다.

평가 기간을 늘리고 날짜가 바뀌어도 개선 효과가 유지되는지 확인하는 것이 중요하다.

---

## 15. 최종 의견 및 멘토링 방향

이번 TMS 프로젝트에서는 처음부터 LSTM 같은 복잡한 모델로 바꾸거나 SMOTE를 적용하는 것보다 다음 세 가지를 먼저 진행하는 것이 좋겠다.

### 첫째. Threshold 최적화

현재 국내 모델의 FPR 1.00%, Recall 46.9%가 어느 정도 개선 가능한지 확인한다.

동일한 FPR 조건에서 국내 모델의 Recall이 얼마나 개선되는지 확인하여 Threshold 문제와 모델의 구분 능력 문제를 구분한다.

### 둘째. Class Weight와 SMOTE 비교

위험 데이터의 불균형이 실제 성능 저하 원인인지 확인한다.

우선 Class Weight를 적용하고 SMOTE는 비교 실험으로 진행한다.

### 셋째. Feature Engineering과 하이퍼파라미터 튜닝

농도 변화 추세와 시간적 특성을 반영하고, ValidationCurve와 GridSearchCV로 파라미터 영향을 확인한다.

이후 Optuna를 활용해 XGBoost 및 LightGBM의 성능을 비교한다.

### 최종 권장 실험 순서

| 순서 | 실험 항목 | 주요 목적 |
|---|---|---|
| 1 | Baseline 및 Threshold 분석 | 현재 모델의 문제점 진단 |
| 2 | Class Weight | 위험 데이터 학습 강화 |
| 3 | SMOTE | 합성 데이터 활용 효과 확인 |
| 4 | Feature Engineering | 위험 패턴 설명력 개선 |
| 5 | ValidationCurve | 주요 파라미터 영향 분석 |
| 6 | GridSearchCV | 기본 파라미터 조합 탐색 |
| 7 | Optuna | 효율적인 파라미터 최적화 |
| 8 | LightGBM 비교 | 알고리즘별 성능 비교 |
| 9 | Threshold 재조정 | 운영 목표에 맞는 최종 기준 설정 |

Parallel Processing은 GridSearchCV, RandomizedSearchCV 등에서 실험 시간을 단축하기 위해 함께 적용한다.

### 멘티에게 전달할 핵심 메시지

모델 성능을 개선한다고 해서 반드시 더 복잡한 AI 모델을 사용해야 하는 것은 아니다.

먼저 현재 모델이 왜 위험 상황을 놓치는지 원인을 분석하고, Threshold 조정, 데이터 불균형 개선, Feature Engineering, 파라미터 튜닝을 단계적으로 적용해야 한다.

파라미터 탐색도 단순히 값을 변경하는 것이 아니라 ValidationCurve, GridSearchCV, Bayesian Optimization 등 적절한 방법을 선택하고 실험 결과를 비교해야 한다.

중요한 것은 여러 기법을 적용했다는 사실이 아니라, **어떤 방법이 실제 위험 탐지율을 얼마나 개선했으며 오경보는 얼마나 증가했는지를 실험으로 증명하는 것**이다.

이번 프로젝트의 최종 목표는 단순히 모델의 점수를 높이는 것이 아니라, 실제 TMS 운영 환경에서 사용할 수 있는 최적의 모델을 찾는 것이다.
