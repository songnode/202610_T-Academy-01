# 국내 TMS·대만 CEMS 데이터 통합 및 AI 시계열 예측 실습 가이드

## 1. 프로젝트 개요

### 1.1 실습 목적

이번 실습에서는 국내 TMS와 대만 CEMS 데이터를 활용하여 데이터 통합부터 AI 시계열 예측 모델 개발까지 전체 과정을 경험합니다.

서로 다른 국가에서 수집된 데이터를 하나의 데이터셋으로 통합하고, 데이터 품질을 확인한 다음 과거 측정값을 이용하여 미래 배출 농도를 예측합니다.

예측 모델은 머신러닝과 딥러닝을 함께 사용하고, 이미지에서 연기를 찾는 비전 모델을 더합니다.

- 머신러닝: Random Forest, XGBoost
- 딥러닝: LSTM
- 비전: YOLO11n (D-Fire 데이터셋으로 연기·불꽃 바운딩박스 탐지)

머신러닝과 딥러닝 모델은 1시간 후 배출 농도를 예측합니다. 비전 모델은 농도를 예측하지 않고, 사진 속 연기의 위치와 신뢰도를 찾습니다.

연기가 탐지되면 미리 연결해 둔 TMS 데이터를 보여 주고, 위의 예측 모델로 1시간 후 농도를 추정합니다. 이 연계 과정은 AI Agent가 맡습니다.

- 연계: AI Agent (도구 호출 순서 제어)

### 1.2 전체 작업 흐름

```text
  국내 TMS 데이터          대만 CEMS 데이터
        │                        │
        ▼                        ▼
    원천 데이터 확인 및 구조 분석
                 │
                 ▼
       컬럼 및 측정 항목 표준화
                 │
                 ▼
        국내 + 대만 데이터 통합
                 │
                 ▼
          데이터 품질 관리
                 │
                 ▼
        시간 단위 및 데이터 정리
                 │
                 ▼
          피처 엔지니어링
                 │
                 ▼
       학습 데이터 생성 및 분리
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
  머신러닝 모델          딥러닝 모델
       │                    │
  Random Forest            LSTM
  XGBoost                   │
       │                    │
       └─────────┬──────────┘
                 ▼
             모델 평가
                 │
                 ▼
       예측 결과 및 성능 비교
                 │
                 │  (승인 모델을 읽기 전용으로 사용)
                 ▼
  D-Fire 이미지 ──▶ YOLO11n 연기 탐지
                 │
                 ▼
         이미지 이상 판정(규칙)
                 │ 연기 탐지
                 ▼
  AI Agent: 매핑표 조회 → TMS 데이터 표시
            → 1시간 후 농도 추정 → 설명
```

### 1.3 최종 구현 목표

이번 실습을 통해 다음 내용을 구현합니다.

1. 국내 TMS와 대만 CEMS 원천 데이터 구조를 분석합니다.
2. 서로 다른 컬럼을 공통 데이터 구조로 변환합니다.
3. 국내와 대만 데이터를 하나의 통합 데이터셋으로 구성합니다.
4. 결측, 중복, 이상값 등 데이터 품질을 확인합니다.
5. 과거 측정값을 활용하여 예측용 피처를 생성합니다.
6. Random Forest와 XGBoost 모델을 학습합니다.
7. LSTM 딥러닝 모델을 학습합니다.
8. 모델별 예측 결과와 성능을 비교합니다.
9. D-Fire 데이터셋으로 YOLO11n 연기 바운딩박스 탐지 모델을 학습합니다.
10. 연기가 탐지되면 AI Agent가 연결된 TMS 데이터를 보여 주고 1시간 후 농도를 추정합니다.

---

## 2. 원천 데이터 확인

먼저 국내 TMS와 대만 CEMS의 원본 데이터를 확인합니다.

데이터를 통합하기 전에 컬럼, 측정 항목, 시간 형식, 측정 주기 및 단위를 파악해야 합니다.

### 2.1 국내 TMS 데이터

국내 TMS 데이터에서 다음 항목을 확인합니다.

| 항목 | 설명 |
|---|---|
| 사업장 정보 | 사업장 식별자 및 사업장명 |
| 배출구 정보 | 굴뚝 또는 배출구 식별자 |
| 측정 항목 | NOx, SO2, 먼지 등 |
| 측정 시각 | 측정값이 기록된 날짜와 시간 |
| 측정값 | 실제 측정된 수치 |
| 측정 단위 | ppm, mg/Sm³ 등 원본에 기재된 단위 |
| 상태 정보 | 정상, 결측, 점검 등 |
| 측정 주기 | 실제 데이터의 측정 간격 |

원본 데이터에 없는 항목은 임의로 생성하지 않습니다.

### 2.2 대만 CEMS 데이터

대만 환경부에서 제공하는 `AQX_P_186` 데이터는 15분 단위 CEMS 측정 데이터입니다.

주요 필드는 다음과 같습니다.

| 필드 | 설명 |
|---|---|
| `epb_county` | 관할 지역 |
| `cno` | 사업장 식별 정보 |
| `abbr` | 사업장 약칭 |
| `polno` | 오염물질 코드 |
| `itemdesc` | 측정 항목 설명 |
| `item` | 측정 항목 |
| `m_time` | 측정 시각 |
| `m_value` | 측정값 |
| `std` | 관련 기준값 |
| `unit` | 측정 단위 |
| `code2` | 측정 데이터 상태 코드 |
| `code2desc` | 상태 코드 설명 |

**데이터 출처**

- 대만 환경부 CEMS: https://data.moenv.gov.tw/dataset/detail/AQX_P_186
- 대만 환경부 API 안내: https://data.moenv.gov.tw/paradigm

### 2.3 데이터 구조 확인

Python으로 파일을 읽고 기본 정보를 확인합니다.

```python
import pandas as pd

df = pd.read_csv("source_data.csv")

# 데이터 크기
print(df.shape)

# 컬럼과 데이터 타입
print(df.info())

# 데이터 샘플
print(df.head())

# 결측 건수
print(df.isnull().sum())

# 컬럼별 고유값
print(df.nunique())
```

다음 내용을 확인하고 정리합니다.

1. 전체 데이터 건수
2. 컬럼명과 데이터 타입
3. 측정 항목 종류
4. 측정 단위
5. 측정 주기
6. 데이터 기간
7. 결측 및 중복 여부
8. 상태 코드 종류

**작업 결과:** 국내 TMS와 대만 CEMS의 데이터 구조를 파악합니다.

---

## 3. 데이터 항목 및 단위 통일

국내 TMS와 대만 CEMS의 컬럼명이 다르더라도 동일한 의미의 데이터라면 공통 컬럼으로 변환할 수 있습니다.

다만 측정값의 단위가 다를 수 있으므로 단위도 함께 확인합니다.

### 3.1 측정 항목 통일

예를 들어 다음과 같이 항목을 통일합니다.

| 국내 데이터 | 대만 데이터 | 공통 항목 |
|---|---|---|
| 질소산화물 | NOx | NOX |
| 황산화물 | SO2 | SO2 |
| 먼지 | Dust | DUST |

실제 원본의 항목 코드와 설명을 확인한 후 매핑합니다.

이름이 비슷하다는 이유만으로 같은 항목으로 처리하지 않습니다.

### 3.2 측정 단위 확인

대표적인 측정 단위는 다음과 같습니다.

| 단위 | 의미 |
|---|---|
| ppm | 백만분율 |
| ppb | 십억분율 |
| mg/Sm³ | 표준 상태 기준 부피당 질량 |
| mg/Nm³ | 정해진 기준 상태에서의 부피당 질량 |

국내와 대만 데이터의 단위가 반드시 같지는 않습니다.

따라서 실제 데이터의 단위 컬럼을 확인해야 합니다.

### 3.3 단위 변환

ppm과 ppb는 다음과 같이 변환할 수 있습니다.

```text
1 ppm = 1,000 ppb
```

기체 농도인 ppm을 mg/Nm³로 변환하려면 분자량과 기준 온도·압력 등의 조건이 필요합니다.

0℃, 1기압, 건조 기준을 가정하면 다음 관계를 사용할 수 있습니다.

```text
mg/Nm³ = ppm × 분자량 / 22.414
```

NOx를 NO2 환산 기준으로 표현하는 경우에는 NO2의 분자량을 사용해야 합니다.

먼지와 같은 입자상 물질에는 기체의 ppm 변환식을 적용하지 않습니다.

### 3.4 실습용 단위 통합 기준

이번 프로젝트는 교육용 데이터 분석이므로 다음과 같이 진행합니다.

| 상황 | 실습 처리 |
|---|---|
| 단위가 동일한 데이터 | 그대로 통합 |
| ppm과 ppb처럼 변환이 가능한 데이터 | 변환 후 통합 |
| 측정 조건이 다른 데이터 | 원본 단위를 보존하고 실습용 통합 기준을 명시 |
| 단위 확인이 어려운 데이터 | 원본값을 유지하고 출처 구분 |

실습용으로 데이터를 통합하는 것이므로 원본값을 보존하면서 통합 데이터셋을 만들 수 있습니다.

다만 단위가 다른 숫자를 그대로 섞어 학습하면 모델이 동일한 수치 범위로 오해할 수 있습니다.

따라서 다음 두 가지를 구분합니다.

- 통합 데이터셋 생성: 국내와 대만 데이터를 행 방향으로 이어 붙입니다.
- 모델 입력 데이터 생성: 실습용 단위 기준과 스케일링을 적용합니다.

### 3.5 측정 조건 확인

단위가 같아도 다음 조건이 다르면 측정값의 의미가 달라질 수 있습니다.

- 건조 기준과 습윤 기준
- 산소 보정 여부
- 기준 온도와 압력
- NOx의 NO2 환산 여부

이번 실습에서는 원본 측정값과 단위를 보존하고, 확인되지 않은 조건은 임의로 보정하지 않습니다.

**작업 결과:** 측정 항목 매핑표와 단위 확인 자료를 확보합니다.

---

## 4. 공통 데이터 구조 설계

국내와 대만 데이터를 하나의 데이터셋으로 합치기 위해 공통 컬럼을 정의합니다.

### 4.1 공통 컬럼

| 컬럼 | 설명 |
|---|---|
| `data_origin` | 데이터 출처: KR 또는 TW |
| `source` | 원천 시스템 |
| `entity_key` | 사업장·배출구 식별자 |
| `item_code` | 측정 항목 |
| `measured_at` | 측정 시각 |
| `raw_value` | 원본 측정값 |
| `raw_unit` | 원본 단위 |
| `normalized_value` | 실습용 변환값 |
| `normalized_unit` | 변환 후 단위 |
| `record_state` | 데이터 상태 |
| `ingested_at` | 데이터 수집 시각 |

### 4.2 데이터 출처 구분

국내와 대만 데이터를 합치더라도 어느 국가에서 수집된 데이터인지 구분할 수 있어야 합니다.

다음과 같이 관리합니다.

| data_origin | source | entity_key |
|---|---|---|
| KR | KOREA_TMS | KR_001_STACK_01 |
| TW | TAIWAN_CEMS | TW_001_STACK_01 |

`entity_key`는 각 국가의 원본 사업장 및 배출구 식별 정보를 이용하여 구성합니다.

원본에 배출구 정보가 없다면 확인 가능한 범위까지만 식별자로 사용합니다.

### 4.3 공통 데이터 변환 예시

```python
import pandas as pd

# 국내 데이터
kr = pd.DataFrame({
    "data_origin": "KR",
    "source": "KOREA_TMS",
    "entity_key": "KR_001_STACK_01",
    "item_code": "NOX",
    "measured_at": ["2026-01-01 00:00:00"],
    "raw_value": [25.3],
    "raw_unit": ["ppm"]
})

# 대만 데이터
tw = pd.DataFrame({
    "data_origin": "TW",
    "source": "TAIWAN_CEMS",
    "entity_key": "TW_001_STACK_01",
    "item_code": "NOX",
    "measured_at": ["2026-01-01 00:00:00"],
    "raw_value": [27.1],
    "raw_unit": ["ppm"]
})

# 데이터 타입 변환
kr["measured_at"] = pd.to_datetime(kr["measured_at"])
tw["measured_at"] = pd.to_datetime(tw["measured_at"])

# 공통 구조로 통합
combined = pd.concat(
    [kr, tw],
    ignore_index=True
)

print(combined.head())
```

위 코드는 데이터 구조를 설명하기 위한 예시입니다. 실제 데이터에서는 원본 컬럼을 공통 컬럼으로 매핑해야 합니다.

### 4.4 통합 데이터 확인

```python
# 국가별 데이터 건수
print(combined["data_origin"].value_counts())

# 측정 항목별 데이터 건수
print(combined["item_code"].value_counts())

# 국가별 측정 항목 건수
print(
    combined.groupby(
        ["data_origin", "item_code"]
    ).size()
)
```

이 단계에서 국내와 대만 데이터가 하나의 DataFrame으로 통합되었는지 확인합니다.

**중요:** 데이터를 합친다는 것은 두 데이터의 행을 이어 붙이는 것입니다. 국내와 대만의 측정값을 서로 더하는 것이 아닙니다.

**작업 결과:** 국내와 대만 데이터가 포함된 통합 데이터셋을 생성합니다.

---

## 5. 데이터 품질 관리

통합 데이터셋을 만든 다음에는 데이터 품질을 확인합니다.

측정값이 존재하더라도 모두 정상적인 데이터는 아닙니다.

결측, 중복, 이상값 및 측정 상태를 확인해야 합니다.

### 5.1 결측 데이터 확인

```python
missing_count = combined["raw_value"].isnull().sum()

missing_rate = (
    missing_count / len(combined) * 100
)

print("결측 건수:", missing_count)
print("결측률:", missing_rate)
```

측정값이 없다는 이유로 0을 입력하면 안 됩니다.

실제 측정값 0과 측정하지 못한 결측값은 의미가 다릅니다.

### 5.2 중복 데이터 확인

동일 사업장, 배출구, 측정 항목, 측정 시각에 중복 데이터가 존재하는지 확인합니다.

```python
key_columns = [
    "data_origin",
    "source",
    "entity_key",
    "item_code",
    "measured_at"
]

duplicates = combined[
    combined.duplicated(
        subset=key_columns,
        keep=False
    )
]

print(duplicates)
```

중복이 발견되면 원인을 확인한 후 처리합니다.

### 5.3 측정값 분포 확인

```python
print(combined["raw_value"].describe())
```

최소값, 최대값, 평균, 표준편차 등을 확인합니다.

다음과 같은 데이터를 살펴봅니다.

- 음수 측정값
- 비정상적으로 큰 값
- 갑자기 급격하게 변한 값
- 장시간 반복되는 동일한 값

단, 값이 크다는 이유만으로 오류 처리하면 안 됩니다. 실제 기준 초과 측정값일 수 있습니다.

### 5.4 대만 CEMS 상태 코드 확인

대만 CEMS의 `code2`와 `code2desc`는 측정 데이터의 상태를 판단하는 데 사용합니다.

프로젝트에서는 다음과 같은 내부 상태로 분류할 수 있습니다.

| 내부 상태 | 의미 | 처리 |
|---|---|---|
| `VALID_NORMAL` | 정상 측정값 | 분석 사용 |
| `VALID_EXCEED` | 기준 초과 측정값 | 분석 사용 |
| `INVALID_TEST` | 시험·교정 데이터 | 일반 분석 제외 |
| `INVALID_INSPECTION` | 검사 관련 데이터 | 별도 관리 |
| `INVALID_MAINTENANCE` | 유지보수 데이터 | 일반 분석 제외 |
| `INVALID_EQUIPMENT` | 장비 이상 | 별도 관리 |
| `INVALID_DATA` | 유효하지 않은 데이터 | 일반 분석 제외 |
| `MISSING` | 결측 데이터 | 결측 관리 |
| `SUBSTITUTED` | 대체 데이터 | 별도 구분 |
| `UNKNOWN` | 확인되지 않은 상태 | 확인 필요 |

실제 상태 코드와 내부 상태의 연결은 공식 코드 설명 및 원본 `code2desc`를 확인한 후 결정합니다.

기준 초과 측정값은 실제 측정값일 수 있으므로 정상적인 데이터와 구분하되 무조건 삭제하지 않습니다.

### 5.5 분석용 데이터 생성

```python
valid_data = combined[
    combined["record_state"].isin([
        "VALID_NORMAL",
        "VALID_EXCEED"
    ])
].copy()
```

이 예시는 유효한 측정 상태만 분석에 사용하는 방식입니다.

국내 데이터도 원본 상태 정보를 확인한 후 동일한 기준으로 내부 상태를 지정합니다.

**작업 결과:** 통합 데이터의 결측, 중복, 이상값 및 상태 코드 확인을 완료합니다.

---

## 6. 시간 단위 통일

국내와 대만의 측정 주기가 다를 수 있으므로 공통 시간 단위를 정합니다.

이번 실습에서는 국내 데이터가 30분 단위이고 대만 데이터가 15분 단위인 경우를 예로 들어 설명합니다.

### 6.1 15분 데이터를 30분으로 변환

대만의 15분 데이터를 30분 단위로 변환할 수 있습니다.

예를 들어 다음과 같습니다.

```text
10:00 = 20 ppm
10:15 = 24 ppm

30분 대표값
= (20 + 24) / 2
= 22 ppm
```

이 방식은 두 15분 데이터가 동일한 측정 조건에서 산출된 구간 평균이라는 전제의 실습용 집계 방법입니다.

### 6.2 변환 시 확인 사항

1. 동일 사업장과 배출구인지 확인합니다.
2. 동일 측정 항목인지 확인합니다.
3. 단위가 동일한지 확인합니다.
4. 두 측정값이 모두 유효한지 확인합니다.
5. 중복 데이터가 없는지 확인합니다.
6. 측정 시각이 예상한 15분 간격인지 확인합니다.

한 구간이 누락되거나 유효하지 않다면 임의로 30분 평균을 만들지 않습니다.

원본 15분 데이터는 그대로 보존합니다.

### 6.3 집계 코드 예시

```python
import pandas as pd

tw = tw.sort_values("measured_at").copy()

tw["measured_at"] = pd.to_datetime(
    tw["measured_at"]
)

tw["time_30m"] = (
    tw["measured_at"].dt.floor("30min")
)

group_columns = [
    "data_origin",
    "source",
    "entity_key",
    "item_code",
    "raw_unit",
    "time_30m"
]

result = (
    tw.groupby(group_columns)
    .agg(
        valid_count=("raw_value", "count"),
        value_mean=("raw_value", "mean")
    )
    .reset_index()
)

# 두 건이 있는 구간만 우선 선택
result = result[
    result["valid_count"] == 2
]
```

실제 작업에서는 30분 구간에 서로 다른 두 개의 15분 측정값이 정확하게 존재하는지도 확인해야 합니다.

두 건이라는 조건만으로 시간 간격과 데이터 품질이 모두 정상이라고 판단할 수는 없습니다.

또한 유효 측정 건수가 다르거나 구간별 집계 기준이 다르면 단순 평균이 적절하지 않을 수 있습니다.

### 6.4 시간대 통일

한국은 UTC+9, 대만은 UTC+8입니다.

국가 간 시간 비교가 필요하다면 시간대를 확인하고 공통 시간대로 변환합니다.

원본 시간은 보존하고 변환된 시간을 별도 컬럼으로 관리하는 것이 좋습니다.

**작업 결과:** 분석에 사용할 시간 단위로 정리된 데이터를 생성합니다.

---

## 7. 피처 엔지니어링

피처 엔지니어링은 원본 데이터를 AI 모델이 학습할 수 있는 입력 변수로 만드는 과정입니다.

이번 실습에서는 과거 NOx 측정값을 이용하여 60분 후의 NOx 농도를 예측합니다.

### 7.1 예측 목표 정의

```text
현재 시점의 측정값을 이용하여
60분 후의 NOx 농도를 예측한다.
```

예를 들어 10:00 데이터를 이용하여 11:00의 NOx 농도를 예측합니다.

### 7.2 입력 피처 구성

| 피처 | 설명 |
|---|---|
| `nox_current` | 현재 NOx 값 |
| `nox_lag_1` | 30분 전 NOx |
| `nox_lag_2` | 60분 전 NOx |
| `nox_lag_3` | 90분 전 NOx |
| `nox_mean_3` | 과거 3개 구간 평균 |
| `nox_mean_6` | 과거 6개 구간 평균 |
| `hour` | 시간 |
| `day_of_week` | 요일 |
| `month` | 월 |
| `data_origin` | 국내 또는 대만 구분 |

실제 데이터에 산소 농도, 배출 유량, 온도 등 추가 측정값이 있다면 품질을 확인한 후 피처로 사용할 수 있습니다.

### 7.3 Lag 피처 생성

Lag는 과거 측정값을 현재 시점의 입력 변수로 사용하는 방법입니다.

```text
현재 시각: 10:00

nox_current = 10:00 측정값
nox_lag_1   = 09:30 측정값
nox_lag_2   = 09:00 측정값
nox_lag_3   = 08:30 측정값
```

코드 예시입니다.

```python
group_columns = [
    "data_origin",
    "source",
    "entity_key",
    "item_code"
]

df = df.sort_values(
    group_columns + ["measured_at"]
)

grouped = df.groupby(
    group_columns
)["normalized_value"]

df["nox_lag_1"] = grouped.shift(1)
df["nox_lag_2"] = grouped.shift(2)
df["nox_lag_3"] = grouped.shift(3)
```

이 코드는 이전 행을 가져옵니다.

따라서 이전 행이 실제로 30분 전 데이터인지 확인해야 합니다. 시간 누락이 있다면 연속된 시간 구간을 확인한 후 사용합니다.

### 7.4 Rolling 피처 생성

Rolling은 일정 구간의 평균이나 변동성을 계산합니다.

```python
df["nox_mean_3"] = (
    df.groupby(group_columns)["normalized_value"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=3,
            min_periods=3
        ).mean()
    )
)
```

위 코드는 현재 시점의 값을 제외하고 이전 3개 행의 평균을 계산합니다.

실제 90분 평균으로 사용하려면 데이터가 연속된 30분 간격인지 확인해야 합니다.

### 7.5 시간 피처 생성

```python
df["hour"] = df["measured_at"].dt.hour

df["day_of_week"] = (
    df["measured_at"].dt.dayofweek
)

df["month"] = df["measured_at"].dt.month
```

시간의 주기적 특성을 반영하려면 다음과 같이 변환할 수 있습니다.

```python
import numpy as np

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)
```

### 7.6 예측 대상 생성

30분 단위 데이터에서 60분 후 값을 예측하려면 2개 구간 이후의 값을 Target으로 사용합니다.

```python
df["target_nox_60m"] = (
    df.groupby(group_columns)["normalized_value"]
    .shift(-2)
)
```

이 코드는 데이터가 연속된 30분 간격이라는 전제가 필요합니다.

현재 시각과 Target 시각의 차이가 실제로 60분인지 확인합니다.

Target 값이 결측이거나 유효하지 않은 상태라면 학습 데이터에서 제외합니다.

### 7.7 데이터 누수 방지

예측 시점 이후의 데이터를 입력 피처로 사용하면 안 됩니다.

예를 들어 10:00 시점의 예측을 수행하면서 10:30 또는 11:00의 실제 측정값을 입력 변수로 사용하면 데이터 누수가 발생합니다.

- Lag: 과거 측정값만 사용
- Rolling: 과거 데이터만 사용
- Target: 미래의 실제 측정값
- 시간 피처: 예측 시점에 이미 알 수 있는 정보

**작업 결과:** 머신러닝과 딥러닝 학습에 활용할 수 있는 피처 및 Target 데이터를 생성합니다.

---

## 8. 학습 데이터 분리

모델 학습 전에 데이터를 Train, Validation, Test로 분리합니다.

시계열 데이터는 시간 순서가 중요하므로 무작위로 섞지 않습니다.

### 8.1 데이터 분리 기준

| 구분 | 비율 | 용도 |
|---|---:|---|
| Train | 70% | 모델 학습 |
| Validation | 15% | 모델 설정 및 비교 |
| Test | 15% | 최종 성능 평가 |

예를 들어 2023~2025년 데이터가 있다면 과거부터 순서대로 나눕니다.

```text
2023년 ~ 2024년 중반
        │
        ▼
      Train
        │
        ▼
2024년 중반 ~ 후반
        │
        ▼
   Validation
        │
        ▼
2025년
        │
        ▼
      Test
```

실제 데이터 기간과 건수에 따라 분할 기준은 조정합니다.

### 8.2 데이터 누수 방지

다음 사항을 지켜야 합니다.

1. 시간 순서대로 데이터를 분리합니다.
2. 전체 데이터를 섞어서 분리하지 않습니다.
3. 스케일링 기준은 Train 데이터에서만 계산합니다.
4. Validation은 모델 설정에 사용합니다.
5. Test는 최종 평가에만 사용합니다.
6. 예측 대상 시각이 데이터 분할 경계를 넘어 겹치지 않도록 확인합니다.

### 8.3 국내와 대만 데이터 통합 학습

이번 실습에서는 국내와 대만 데이터를 하나로 합쳐 학습합니다.

다만 다음 컬럼은 유지합니다.

- `data_origin`
- `entity_key`
- `item_code`
- `measured_at`

국가별 데이터의 특성을 확인하기 위해 `data_origin`을 모델 입력 피처로 활용할 수 있습니다.

국가별 시설과 측정 환경이 다를 수 있으므로 전체 성능과 국가별 성능을 모두 확인합니다.

---

## 9. 머신러닝 모델 학습

먼저 머신러닝 모델을 학습합니다.

이번 실습에서는 Persistence, Random Forest, XGBoost를 사용합니다.

### 9.1 Persistence 기준 모델

Persistence는 현재 측정값이 미래에도 유지된다고 가정하는 가장 단순한 예측 모델입니다.

```python
baseline_prediction = X_test["nox_current"]
```

복잡한 모델을 만들기 전에 기준 성능을 확인하는 데 사용합니다.

### 9.2 Random Forest

Random Forest는 여러 개의 의사결정나무를 학습하고 결과를 결합하는 모델입니다.

주요 특징은 다음과 같습니다.

- 비선형 관계 학습
- 여러 피처의 조합 활용
- 비교적 간단한 학습 과정
- 변수 중요도 확인 가능

학습 예시입니다.

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
```

### 9.3 XGBoost

XGBoost는 여러 모델을 순차적으로 학습하면서 예측 성능을 개선하는 부스팅 계열 모델입니다.

주요 특징은 다음과 같습니다.

- 복잡한 변수 관계 학습
- 다양한 파라미터 조정
- 표 형태 데이터에 적합
- 과적합 방지를 위한 설정 필요

학습 예시입니다.

```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
```

### 9.4 머신러닝 모델 학습 결과

다음 항목을 확인합니다.

- Persistence보다 성능이 개선되었는가?
- Random Forest와 XGBoost 중 어떤 모델이 좋은가?
- 어떤 피처가 예측에 중요한 영향을 주는가?
- 국내와 대만 데이터의 예측 성능 차이는 어떠한가?

**작업 결과:** Persistence, Random Forest, XGBoost 모델과 예측 결과를 확보합니다.

---

## 10. 딥러닝 LSTM 모델 학습

머신러닝 모델 학습이 끝나면 LSTM 딥러닝 모델을 학습합니다.

LSTM은 시계열 데이터의 시간적 흐름과 과거 패턴을 학습할 수 있는 순환 신경망 계열 모델입니다.

Random Forest와 XGBoost가 각 행의 피처를 이용하여 예측한다면, LSTM은 일정 기간의 연속된 데이터를 입력으로 받아 시간 흐름을 학습합니다.

### 10.1 LSTM의 기본 개념

예를 들어 다음과 같은 데이터가 있다고 가정합니다.

```text
08:00 NOx = 20
08:30 NOx = 22
09:00 NOx = 24
09:30 NOx = 25
10:00 NOx = 27
```

LSTM은 이처럼 시간 순서로 연결된 데이터를 입력받습니다.

과거 데이터의 변화 흐름을 학습하여 미래 값을 예측합니다.

### 10.2 LSTM 입력 데이터 구조

일반적인 머신러닝 모델은 다음과 같은 2차원 데이터를 사용합니다.

```text
[샘플 수, 피처 수]
```

LSTM은 다음과 같은 3차원 데이터 구조를 사용합니다.

```text
[샘플 수, 시간 단계, 피처 수]
```

예를 들어 다음과 같습니다.

```text
X_train.shape = (10000, 12, 5)
```

의미는 다음과 같습니다.

- 10,000: 학습 샘플 수
- 12: 과거 12개 시간 구간
- 5: 각 시간 구간에서 사용하는 피처 수

30분 단위 데이터에서 12개 구간은 과거 6시간에 해당합니다.

### 10.3 LSTM 입력 시퀀스 생성

과거 12개 구간을 입력으로 사용하고, 60분 후의 NOx 농도를 예측하는 예시입니다.

```python
import numpy as np

def make_sequences(X, y, window_size=12):
    X_seq = []
    y_seq = []

    for i in range(window_size, len(X)):
        X_seq.append(
            X[i-window_size:i]
        )

        y_seq.append(
            y[i]
        )

    return (
        np.array(X_seq),
        np.array(y_seq)
    )
```

사용 예시는 다음과 같습니다.

```python
X_seq, y_seq = make_sequences(
    X_scaled,
    y_scaled,
    window_size=12
)

print(X_seq.shape)
print(y_seq.shape)
```

이 함수는 하나의 연속된 시계열을 대상으로 하는 기본 예시입니다.

실제 통합 데이터에서는 다음 조건을 지켜야 합니다.

1. 국가별 데이터를 구분합니다.
2. 사업장·배출구별 데이터를 구분합니다.
3. 측정 항목별 데이터를 구분합니다.
4. 시간 순서대로 정렬합니다.
5. 서로 다른 사업장이나 국가의 데이터가 하나의 시퀀스로 연결되지 않도록 합니다.
6. 중간에 누락된 시간 구간이 있다면 연속된 시퀀스로 취급하지 않습니다.

예를 들어 한국 사업장의 마지막 데이터 다음에 대만 사업장의 첫 데이터가 이어지도록 시퀀스를 만들면 안 됩니다.

### 10.4 데이터 스케일링

LSTM은 입력 데이터의 크기 차이에 영향을 받을 수 있으므로 일반적으로 스케일링을 적용합니다.

대표적으로 MinMaxScaler를 사용할 수 있습니다.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Train 데이터로만 스케일러 학습
X_train_scaled = scaler.fit_transform(X_train)

# Validation과 Test는 Train 기준 적용
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

Target 값도 스케일링하는 경우 별도의 스케일러를 사용합니다.

예측 결과를 실제 단위로 비교할 때는 역변환해야 합니다.

중요한 점은 스케일러를 전체 데이터에 먼저 적용하지 않는 것입니다. 반드시 Train 데이터에서 기준을 학습한 다음 Validation과 Test에 적용합니다.

### 10.5 LSTM 모델 구성

TensorFlow Keras를 이용한 기본 모델 예시입니다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(
        64,
        input_shape=(12, 5)
    ),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()
```

위 예시는 입력 피처가 5개이고 과거 12개 구간을 사용하는 경우입니다.

실제 모델에서는 사용하는 피처 수에 맞게 입력 형태를 변경합니다.

### 10.6 LSTM 학습

```python
history = model.fit(
    X_train_seq,
    y_train_seq,
    validation_data=(
        X_val_seq,
        y_val_seq
    ),
    epochs=30,
    batch_size=64,
    shuffle=False
)
```

시계열 데이터에서는 시간 순서가 중요하므로 `shuffle=False`로 설정합니다.

학습 과정에서 Train과 Validation의 Loss 변화를 확인합니다.

- Train Loss와 Validation Loss가 함께 감소하는지 확인합니다.
- Train Loss만 감소하고 Validation Loss가 증가하면 과적합을 의심합니다.
- Epoch 수를 무조건 늘리는 것이 좋은 것은 아닙니다.

필요하면 EarlyStopping을 적용하여 과적합을 줄일 수 있습니다.

### 10.7 LSTM 예측

```python
lstm_pred_scaled = model.predict(
    X_test_seq
)
```

Target을 스케일링했다면 실제 단위로 역변환합니다.

```python
lstm_pred = target_scaler.inverse_transform(
    lstm_pred_scaled
)
```

실제값도 동일한 기준으로 역변환하여 평가합니다.

### 10.8 LSTM 실습 시 주의사항

- 한 시퀀스에는 동일한 사업장·배출구·측정 항목의 데이터만 포함합니다.
- 국가나 사업장이 바뀌는 경계에서 시퀀스를 연결하지 않습니다.
- 시간 간격이 일정한 데이터로 시퀀스를 구성합니다.
- Train, Validation, Test 순서를 유지합니다.
- 스케일러는 Train 데이터로만 학습합니다.
- 예측 결과는 원래 측정 단위로 역변환하여 평가합니다.

**작업 결과:** 과거 시계열 패턴을 학습한 LSTM 모델과 예측 결과를 확보합니다.

---

## 11. 모델 성능 평가 및 비교

이제 머신러닝과 딥러닝 모델의 성능을 비교합니다.

이번 실습에서는 다음 모델을 비교합니다.

| 구분 | 모델 | 설명 |
|---|---|---|
| 기준 모델 | Persistence | 현재 값 유지 가정 |
| 머신러닝 | Random Forest | 의사결정나무 앙상블 |
| 머신러닝 | XGBoost | 부스팅 기반 예측 |
| 딥러닝 | LSTM | 시계열 패턴 학습 |

### 11.1 MAE

MAE는 실제값과 예측값 사이의 절대 오차 평균입니다.

값이 작을수록 예측 오차가 작습니다.

### 11.2 RMSE

RMSE는 큰 오차에 더 큰 영향을 받는 평가 지표입니다.

일부 시간대에서 큰 예측 오차가 발생하는지 확인할 때 유용합니다.

### 11.3 R²

R²는 실제 데이터의 변동을 모델이 얼마나 설명하는지 나타냅니다.

일반적으로 1에 가까울수록 설명력이 높습니다.

다만 R²만으로 모델 성능을 판단하지 말고 MAE와 RMSE를 함께 확인합니다.

### 11.4 평가 코드

```python
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
```

모델별 결과를 비교합니다.

```python
results = {
    "Random Forest": evaluate_model(
        y_test, rf_pred
    ),

    "XGBoost": evaluate_model(
        y_test, xgb_pred
    ),

    "LSTM": evaluate_model(
        y_test_lstm, lstm_pred
    )
}

for name, metrics in results.items():
    print(name, metrics)
```

모델별 평가 데이터의 측정 시각과 실제 Target이 동일한지 확인해야 합니다.

LSTM은 시퀀스 생성 과정에서 앞부분 데이터가 제외될 수 있으므로 Random Forest와 XGBoost의 테스트 행과 단순히 인덱스를 맞추면 안 됩니다.

### 11.5 국가별 성능 비교

통합 데이터의 전체 성능뿐 아니라 국내와 대만의 성능을 따로 확인합니다.

| 평가 구분 | 확인 목적 |
|---|---|
| 전체 데이터 | 통합 모델의 전반적인 성능 |
| 국내 TMS | 국내 데이터 예측 성능 |
| 대만 CEMS | 대만 데이터 예측 성능 |
| 측정 항목별 | NOx, SO2 등 항목별 성능 |

국가별 데이터 특성에 따라 성능 차이가 발생할 수 있습니다.

전체 평균만 확인하지 말고 국가별 결과도 함께 비교합니다.

### 11.6 실제값과 예측값 시각화

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.plot(
    y_test,
    label="Actual"
)

plt.plot(
    rf_pred,
    label="Random Forest"
)

plt.plot(
    xgb_pred,
    label="XGBoost"
)

plt.title("NOx Prediction Comparison")
plt.xlabel("Time")
plt.ylabel("NOx")

plt.legend()
plt.grid(True)
plt.show()
```

LSTM도 동일한 테스트 구간에 맞춰 별도로 비교합니다.

다음 내용을 확인합니다.

- 실제값과 예측값의 흐름이 비슷한가?
- 급격한 증가 구간을 예측하는가?
- 특정 시간대에서 오차가 커지는가?
- 머신러닝과 LSTM 중 어느 모델이 더 좋은가?
- 국내와 대만 데이터의 성능 차이는 어떠한가?

**작업 결과:** 모델별 성능 지표와 실제값·예측값 비교 결과를 확보합니다.

---

## 12. 예측 결과 및 경고 정보 생성

모델 학습과 평가가 끝나면 최종 예측 결과를 정리합니다.

### 12.1 예측 결과 데이터 구조

| 컬럼 | 설명 |
|---|---|
| `entity_key` | 사업장·배출구 식별자 |
| `data_origin` | 국내 또는 대만 |
| `item_code` | 측정 항목 |
| `measured_at` | 현재 측정 시각 |
| `target_at` | 예측 대상 시각 |
| `predicted_value` | 예측값 |
| `model_name` | 모델명 |
| `model_version` | 모델 버전 |
| `prediction_created_at` | 예측 생성 시각 |

실제 측정값은 예측 시점에는 알 수 없으므로, 이후 실제 데이터가 수집되면 예측 결과와 연결하여 평가합니다.

### 12.2 규칙 기반 경고 판정

예측값과 국가별·측정 항목별 기준값을 비교하여 경고 여부를 판단할 수 있습니다.

```text
예측값 < 적용 기준값
       │
       ▼
      정상

예측값 >= 적용 기준값
       │
       ▼
    경고 검토
```

국내와 대만의 기준값은 각각 별도로 관리합니다.

대만의 `std` 값을 국내 법적 기준으로 사용하지 않습니다.

### 12.3 LLM 활용

LLM을 사용하는 경우에는 확정된 예측값과 판정 결과를 사람이 이해하기 쉬운 문장으로 설명하도록 활용할 수 있습니다.

다만 실제 측정 데이터만으로 확인할 수 없는 원인을 임의로 추정하지 않도록 합니다.

---

## 13. D-Fire + YOLO11n 연기 탐지와 AI Agent 연계

이 장에서는 이미지에서 연기를 찾는 모델을 만들고, 연기가 탐지되면 AI Agent가 TMS 데이터와 1시간 후 추정값을 함께 보여 주도록 연결합니다.

### 13.1 먼저 알아 둘 전제

이 실습의 이미지와 TMS 데이터는 **서로 다른 곳에서 수집된 데이터**입니다.

| 구분 | 내용 |
|---|---|
| D-Fire 이미지 | 산불·도시 화재 등 일반 장면의 연기·불꽃 이미지 |
| TMS 데이터 | 국내 TMS·대만 CEMS 굴뚝 측정값 |
| 두 데이터의 관계 | 원래는 없습니다. 같은 굴뚝, 같은 시각에 찍은 이미지가 아닙니다. |

따라서 이미지와 TMS 데이터의 연결은 **시연용 매핑표**로 미리 정해 둡니다.

다음 원칙을 지킵니다.

1. 매핑표는 시연 시나리오이며, 이미지와 측정값 사이의 인과 관계를 뜻하지 않습니다.
2. 탐지 결과를 시계열 모델의 입력 피처나 학습 라벨로 쓰지 않습니다.
3. 화면과 설명에 "시연용 매핑"임을 표시합니다.
4. 연기 탐지는 "시각적 이상 후보"이며, 배출 기준 초과 판정이 아닙니다.

굴뚝에서 나오는 흰 수증기는 연기와 비슷하게 보일 수 있습니다. D-Fire에는 굴뚝 수증기 장면이 거의 없으므로 실제 굴뚝 영상에 적용하면 오탐이 생길 수 있다는 점을 한계로 기록합니다.

### 13.2 D-Fire 데이터셋

D-Fire는 화재·연기 탐지용 공개 이미지 데이터셋입니다. 라벨이 YOLO 형식(클래스, 중심 x, 중심 y, 너비, 높이)으로 제공되어 바로 학습에 쓸 수 있습니다.

| 항목 | 내용 |
|---|---|
| 클래스 | `smoke`, `fire` |
| 이미지 구성 | 연기만, 불꽃만, 연기+불꽃, 아무것도 없는 이미지 |
| 라벨 형식 | 이미지별 `.txt` 파일(YOLO 형식) |
| 분할 | train / test 제공 |

**데이터 출처**

- D-Fire: https://github.com/gaiasd/DFireDataset

사용 전에 다음을 직접 확인합니다.

1. 저장소의 라이선스와 인용 요구 사항
2. 클래스 번호와 이름의 대응(예: 0 = smoke, 1 = fire)
3. 이미지 수와 클래스별 박스 수
4. 라벨이 비어 있는 이미지(배경 이미지)의 수

validation 분할이 따로 없다면 train에서 일부를 떼어 만듭니다. test는 최종 평가에만 씁니다.

#### 13.2.1 분류 모델과 탐지 모델의 차이

YOLO는 사진을 "정상/비정상"으로 나누는 분류 모델이 아닙니다. 사진 속 물체가 **어디에 있는지** 찾는 탐지 모델입니다.

| 구분 | 분류 모델 | YOLO(탐지 모델) |
|---|---|---|
| 학습 라벨 | 사진 한 장에 "정상" 또는 "비정상" | 사진 속 연기·불꽃 위치에 박스 + 클래스 |
| 출력 | 사진 전체의 클래스 하나 | 박스 여러 개(위치, 클래스, 신뢰도) |

D-Fire로 학습한 YOLO의 클래스는 `smoke`와 `fire` 두 개뿐입니다. **"정상"이라는 클래스는 없습니다.**

- 정상(배경) 사진은 라벨 파일이 비어 있는 사진입니다. 모델은 이런 사진에서 "아무것도 없음"을 배워 오탐을 줄입니다.
- 정상/비정상은 학습으로 정하지 않습니다. 추론 결과에 규칙을 적용하여 정합니다(13.6).

#### 13.2.2 바운딩박스 라벨 형식

학습에 쓰는 박스는 사람이 미리 쳐 둔 정답입니다. D-Fire는 박스 라벨이 이미 만들어져 있으므로 직접 칠 필요가 없습니다.

사진마다 같은 이름의 `.txt` 파일이 있고, 한 줄이 박스 하나입니다.

```text
# 클래스  중심x   중심y   너비   높이   (사진 크기 대비 0~1 비율)
0        0.512   0.334   0.420  0.380
1        0.250   0.700   0.100  0.150
```

- 첫 줄: 연기(0) 박스
- 둘째 줄: 불꽃(1) 박스
- 연기와 불꽃이 모두 없는 사진: 빈 파일

좌표는 픽셀이 아니라 사진 너비·높이에 대한 비율입니다.

라벨이 제대로 들어 있는지 학습 전에 몇 장을 그려서 확인합니다.

```python
import cv2

def draw_labels(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path) as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.split())

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img
```

#### 13.2.3 사진을 직접 추가하여 라벨링하기

굴뚝 사진이나 수증기 사진을 추가하여 오탐을 줄이려면 직접 박스를 그려야 합니다.

| 도구 | 특징 |
|---|---|
| Label Studio | 웹 기반, 설치형, YOLO 형식 내보내기 지원 |
| CVAT | 웹 기반, 팀 작업과 검수 기능 |
| Roboflow | 웹 서비스, 증강·분할 기능 포함 |

작업 순서는 다음과 같습니다.

1. 사진을 불러옵니다.
2. 연기 영역을 드래그하여 사각형을 그립니다.
3. 클래스(`smoke` 또는 `fire`)를 고릅니다.
4. 연기·불꽃이 없는 사진은 박스 없이 저장합니다(배경 이미지).
5. YOLO 형식으로 내보내어 `images/`와 `labels/`에 같은 파일명으로 둡니다.

연기는 경계가 흐려서 사람마다 박스 크기가 달라지기 쉽습니다. 라벨링 전에 기준을 정하고 문서로 남깁니다.

| 기준 | 예시 |
|---|---|
| 박스 범위 | 눈으로 확실히 보이는 짙은 부분까지만 포함 |
| 여러 덩어리 | 서로 떨어져 있으면 박스를 나누어 그림 |
| 굴뚝 수증기 | 박스를 그리지 않고 배경으로 둠 |
| 판단이 어려운 사진 | 학습에서 빼고 목록으로 따로 관리 |

추가한 사진은 기존 D-Fire 사진과 섞되, train·val·test에 골고루 나눕니다. 같은 장소에서 연속으로 찍은 사진이 train과 test에 함께 들어가면 성능이 실제보다 좋아 보이므로 장소·시점 단위로 나눕니다.

### 13.3 학습 설정 파일

```yaml
# dfire.yaml
path: data/raw/dfire
train: train/images
val: val/images
test: test/images

names:
  0: smoke
  1: fire
```

클래스 번호는 저장소에서 확인한 값으로 맞춥니다.

### 13.4 YOLO11n 학습

YOLO11n은 Ultralytics YOLO11 계열에서 가장 작은 모델입니다. 학습과 추론이 빨라 실습에 적합합니다.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dfire.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    seed=42
)
```

사전 학습 가중치(`yolo11n.pt`)에서 시작하여 D-Fire로 다시 학습합니다.

Ultralytics 패키지의 라이선스(AGPL-3.0)도 함께 확인합니다.

### 13.5 탐지 성능 평가

```python
metrics = model.val(
    data="dfire.yaml",
    split="test"
)

print(metrics.box.map50)
print(metrics.box.map)
```

| 지표 | 의미 |
|---|---|
| Precision | 탐지한 박스 중 맞은 비율 |
| Recall | 실제 연기 중 찾아낸 비율 |
| mAP50 | IoU 0.5 기준 평균 정밀도 |
| mAP50-95 | IoU 0.5~0.95 평균 정밀도 |

연기 탐지는 놓치는 경우가 더 위험하므로 `smoke` 클래스의 Recall을 따로 확인합니다.

배경 이미지에서 연기로 잘못 탐지한 비율(오탐률)도 함께 기록합니다.

### 13.6 이미지 이상 판정 규칙

"바운딩한 이미지가 이상"이라는 판단은 사람이 정한 규칙으로 내립니다. LLM이나 Agent가 정하지 않습니다.

| 판정 | 조건(예시) | 처리 |
|---|---|---|
| `SMOKE_DETECTED` | `smoke` 박스 신뢰도 ≥ 0.50 | Agent 연계 실행 |
| `LOW_CONFIDENCE` | `smoke` 박스 신뢰도 0.25 ~ 0.50 | 사람 확인 표시 |
| `NO_SMOKE` | `smoke` 박스 없음 | 연계 실행 안 함 |

임계값 0.50과 0.25는 예시입니다. validation 결과로 정하고, test 평가 전에 고정합니다.

#### 13.6.1 신뢰도와 임계값

**신뢰도는 학습된 모델이 박스마다 직접 내놓는 값**입니다. 학습 과정에서 모델은 정답 박스와 많이 겹치는 예측에는 높은 점수를, 배경에는 낮은 점수를 내도록 함께 학습됩니다. 따라서 신뢰도를 계산하는 학습을 따로 할 필요는 없습니다.

```python
r = model.predict("incoming/img_001.jpg", conf=0.25)[0]

for box in r.boxes:
    print(int(box.cls), float(box.conf), box.xyxy.tolist())
```

`conf=0.25`는 이 값보다 낮은 박스를 출력하지 않는다는 뜻입니다.

학습으로 정해야 하는 것은 신뢰도가 아니라 **판정 임계값**입니다.

| 구분 | 누가 정하나 | 방법 |
|---|---|---|
| 신뢰도 | 모델 | 학습된 모델이 박스마다 출력 |
| 판정 임계값 | 사람(규칙) | validation 결과를 보고 선택 후 고정 |

임계값은 다음 순서로 정합니다.

1. validation 데이터로 `model.val()`을 실행합니다.
2. 결과 폴더의 `F1_curve.png`, `P_curve.png`, `R_curve.png`에서 신뢰도별 F1, Precision, Recall을 확인합니다.
3. `SMOKE_DETECTED` 임계값은 F1이 가장 높은 값 근처에서 고르되, 연기를 놓치지 않도록 `smoke` Recall이 목표 이상인지 확인합니다.
4. `LOW_CONFIDENCE` 하한은 그보다 낮게 두어, 애매한 사진을 사람이 확인하도록 넘깁니다.
5. 고른 값과 이유를 문서에 적고, test 평가 전에 고정합니다.

```python
metrics = model.val(
    data="dfire.yaml",
    split="val"
)
```

**주의:** 신뢰도 0.7은 "연기일 확률 70%"를 뜻하지 않습니다. 모델이 매긴 상대적 점수일 뿐이므로 화면에는 "신뢰도"로 표시하고, 확률처럼 설명하지 않습니다. 확률처럼 쓰려면 별도의 보정(calibration) 과정이 필요한데, 이번 실습 범위에는 넣지 않습니다.

`fire` 박스는 이번 실습에서 연기와 같은 이상으로 처리할지 먼저 정하고 문서에 적어 둡니다.

```python
SMOKE_CLASS = 0
HIGH_CONF = 0.50
LOW_CONF = 0.25

def judge_image(result):
    boxes = result.boxes

    smoke_conf = [
        float(c)
        for c, k in zip(boxes.conf, boxes.cls)
        if int(k) == SMOKE_CLASS
    ]

    if not smoke_conf:
        return "NO_SMOKE", 0.0

    top = max(smoke_conf)

    if top >= HIGH_CONF:
        return "SMOKE_DETECTED", top

    if top >= LOW_CONF:
        return "LOW_CONFIDENCE", top

    return "NO_SMOKE", top
```

### 13.7 이미지–TMS 시연용 매핑표

이미지마다 보여 줄 TMS 시점을 미리 정해 둡니다.

| 컬럼 | 설명 |
|---|---|
| `image_id` | 이미지 식별자 |
| `image_path` | 이미지 경로 |
| `data_origin` | KR 또는 TW |
| `entity_key` | 사업장·배출구 식별자 |
| `item_code` | 측정 항목 |
| `measured_at` | 연결할 측정 시각 |
| `scenario_tag` | 시연 시나리오(예: 상승 구간, 정상 구간) |
| `mapping_note` | 이 시점을 고른 이유 |

매핑표를 만들 때 지킬 점입니다.

1. `measured_at`은 **Test 기간**에서만 고릅니다. 모델이 학습에 쓴 시점을 고르면 추정 성능이 실제보다 좋아 보입니다.
2. `measured_at` 기준으로 과거 6시간이 연속된 유효 데이터인 시점만 고릅니다. 그래야 Lag·Rolling·LSTM 입력을 만들 수 있습니다.
3. 연기 이미지는 농도 상승 구간에, 배경 이미지는 정상 구간에 연결하는 식으로 시나리오를 고를 수 있습니다. 이는 **의도적으로 고른 연결**이므로 `mapping_note`에 이유를 적습니다.
4. 매핑표는 실습 시작 전에 만들고 이후에 바꾸지 않습니다. 결과를 보고 연결을 바꾸면 시연 결과가 왜곡됩니다.

```python
scene_map = pd.read_csv(
    "data/mapping/image_scene_map.csv",
    parse_dates=["measured_at"]
)
```

### 13.8 AI Agent 연계 흐름

Agent는 도구를 정해진 순서로 호출하고 결과를 모아 보여 주는 역할만 합니다.

탐지·판정·추정 값은 모두 Python 도구가 만들고, LLM은 그 값을 문장으로 설명합니다.
<img width="745" height="395" alt="image" src="https://github.com/user-attachments/assets/d620952c-5cae-41db-818b-50ac49d01256" />


```text
이미지 입력
   │
   ▼
① detect_smoke        YOLO11n 추론 → 박스·신뢰도
   │
   ▼
② judge_image         규칙 판정 (13.6)
   │
   ├── NO_SMOKE ──────────▶ 종료 (탐지 결과만 표시)
   │
   ▼ SMOKE_DETECTED / LOW_CONFIDENCE
③ lookup_scene_map    image_id → entity_key, item_code, measured_at
   │
   ▼
④ get_tms_window      measured_at 이전 6시간 측정값 조회
   │
   ▼
⑤ model_infer         60분 후 농도 추정 (RF, XGBoost, LSTM)
   │
   ▼
⑥ compare_standard    추정값과 적용 기준값 비교 (12.2 규칙)
   │
   ▼
⑦ explain             LLM 설명 문장 + 숫자 대조
   │
   ▼
화면 표시 + 실행 이력 저장
```

| 도구 | 담당 | 입력 | 출력 |
|---|---|---|---|
| `detect_smoke` | Python(YOLO) | 이미지 | 박스, 클래스, 신뢰도 |
| `judge_image` | Python(규칙) | 탐지 결과 | 판정, 최고 신뢰도 |
| `lookup_scene_map` | Python | `image_id` | 매핑 행 |
| `get_tms_window` | Python | 매핑 행 | `measured_at` 이전 측정값 |
| `model_infer` | Python(승인 모델) | 측정값 창 | 모델별 60분 후 추정값 |
| `compare_standard` | Python(규칙) | 추정값, 기준값 | 기준대비율, 경고 검토 여부 |
| `explain` | LLM | 위 결과 전체 | 설명 문장 |

### 13.9 1시간 후 농도 추정

1시간 후 추정은 9~10장에서 학습한 60분 후 예측 모델을 그대로 씁니다. 새 모델을 만들지 않습니다.

```python
def model_infer(entity_key, item_code, measured_at):
    window = get_tms_window(
        entity_key,
        item_code,
        measured_at,
        hours=6
    )

    # 7장과 같은 함수로 피처를 만든다
    features = build_features(window)

    return {
        "target_at": measured_at + pd.Timedelta(minutes=60),
        "rf": float(rf_model.predict(features)[0]),
        "xgb": float(xgb_model.predict(features)[0]),
        "lstm": predict_lstm(window)
    }
```

지킬 점입니다.

1. 입력에는 `measured_at` 이전과 당시 데이터만 씁니다. 7.7절의 데이터 누수 방지 원칙과 같습니다.
2. 피처 생성은 학습 때와 같은 함수를 씁니다. 따로 만들면 학습과 추론의 피처가 달라집니다.
3. 모델은 읽기 전용으로 불러옵니다. Agent가 모델을 다시 학습하거나 설정을 바꾸지 않습니다.
4. 모델별 추정값이 크게 다르면 그 차이를 그대로 보여 주고 "추가 확인 필요"로 표시합니다.

매핑 시점이 과거 Test 데이터이므로 60분 후 실측값이 이미 있습니다. 이 값은 추정이 끝난 다음 **비교용 사후 실측**으로만 따로 보여 주고, 추정 입력에는 넣지 않습니다.

### 13.10 Agent 결과 화면과 이력

화면에는 다음 내용을 함께 보여 줍니다.

| 영역 | 내용 |
|---|---|
| 이미지 | 바운딩박스가 그려진 이미지, 판정, 신뢰도 |
| 매핑 정보 | 사업장·배출구, 측정 항목, `measured_at`, "시연용 매핑" 표시 |
| TMS 데이터 | `measured_at` 이전 6시간 측정값 그래프와 표 |
| 1시간 후 추정 | RF·XGBoost·LSTM 추정값, 모델 간 차이, 기준대비율 |
| 사후 실측 | 60분 후 실제 측정값(비교용) |
| 설명 | LLM 설명 문장(숫자 대조 통과 여부 포함) |
| 안내 | "판정 아님 · 시연용 매핑" |

Agent 결과는 판정 흐름 밖의 결과입니다. 따라서 Gold, 경고 이력, Slack 알림에 쓰지 않습니다.

실행 이력은 별도 테이블(예: `ops.image_agent_log`)에 남깁니다.

| 컬럼 | 설명 |
|---|---|
| `run_id` | 실행 식별자 |
| `image_id` | 입력 이미지 |
| `judge_result` | 이미지 판정 |
| `top_confidence` | 최고 연기 신뢰도 |
| `entity_key`, `item_code`, `measured_at` | 매핑 정보 |
| `pred_rf`, `pred_xgb`, `pred_lstm` | 모델별 추정값 |
| `actual_60m` | 사후 실측(비교용) |
| `model_version`, `yolo_version` | 사용한 모델 버전 |
| `explanation` | 설명 문장 |
| `created_at` | 실행 시각 |

### 13.11 LLM 설명 규칙

LLM에는 도구 결과만 넘기고 다음 규칙을 지키게 합니다.

1. 도구 결과에 있는 숫자만 씁니다. 설명에 나온 숫자는 도구 결과와 자동으로 대조합니다.
2. 이미지 속 연기를 해당 굴뚝 배출의 원인이나 증거로 설명하지 않습니다.
3. 추정값을 "기준 초과 확정"처럼 표현하지 않습니다.
4. 대조에 실패하면 설명 문장 대신 도구 결과 표만 보여 줍니다.

### 13.12 평가

| 평가 대상 | 지표 |
|---|---|
| 연기 탐지 | Precision, Recall, mAP50, mAP50-95, 배경 이미지 오탐률 |
| 이미지 판정 | 판정별 건수, `LOW_CONFIDENCE` 비율 |
| 1시간 후 추정 | 매핑 시점의 추정값과 사후 실측의 MAE (11장 Test 성능과 비교) |
| Agent 연계 | 도구 호출 순서 준수, 숫자 대조 통과율, 실행 실패율 |

1시간 후 추정 성능은 매핑표에 있는 시점 몇 개로만 계산되므로 모델 성능 평가가 아닙니다. 모델 성능은 11장의 Test 결과로 판단합니다.

**작업 결과:** YOLO11n 연기 탐지 모델, 이미지 판정 규칙, 시연용 매핑표, Agent 연계 화면과 실행 이력을 확보합니다.

---

## 14. 최종 산출물

각 단계에서 생성한 결과를 정리합니다.

| 단계 | 산출물 |
|---|---|
| 원천 데이터 분석 | 국내·대만 데이터 구조 분석 |
| 항목 및 단위 통일 | 항목 매핑표 및 단위 정리 |
| 데이터 통합 | 통합 데이터셋 |
| 품질 관리 | 결측·중복·상태 코드 분석 결과 |
| 시간 통일 | 공통 시간 단위 데이터 |
| 피처 엔지니어링 | Lag, Rolling, 시간 피처 |
| 학습 데이터 구성 | Train / Validation / Test |
| 머신러닝 | Random Forest 및 XGBoost 모델 |
| 딥러닝 | LSTM 모델 |
| 모델 평가 | MAE / RMSE / R² |
| 최종 결과 | 모델별 예측 결과 및 비교 자료 |
| 연기 탐지 | YOLO11n(D-Fire) 모델, 탐지 평가 결과, 이미지 판정 규칙 |
| Agent 연계 | 시연용 매핑표, Agent 결과 화면, 실행 이력 |

### 14.1 권장 프로젝트 디렉터리

```text
tms-ai-project/
│
├── data/
│   ├── raw/
│   │   ├── korea/
│   │   ├── taiwan/
│   │   └── dfire/
│   │
│   ├── mapping/
│   │   └── image_scene_map.csv
│   │
│   ├── processed/
│   │   ├── korea_cleaned.parquet
│   │   ├── taiwan_cleaned.parquet
│   │   └── integrated.parquet
│   │
│   └── features/
│       ├── train.parquet
│       ├── validation.parquet
│       └── test.parquet
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_integration.ipynb
│   ├── 03_data_quality.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_xgboost.ipynb
│   ├── 07_lstm.ipynb
│   ├── 08_yolo_dfire.ipynb
│   └── 09_image_agent.ipynb
│
├── configs/
│   └── dfire.yaml
│
├── models/
│   ├── random_forest/
│   ├── xgboost/
│   ├── lstm/
│   └── yolo11n_dfire/
│
└── reports/
    ├── data_quality_report.md
    ├── model_evaluation.md
    └── smoke_detection_evaluation.md
```

---

## 15. 전체 작업 순서 요약

실제 구현은 다음 순서대로 진행합니다.

| 순서 | 작업 | 핵심 내용 |
|---:|---|---|
| 1 | 원천 데이터 확인 | 국내 TMS·대만 CEMS 구조 분석 |
| 2 | 항목 및 단위 통일 | 컬럼 매핑, 단위 확인 |
| 3 | 데이터 통합 | 국내와 대만 데이터 이어 붙이기 |
| 4 | 데이터 품질 관리 | 결측, 중복, 이상값, 상태 코드 |
| 5 | 시간 단위 통일 | 15분 및 30분 데이터 정리 |
| 6 | 피처 엔지니어링 | Lag, Rolling, 시간 정보 |
| 7 | Target 생성 | 60분 후 측정값 |
| 8 | 데이터 분리 | Train / Validation / Test |
| 9 | 머신러닝 학습 | Persistence, Random Forest, XGBoost |
| 10 | 딥러닝 학습 | LSTM 시퀀스 생성 및 학습 |
| 11 | 모델 평가 | MAE, RMSE, R² |
| 12 | 결과 비교 | 모델별·국가별 성능 비교 |
| 13 | 결과 활용 | 예측값 및 경고 정보 생성 |
| 14 | 연기 탐지 | D-Fire로 YOLO11n 학습·평가, 이미지 판정 규칙 |
| 15 | Agent 연계 | 매핑표 조회 → TMS 표시 → 1시간 후 추정 → 설명 |

---

## 16. 반드시 기억할 핵심 원칙

이번 실습에서는 국내와 대만 데이터를 하나로 통합하여 모델을 학습합니다.

교육 목적에 맞춰 통합 데이터셋을 구성하되 다음 원칙은 지킵니다.

1. 국내와 대만 데이터를 행 방향으로 이어 붙입니다.
2. 원본 데이터와 원본 단위는 보존합니다.
3. `data_origin`으로 데이터 출처를 구분합니다.
4. 단위가 다른 경우 실습용 변환 기준을 명시합니다.
5. 결측값을 임의로 0으로 채우지 않습니다.
6. 실제 기준 초과 측정값은 무조건 제거하지 않습니다.
7. 시간 순서를 유지하여 피처를 생성합니다.
8. 미래 측정값이 입력 피처에 포함되지 않도록 합니다.
9. LSTM 시퀀스는 동일한 사업장·배출구·측정 항목 안에서 생성합니다.
10. Train 데이터로만 스케일러를 학습합니다.
11. Random Forest, XGBoost, LSTM을 동일한 예측 목표로 비교합니다.
12. 전체 성능과 국가별 성능을 함께 확인합니다.
13. 이미지와 TMS 데이터의 연결은 시연용 매핑이며, 탐지 결과를 시계열 모델의 피처나 라벨로 쓰지 않습니다.
14. 이미지 이상 판정과 기준 비교는 규칙 코드가 하고, Agent는 도구 호출과 결과 표시만 맡습니다.
15. 1시간 후 추정은 승인된 60분 예측 모델을 읽기 전용으로 쓰고, 결과는 판정이 아니므로 Gold·경고 이력·Slack에 쓰지 않습니다.

**이번 실습의 핵심은 데이터 통합부터 머신러닝과 딥러닝 모델 학습·평가, 그리고 이미지 탐지와 Agent 연계까지 전체 과정을 직접 구현해 보는 것입니다.**

데이터를 먼저 이해하고, 정리하고, 피처를 만들고, 모델을 학습한 다음 결과를 비교하는 순서로 진행합니다.
