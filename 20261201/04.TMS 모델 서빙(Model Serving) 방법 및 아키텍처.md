# TMS 모델 서빙(Model Serving) 방법 및 아키텍처

## 1. 모델 서빙(Model Serving)이란?

모델 서빙(Model Serving)은 학습이 완료된 AI 모델을 실제 운영 환경에 배포하고, 새로운 데이터가 들어왔을 때 예측 결과를 반환하도록 만드는 과정이다.

쉽게 말하면 다음과 같다.

- 모델 학습(Model Training): 데이터를 이용해 예측 모델을 만든다.
- 모델 평가(Model Evaluation): 학습된 모델의 예측 성능을 검증한다.
- 모델 등록(Model Registry): 검증된 모델을 버전별로 관리한다.
- 모델 서빙(Model Serving): 운영 데이터가 들어오면 등록된 모델로 예측한다.
- 최종 판정(Decision Engine): 예측 결과와 업무 규칙을 결합하여 실제 경고 여부를 결정한다.

TMS 프로젝트에서는 NOx(질소산화물), SO2(황산화물) 측정 데이터를 이용해 미래 농도 또는 기준 초과 가능성을 예측하고, 그 결과를 사전 경고 시스템에 전달하는 것이 모델 서빙의 목적이다.

### 모델 학습과 모델 서빙의 차이

| 구분 | 모델 학습(Model Training) | 모델 서빙(Model Serving) |
|---|---|---|
| 목적 | 예측 모델 생성 | 실제 데이터 예측 |
| 입력 | 과거 학습 데이터 | 새로운 운영 데이터 |
| 처리 | 모델 파라미터 학습 | 학습된 모델로 추론 |
| 실행 | 학습 시점 | 운영 중 반복 실행 |
| 결과 | 학습된 모델 파일 | 예측값, 확률 등 |
| 예시 | XGBoost 학습 | 다음 30분 NOx 농도 예측 |

**핵심:** 모델 학습은 예측 능력을 만드는 과정이고, 모델 서빙은 그 능력을 실제 서비스에서 사용하는 과정이다.

---

## 2. TMS 전체 ML 파이프라인

TMS 프로젝트에서는 데이터 수집부터 최종 경고까지 다음과 같은 흐름으로 구성할 수 있다.

```text
[실제 TMS 측정 데이터]
          |
          v
   [Kafka / 수집 시스템]
          |
          v
     [Bronze Layer]
      원본 데이터 보존
          |
          v
     [Silver Layer]
    데이터 정제 및 표준화
          |
          v
      [Gold Layer]
    ML 학습용 Feature 생성
          |
          v
     [Model Training]
   Random Forest / XGBoost
          / LSTM
          |
          v
     [Model Evaluation]
     모델 성능 검증
          |
          v
         [MLflow]
   실험 관리 및 모델 버전 관리
          |
          v
     [Model Registry]
     승인된 모델 등록
          |
          v
      [Model Serving]
       모델 추론 실행
          |
          v
     [Python Rule Engine]
   예측 결과 + 운영 기준 결합
          |
          v
      [Final Decision]
      최종 경고 여부 결정
          |
          v
   [Gold / PostgreSQL 저장]
          |
          v
    [Grafana / Slack 알림]
```

여기서 중요한 것은 모델이 예측한 결과와 실제 운영 경고를 구분하는 것이다.

예를 들어 모델이 다음과 같은 결과를 반환했다고 가정해 보자.

```json
{
  "stack_code": "STACK_001",
  "item_code": "nox",
  "predicted_value_30m": 92.5,
  "exceedance_probability": 0.87,
  "model_version": "xgboost_v1.2"
}
```

이 결과는 모델의 예측 결과일 뿐, 아직 최종 경고가 확정된 것은 아니다.

Python Rule Engine이 운영 기준, 현재 측정값, 예측값, 데이터 품질 등을 함께 검토하여 최종 경고를 결정한다.

---

## 3. 모델 서빙에 사용되는 주요 기술

TMS 프로젝트에서 사용할 수 있는 기술을 역할별로 구분하면 다음과 같다.

| 기술 | 주요 역할 | 쉽게 설명하면 |
|---|---|---|
| MLflow | 실험 및 모델 관리 | 어떤 모델을 학습했고 어떤 모델이 좋은지 관리 |
| Amazon S3 | 모델 및 데이터 저장 | 모델 파일을 보관하는 저장소 |
| Airflow | 작업 스케줄링 | 언제 데이터를 처리하고 모델을 실행할지 관리 |
| Python | 데이터 처리 및 추론 | 실제 모델을 실행하는 코드 |
| FastAPI | 추론 API 제공 | 외부 시스템이 모델을 호출할 수 있도록 API 제공 |
| Docker | 실행 환경 패키징 | 모델과 실행 환경을 하나로 묶음 |
| ECS | 컨테이너 실행 | Docker 애플리케이션을 AWS에서 운영 |
| SageMaker | 관리형 ML 서비스 | AWS에서 모델 배포와 추론을 관리 |
| Kafka | 이벤트 전달 | 데이터가 들어오는 즉시 전달 |
| Spark | 대규모 데이터 처리 | 많은 데이터를 분산 처리 |
| Grafana | 모니터링 및 시각화 | 예측 결과와 운영 상태를 대시보드로 확인 |

각 기술은 서로 대체하는 관계가 아니다.

예를 들어 Airflow는 작업 실행 시점을 관리하고, Python은 실제 추론 코드를 실행하며, MLflow는 사용할 모델의 버전을 관리한다.

---

## 4. MLflow란 무엇인가?

MLflow는 머신러닝 모델의 학습부터 평가, 등록, 배포까지 모델의 생명주기(Model Lifecycle)를 관리하는 오픈소스 플랫폼이다.

TMS 프로젝트에서는 여러 모델의 성능을 비교하고, 운영에 사용할 모델을 버전별로 관리하는 데 활용할 수 있다.

### 4.1 MLflow의 주요 기능

| 기능 | 설명 | TMS 적용 예시 |
|---|---|---|
| Experiment Tracking | 학습 실험 기록 | XGBoost와 Random Forest 성능 비교 |
| Parameter Tracking | 모델 학습 설정 기록 | max_depth, learning_rate 등 |
| Metric Tracking | 모델 평가 지표 기록 | F1-score, Recall, MAE, RMSE |
| Artifact Management | 학습 결과물 저장 | 모델 파일, 그래프, 평가 결과 |
| Model Registry | 모델 버전 및 상태 관리 | xgboost_v1.0, xgboost_v1.1 |
| Model Deployment | 모델 배포 지원 | 추론 환경에서 모델 로딩 및 실행 |

### 4.2 MLflow를 사용하는 이유

TMS 프로젝트에서 다음과 같은 상황을 생각해 보자.

- Random Forest 모델의 Recall은 0.78이다.
- XGBoost 모델의 Recall은 0.86이다.
- LSTM 모델의 Recall은 0.89이다.

단순히 성능이 높은 모델을 선택하는 것만으로는 부족하다.

실제 운영에서는 모델의 버전, 학습 데이터, 평가 결과, 사용한 Feature, Scaler 등을 함께 관리해야 한다.

MLflow를 이용하면 이러한 정보를 실험 단위로 기록하고, 운영에 사용할 모델을 명확하게 관리할 수 있다.

### 4.3 MLflow Model Registry

Model Registry는 검증된 모델을 버전별로 등록하고 관리하는 기능이다.

예를 들어 다음과 같이 관리할 수 있다.

```text
TMS NOx Prediction Model
  |
  +-- xgboost_v1.0
  |
  +-- xgboost_v1.1
  |
  +-- lstm_v1.0
  |
  +-- xgboost_v1.2  <-- 운영 승인 모델
```

모델이 개선되었다고 해서 즉시 운영 모델을 교체하는 것은 바람직하지 않다.

새 모델의 성능을 검증하고 승인한 다음 운영 환경에 배포해야 한다.

참고:
- MLflow 공식 문서: https://mlflow.org/docs/latest/
- MLflow Model Registry: https://mlflow.org/docs/latest/ml/model-registry/

---

## 5. AWS에서 모델 서빙하는 방법

AWS에서는 모델을 운영 환경에 배포하기 위해 여러 가지 방법을 제공한다.

대표적인 방식은 다음과 같다.

1. Amazon SageMaker
2. Amazon ECS + Docker + FastAPI
3. Amazon EKS + Kubernetes
4. AWS Lambda + API Gateway

이 중 어떤 방식이 가장 좋다고 단정할 수는 없다. 모델의 크기, 요청량, 응답 시간, 운영 인력, 비용 등에 따라 선택해야 한다.

### 5.1 Amazon SageMaker 기반 서빙

Amazon SageMaker는 AWS에서 제공하는 머신러닝 전용 관리형 서비스다.

모델 학습, 모델 관리, 배포, 추론, 모니터링 등을 통합적으로 지원한다.

주요 특징:

- 모델 배포 및 추론 API 제공
- Real-time Endpoint 지원
- Serverless Inference 지원
- Asynchronous Inference 지원
- Batch Transform 지원
- 모델 확장 및 운영 관리 기능 제공

예를 들어 MLflow에 등록된 XGBoost 모델을 SageMaker Endpoint에 배포하면 외부 애플리케이션에서 API를 호출하여 예측 결과를 받을 수 있다.

```text
[MLflow Model Registry]
          |
          v
   [SageMaker Model]
          |
          v
  [SageMaker Endpoint]
          |
          v
     [API Request]
          |
          v
      [Prediction]
```

**장점**
- AWS 관리형 서비스이므로 인프라 운영 부담을 줄일 수 있다.
- 모델 배포와 확장 기능을 활용할 수 있다.
- MLflow와 연동하여 모델 관리 체계를 구성할 수 있다.

**단점**
- 사용 방식에 따라 비용이 발생한다.
- SageMaker 서비스 구조와 설정을 이해해야 한다.
- 초기 PoC에서는 기능이 다소 복잡하게 느껴질 수 있다.

### 5.2 Amazon ECS + Docker + FastAPI

ECS는 Docker 컨테이너를 AWS에서 실행하고 관리하는 서비스다.

FastAPI는 Python으로 API 서버를 개발하기 위한 웹 프레임워크다.

이 방식에서는 모델을 Docker 이미지에 포함하거나 실행 시 S3에서 모델 파일을 내려받아 로딩한다.

```text
[MLflow Model Registry]
          |
          v
      [Amazon S3]
          |
          v
    [Docker Image]
          |
          v
   [Amazon ECS]
          |
          v
      [FastAPI]
          |
          v
    [Prediction API]
```

**장점**
- Python 기반으로 모델 추론 로직을 자유롭게 구현할 수 있다.
- Rule Engine 등 업무 로직과 함께 운영하기 좋다.
- SageMaker를 사용하지 않고도 추론 API를 구성할 수 있다.
- ECS Fargate를 사용하면 서버 인프라 관리 부담을 줄일 수 있다.

**단점**
- API 서버와 컨테이너 운영 구조를 직접 설계해야 한다.
- 모델 배포, 모니터링, 확장 정책 등을 직접 구성해야 한다.

TMS에서는 모델 추론뿐 아니라 Python Rule Engine까지 함께 운영할 수 있다는 장점이 있다.

### 5.3 Amazon EKS + Kubernetes

EKS는 AWS에서 Kubernetes를 관리형으로 사용할 수 있도록 제공하는 서비스다.

여러 개의 AI 모델과 추론 서비스를 대규모로 운영할 때 고려할 수 있다.

```text
          [Amazon EKS]
                |
       +--------+--------+
       |        |        |
       v        v        v
    [NOx]    [SO2]   [Vision]
    Model    Model    Model
       |        |        |
       +--------+--------+
                |
                v
         [Inference API]
```

**장점**
- 여러 모델을 독립적인 서비스로 운영할 수 있다.
- Kubernetes 기반 자동 확장과 배포 전략을 활용할 수 있다.
- GPU 기반 모델 서빙 플랫폼으로 확장할 수 있다.

**단점**
- Kubernetes 운영 역량이 필요하다.
- 초기 구축과 운영 복잡도가 높다.
- 소규모 프로젝트에서는 과도한 구성이 될 수 있다.

따라서 현재 TMS 프로젝트의 초기 구현 단계에서는 우선순위를 낮추는 것이 좋다.

### 5.4 AWS Lambda + API Gateway

Lambda는 서버를 직접 관리하지 않고 코드를 실행할 수 있는 서버리스 컴퓨팅 서비스다.

API Gateway와 함께 사용하면 간단한 추론 API를 만들 수 있다.

```text
[Client]
   |
   v
[API Gateway]
   |
   v
[AWS Lambda]
   |
   v
[Lightweight Model]
   |
   v
[Prediction]
```

가벼운 모델이나 간헐적인 요청 처리에는 적합할 수 있다.

다만 모델 크기, 실행 시간, 메모리, 패키지 크기 등의 제약을 고려해야 하므로 모든 ML 모델에 적합한 것은 아니다.

---

## 6. AWS 현업에서 선택하는 서빙 구조

AWS 기반 현업 시스템에서는 크게 다음과 같은 선택지를 볼 수 있다.

| 방식 | 주요 특징 | 적합한 환경 |
|---|---|---|
| SageMaker | ML 전용 관리형 추론 | ML 전담 운영 및 MLOps 환경 |
| ECS + FastAPI | Docker 기반 자체 추론 API | 일반적인 Python ML 서비스 |
| EKS + Kubernetes | 컨테이너 플랫폼 기반 통합 운영 | 여러 AI 서비스와 대규모 운영 |
| Lambda | 서버리스 함수 실행 | 가볍고 간헐적인 추론 |

기업별 사용 비중을 나타내는 공식적인 공통 순위가 있는 것은 아니다.

실무에서는 기존 AWS 인프라, 개발 조직의 기술 역량, 운영 규모, 비용에 따라 선택한다.

특히 다음 두 가지 방식은 TMS와 같은 프로젝트에서 우선 검토할 만하다.

- **SageMaker:** AWS의 ML 관리형 기능을 적극적으로 활용하고 싶을 때
- **ECS + FastAPI:** Python 기반 추론과 업무 로직을 유연하게 직접 구현하고 싶을 때

---

## 7. TMS에 적용할 모델 서빙 방식

모델 서빙은 데이터를 어떻게 전달하고 추론을 실행하느냐에 따라 Batch Serving, Real-time Serving, Streaming Serving으로 구분하여 이해할 수 있다.

### 7.1 Batch Serving

Batch Serving은 일정한 시간 간격으로 데이터를 모아서 한꺼번에 모델을 실행하는 방식이다.

예를 들어 TMS 측정 데이터가 30분마다 들어온다면, 30분 주기로 데이터를 읽고 모델 추론을 수행할 수 있다.

```text
[30분 단위 TMS 데이터 수집]
              |
              v
         [Amazon S3]
              |
              v
          [Airflow]
       30분 주기 실행
              |
              v
      [Python Inference]
              |
              v
         [MLflow Model]
              |
              v
         [Prediction]
              |
              v
       [Python Rule Engine]
              |
              v
        [최종 경고 판정]
```

예를 들어 다음과 같이 처리할 수 있다.

1. 10:00~10:30 측정 데이터 수집
2. 10:30 이후 데이터 적재 완료 확인
3. Airflow DAG 실행
4. Gold Layer에서 Feature 조회
5. MLflow에서 승인된 모델 버전 확인
6. Python으로 모델 추론
7. Rule Engine으로 최종 판정
8. 예측 결과 및 경고 이력 저장

**장점**
- 구조가 단순하다.
- 대량의 데이터를 효율적으로 처리할 수 있다.
- API 서버를 항상 실행할 필요가 없다.
- Airflow를 이용해 작업 순서와 실패 재처리를 관리할 수 있다.

**단점**
- 실시간 응답이 필요한 서비스에는 적합하지 않다.
- 다음 배치 실행까지 기다려야 한다.

현재 TMS 프로젝트의 초기 모델 서빙에는 Batch Serving을 우선 추천한다.

### 7.2 Real-time Serving

Real-time Serving은 API 요청이 들어왔을 때 즉시 모델을 실행하고 결과를 반환하는 방식이다.

예를 들어 사용자가 특정 굴뚝의 NOx 농도 예측을 요청하면 API 서버가 모델을 실행하여 결과를 반환한다.

```text
[Client / Dashboard]
          |
          v
     [API Request]
          |
          v
       [FastAPI]
          |
          v
     [MLflow Model]
          |
          v
      [Prediction]
          |
          v
    [Rule Engine]
          |
          v
     [API Response]
```

예시 API:

```http
POST /api/v1/predictions/nox
```

요청 데이터:

```json
{
  "stack_code": "STACK_001",
  "item_code": "nox",
  "current_value": 85.2,
  "temperature": 120.5,
  "flow_rate": 35.4
}
```

응답 데이터:

```json
{
  "stack_code": "STACK_001",
  "item_code": "nox",
  "predicted_value_30m": 92.5,
  "exceedance_probability": 0.87,
  "model_version": "xgboost_v1.2"
}
```

**장점**
- API 요청에 즉시 응답할 수 있다.
- 외부 시스템과 쉽게 연동할 수 있다.
- 대시보드에서 사용자가 직접 예측을 요청할 수 있다.

**단점**
- API 서버를 운영해야 한다.
- 동시 요청과 응답 시간을 관리해야 한다.
- 서버 확장과 장애 대응이 필요하다.

AWS에서는 ECS + FastAPI 또는 SageMaker Endpoint를 이용해 구성할 수 있다.

### 7.3 Streaming Serving

Streaming Serving은 데이터가 지속적으로 발생하는 환경에서 이벤트를 연속적으로 처리하고 모델 추론을 수행하는 방식이다.

Kafka와 Spark Structured Streaming 등을 이용해 데이터 유입부터 전처리, 추론까지 연결할 수 있다.

```text
[TMS 측정 데이터]
          |
          v
        [Kafka]
          |
          v
   [Spark Streaming]
          |
          v
    [Feature Processing]
          |
          v
     [Model Inference]
          |
          v
      [Rule Engine]
          |
          v
     [Alert Storage]
          |
          v
    [Grafana / Slack]
```

**장점**
- 데이터가 유입되는 대로 지속적으로 처리할 수 있다.
- 대규모 이벤트 처리에 적합하다.
- 여러 데이터 소스를 하나의 처리 파이프라인으로 연결할 수 있다.

**단점**
- Kafka, Spark 등의 운영 복잡도가 증가한다.
- 이벤트 중복, 순서, 지연, 장애 복구 등을 고려해야 한다.
- 데이터 발생 빈도가 낮으면 복잡도에 비해 얻는 효과가 작을 수 있다.

참고로 Streaming Serving은 반드시 데이터 한 건씩 처리하는 방식만을 의미하지 않는다. Spark Structured Streaming은 마이크로 배치(Micro-batch) 방식으로도 처리할 수 있다.

---

## 8. Batch Serving과 Real-time Serving, Streaming Serving의 차이

세 가지 방식을 비교하면 다음과 같다.

| 구분 | Batch Serving | Real-time Serving | Streaming Serving |
|---|---|---|---|
| 데이터 전달 | 일정량을 모아서 처리 | API 요청으로 전달 | 이벤트가 지속적으로 유입 |
| 처리 방식 | 배치 단위 | 요청 단위 | 이벤트 또는 마이크로 배치 |
| 실행 시점 | 정해진 주기 | 요청이 발생할 때 | 데이터가 유입될 때 |
| 주요 기술 | Airflow, Python | FastAPI, SageMaker | Kafka, Spark |
| 응답 형태 | 배치 결과 저장 | API 응답 | 처리 결과 이벤트 |
| 주요 목적 | 주기적인 예측 | 즉각적인 요청 처리 | 지속적인 이벤트 처리 |
| TMS 적용 | 초기 구현 추천 | 운영 기능 확장 | 향후 확장 |

주의할 점은 Real-time과 Streaming이 완전히 서로 배타적인 개념은 아니라는 것이다.

Streaming 파이프라인 내부에서 FastAPI를 호출할 수도 있고, Streaming 시스템이 마이크로 배치로 동작할 수도 있다.

핵심은 데이터가 시스템에 전달되는 방식과 처리 목적이 다르다는 것이다.

### TMS에서는 어떤 방식을 선택해야 할까?

현재 프로젝트는 30분 단위 측정 데이터를 기준으로 설계하고 있다.

따라서 초기에는 Batch Serving을 구현하고, 이후 운영 요구사항에 따라 Real-time 또는 Streaming을 추가하는 것이 합리적이다.

```text
1단계: Batch Serving
       Airflow + Python + MLflow

2단계: Real-time Serving
       ECS + FastAPI 또는 SageMaker Endpoint

3단계: Streaming Serving
       Kafka + Spark Structured Streaming
```

---

## 9. TMS 모델 서빙 상세 아키텍처

앞서 설명한 기술을 TMS에 적용하면 다음과 같이 구성할 수 있다.

### 9.1 전체 아키텍처

```text
                   [TMS Data Source]
                           |
                           v
                     [Kafka / API]
                           |
                           v
                    [Amazon S3]
                           |
                           v
                    [Bronze Layer]
                     원본 데이터
                           |
                           v
                    [Silver Layer]
                    정제 및 표준화
                           |
                           v
                     [Gold Layer]
                   ML Feature 생성
                           |
                +----------+----------+
                |                     |
                v                     v
         [Model Training]       [Batch Inference]
                |                     ^
                v                     |
         [Model Evaluation]           |
                |                     |
                v                     |
             [MLflow]                 |
                |                     |
                v                     |
         [Model Registry]             |
                |                     |
                v                     |
           [Amazon S3]                |
                |                     |
                +---------------------+
                           |
                           v
                   [Model Serving]
                           |
                           v
                  [Python Rule Engine]
                           |
                           v
                    [Final Decision]
                           |
                +----------+----------+
                |                     |
                v                     v
         [Gold / Iceberg]      [PostgreSQL]
                |                     |
                +----------+----------+
                           |
                           v
                    [Grafana]
                           |
                           v
                  [Slack Notification]
```

위 구조에서 Training과 Inference는 구분해야 한다.

- Training: 새로운 모델을 학습하고 평가한다.
- Inference: 이미 학습된 모델을 이용하여 새로운 데이터를 예측한다.

운영 중에는 매번 모델을 학습하는 것이 아니라 승인된 모델을 반복해서 사용한다.

### 9.2 모델 파일 구성

모델 파일 하나만 저장하는 것으로는 충분하지 않을 수 있다.

실제 추론 시에는 학습 때 사용한 전처리 과정과 Feature 구성이 동일해야 한다.

예를 들어 다음과 같이 관리할 수 있다.

```text
models/
  nox/
    xgboost_v1.2/
      model.pkl
      scaler.pkl
      feature_config.json
      model_metadata.json
      evaluation_metrics.json

  so2/
    xgboost_v1.1/
      model.pkl
      scaler.pkl
      feature_config.json
      model_metadata.json
      evaluation_metrics.json
```

파일별 역할:

| 파일 | 설명 |
|---|---|
| model.pkl | 학습된 모델 |
| scaler.pkl | 데이터 정규화 및 스케일링 정보 |
| feature_config.json | 모델 입력 Feature 명세 |
| model_metadata.json | 모델 버전 및 학습 정보 |
| evaluation_metrics.json | 모델 평가 결과 |

참고로 모든 모델이 Scaler를 사용하는 것은 아니다. Random Forest와 XGBoost는 일반적으로 별도 스케일링이 필수는 아니지만, LSTM과 같은 모델은 학습 시 사용한 Scaler를 추론에서도 동일하게 적용해야 한다.

LSTM은 특히 다음 항목이 학습과 추론에서 일치해야 한다.

- Feature 순서
- Sequence 길이
- Window 구성
- Scaler
- 입력 데이터 형태

---

## 10. Python Rule Engine과 최종 경고 판정

TMS 프로젝트에서는 모델의 예측값만으로 최종 경고를 결정하지 않는 것이 중요하다.

모델은 미래의 NOx 또는 SO2 농도와 기준 초과 가능성을 예측한다.

하지만 최종 경고는 운영 기준과 데이터 품질 등을 함께 검토하여 결정해야 한다.

### 10.1 모델 예측 결과

```json
{
  "stack_code": "STACK_001",
  "item_code": "nox",
  "current_value": 85.2,
  "predicted_value_30m": 92.5,
  "exceedance_probability": 0.87,
  "model_version": "xgboost_v1.2"
}
```

### 10.2 Rule Engine 처리

```text
[Model Prediction]
        |
        v
[Data Quality Check]
        |
        v
[Current Value Check]
        |
        v
[Prediction Threshold Check]
        |
        v
[Operational Rule Check]
        |
        v
[Final Decision]
```

예를 들어 다음과 같은 조건을 구성할 수 있다.

| 조건 | 판정 예시 |
|---|---|
| 데이터 품질 이상 | 데이터 확인 필요 |
| 현재 측정값이 기준에 근접 | 주의 |
| 미래 예측값이 기준 초과 예상 | 사전 경고 |
| 예측 확률이 낮고 운영 기준 충족 | 정상 |
| 데이터 부족 또는 모델 오류 | 판단 보류 |

위 표는 업무 규칙 설계를 위한 예시이며 실제 운영 기준과 경고 임계값은 별도로 정의해야 한다.

**핵심:** ML 모델은 예측을 담당하고, Python Rule Engine은 예측 결과와 운영 규칙을 결합하여 최종 판정을 담당한다.

---

## 11. 모델 배포(Model Deployment)와 운영 모델 교체

모델을 학습했다고 해서 바로 운영 환경에 적용하면 안 된다.

새로운 모델을 배포하기 전에는 성능 검증과 운영 안정성 확인이 필요하다.

### 11.1 모델 배포 과정

```text
[Model Training]
       |
       v
[Model Evaluation]
       |
       v
[MLflow Registry]
       |
       v
[Model Approval]
       |
       v
[Deployment]
       |
       v
[Shadow Testing]
       |
       v
[Production Serving]
```

### 11.2 운영 모델 교체 전략

| 전략 | 설명 |
|---|---|
| Shadow Testing | 기존 모델과 신규 모델을 동시에 실행하고 결과 비교 |
| Canary Deployment | 일부 요청에만 신규 모델 적용 |
| Blue-Green Deployment | 기존 환경과 신규 환경을 분리하여 전환 |
| Rollback | 문제가 발생하면 이전 모델로 복구 |

TMS에서는 초기에는 Shadow Testing과 Rollback을 우선 적용하는 것이 좋다.

예를 들어 기존 모델인 XGBoost v1.1과 신규 모델인 XGBoost v1.2를 동시에 실행하여 예측 결과를 비교한다.

신규 모델이 안정적이라고 판단되면 운영 모델로 교체한다.

운영 모델 버전은 반드시 예측 이력에 저장해야 한다.

---

## 12. 모델 서빙 모니터링(Model Monitoring)

모델을 운영 환경에 배포했다고 해서 끝나는 것은 아니다.

실제 운영 데이터에서도 모델이 안정적으로 동작하는지 지속적으로 확인해야 한다.

### 12.1 주요 모니터링 항목

| 항목 | 설명 | TMS 적용 |
|---|---|---|
| Prediction Monitoring | 예측 결과 모니터링 | NOx, SO2 예측값 |
| Data Drift | 입력 데이터 분포 변화 | 측정 데이터 분포 변화 |
| Model Drift | 모델 예측 성능 변화 | 예측 오차 증가 |
| Latency | 추론 응답 시간 | 모델 실행 시간 |
| Error Rate | 추론 오류 비율 | 모델 실행 실패율 |
| Missing Rate | 데이터 결측률 | 측정 데이터 누락 비율 |

### 12.2 Data Drift와 Model Drift 차이

Data Drift는 모델에 입력되는 데이터의 분포가 학습 당시와 달라지는 현상이다.

예를 들어 학습 당시 NOx 측정값은 대부분 30~60 범위였는데, 최근에는 70~90 범위의 데이터가 지속적으로 발생하는 상황이다.

Model Drift는 실제 예측 성능이 떨어지는 현상이다.

예를 들어 모델이 미래 NOx 농도를 80으로 예측했지만 실제 측정값은 100으로 반복해서 나타나는 상황이다.

다만 Data Drift가 발생했다고 반드시 Model Drift가 발생하는 것은 아니다.

실제 예측값과 이후 확정된 측정값을 비교해야 모델 성능 저하 여부를 확인할 수 있다.

### 12.3 TMS의 예측 결과 검증

TMS는 30분 단위로 데이터를 수집하므로 예측 결과와 실제 측정값을 시간 기준으로 연결할 수 있다.

```text
10:00 측정 데이터
      |
      v
10:00 모델 예측
      |
      v
10:30 실제 측정값 도착
      |
      v
예측값과 실제값 비교
      |
      v
MAE / RMSE / Recall 평가
      |
      v
모델 성능 모니터링
```

30분 후 실제 측정값이 도착하면 10:00 시점의 예측값과 비교할 수 있다.

60분 후 예측도 동일한 방식으로 검증할 수 있다.

이러한 평가 결과는 모델 개선과 재학습 여부를 판단하는 근거가 된다.

---

## 13. Batch Serving은 강화학습에도 활용할 수 있을까?

가능하다.

다만 Batch Serving 자체가 강화학습은 아니다.

Batch Serving은 일정한 주기로 데이터를 처리하고 모델을 실행하는 방식이며, 강화학습은 상태(State), 행동(Action), 보상(Reward)을 이용해 정책(Policy)을 학습하는 방법이다.

TMS에서는 Batch Serving으로 수집한 과거 운영 데이터를 활용하여 다음과 같은 작업을 수행할 수 있다.

1. 과거 측정 데이터와 설비 운영 이력 수집
2. State, Action, Reward 데이터 구성
3. Offline Reinforcement Learning 학습
4. 기존 정책과 신규 정책 성능 비교
5. 시뮬레이션 환경에서 정책 검증
6. 운영자 승인 후 제한적인 운영 적용

예를 들어 다음과 같은 구조를 생각할 수 있다.

```text
[Historical TMS Data]
          |
          v
    [Batch Processing]
          |
          v
 [State / Action / Reward]
          |
          v
 [Offline RL Training]
          |
          v
   [Policy Evaluation]
          |
          v
    [Simulation Test]
          |
          v
    [Human Approval]
          |
          v
 [Policy Deployment]
```

중요한 점은 TMS 측정 데이터만으로는 충분한 강화학습 환경을 만들기 어렵다는 것이다.

설비 조작 이력, 조작에 따른 상태 변화, 보상 기준 등이 필요하다.

또한 실제 설비 제어에 강화학습 정책을 적용할 때는 안전성 검증과 운영자 승인 절차가 반드시 필요하다.

---

## 14. TMS 프로젝트 단계별 구현 전략

처음부터 모든 기술을 도입할 필요는 없다.

프로젝트에서는 핵심 기능을 먼저 완성한 다음 운영 수준을 높이는 방식이 좋다.

### 1단계: Batch Serving 구현

목표: 학습된 모델을 실제 데이터에 적용하여 예측 결과를 저장한다.

주요 기술:
- Python
- MLflow
- Amazon S3
- Airflow
- PostgreSQL

구현 항목:
- MLflow 실험 관리
- 모델 버전 등록
- 승인된 모델 로딩
- Batch Inference
- Rule Engine
- 예측 결과 저장

### 2단계: Real-time Serving 확장

목표: 외부 시스템이 API로 예측 결과를 요청할 수 있도록 한다.

주요 기술:
- FastAPI
- Docker
- Amazon ECS
- Application Load Balancer

구현 항목:
- 추론 API 개발
- Docker 이미지 생성
- ECS 배포
- API 응답 및 오류 처리
- 운영 모델 버전 관리

### 3단계: Streaming Serving 확장

목표: 이벤트가 지속적으로 발생하는 환경에서 데이터를 실시간 처리한다.

주요 기술:
- Kafka
- Spark Structured Streaming
- Python Inference
- AWS S3

구현 항목:
- Kafka 이벤트 수신
- 데이터 전처리
- Feature 생성
- 모델 추론
- 최종 경고 판정
- 이벤트 저장 및 알림

### 4단계: MLOps 및 강화학습 확장

목표: 모델 성능을 지속적으로 관리하고 개선한다.

구현 항목:
- 모델 성능 자동 평가
- Data Drift / Model Drift 모니터링
- 재학습 파이프라인
- 모델 승인 및 배포 자동화
- Offline Reinforcement Learning 검토
- 운영자 승인 기반 정책 적용

---

## 15. TMS 프로젝트 최종 권장 아키텍처

현재 프로젝트의 데이터 처리 주기와 교육 목적, AWS 활용 범위를 고려하면 다음과 같은 구성을 추천한다.

| 영역 | 추천 기술 | 선정 이유 |
|---|---|---|
| 데이터 저장 | Amazon S3 | Bronze/Silver/Gold 데이터 저장 |
| 데이터 처리 | Python + Spark | 전처리 및 대용량 데이터 처리 |
| 테이블 관리 | Apache Iceberg | 데이터 버전 및 테이블 관리 |
| 작업 스케줄링 | Airflow | 데이터 처리 및 추론 작업 자동화 |
| 모델 학습 | Scikit-learn / XGBoost / LSTM | 예측 모델 구현 |
| 모델 관리 | MLflow | 실험 및 모델 버전 관리 |
| 모델 저장 | Amazon S3 | 모델 파일 및 아티팩트 저장 |
| 초기 모델 서빙 | Python Batch Inference | 30분 단위 추론 |
| API 서빙 확장 | FastAPI + ECS | 외부 시스템 연동 |
| 관리형 ML 서빙 | SageMaker | ML 모델 배포 및 운영 고도화 |
| 최종 판정 | Python Rule Engine | 예측 결과와 업무 규칙 결합 |
| 메타데이터 | PostgreSQL | 예측 및 경고 이력 관리 |
| 시각화 | Grafana | 운영 결과 및 모델 상태 모니터링 |
| 알림 | Slack / n8n | 경고 알림 자동화 |

### 최종 아키텍처 흐름

```text
[Data Ingestion]
   Kafka / API
        |
        v
[Amazon S3]
        |
        v
[Bronze / Silver / Gold]
        |
        v
[Airflow + Spark + Python]
        |
        +--------------------------+
        |                          |
        v                          v
[Model Training]             [Batch Inference]
        |                          ^
        v                          |
[Model Evaluation]                 |
        |                          |
        v                          |
     [MLflow]                      |
        |                          |
        v                          |
[Model Registry]                   |
        |                          |
        v                          |
[Amazon S3 Model Artifacts]        |
        |                          |
        +--------------------------+
                   |
                   v
             [Prediction]
                   |
                   v
          [Python Rule Engine]
                   |
                   v
             [Final Decision]
                   |
          +--------+--------+
          |                 |
          v                 v
    [Iceberg/Gold]     [PostgreSQL]
          |                 |
          +--------+--------+
                   |
                   v
              [Grafana]
                   |
                   v
           [Slack / n8n Alert]
```

Real-time API가 필요해지면 ECS + FastAPI 또는 SageMaker Endpoint를 추가하고, 지속적인 이벤트 처리가 필요해지면 Kafka + Spark Streaming을 확장한다.

---

## 16. 멘티들이 반드시 이해해야 할 핵심 개념

| 개념 | 기억해야 할 내용 |
|---|---|
| Model Training | AI 모델을 학습시키는 과정 |
| Model Evaluation | 학습된 모델의 성능을 검증하는 과정 |
| MLflow | 학습 실험과 모델 버전을 관리하는 도구 |
| Model Registry | 운영에 사용할 모델을 등록하고 관리하는 저장소 |
| Model Serving | 학습된 모델을 운영 환경에서 실행하는 과정 |
| Batch Serving | 일정 주기로 데이터를 모아 예측 |
| Real-time Serving | API 요청에 따라 즉시 예측 |
| Streaming Serving | 지속적으로 들어오는 이벤트를 처리하고 예측 |
| FastAPI | Python 기반 추론 API를 개발하는 프레임워크 |
| Docker | 모델과 실행 환경을 컨테이너로 패키징하는 기술 |
| ECS | AWS에서 Docker 컨테이너를 실행하는 서비스 |
| SageMaker | AWS의 관리형 머신러닝 서비스 |
| Rule Engine | 모델 예측 결과와 업무 규칙을 결합하는 판정 로직 |
| Model Monitoring | 운영 중 모델과 데이터 상태를 지속적으로 확인 |

### 최종 정리

모델을 학습했다고 AI 서비스가 완성되는 것은 아니다.

실제 운영 환경에서는 모델의 버전을 관리하고, 적절한 시점에 추론을 실행하며, 예측 결과를 저장하고, 성능이 저하되는지 지속적으로 확인해야 한다.

TMS 프로젝트에서는 다음 원칙을 중심으로 설계하는 것이 좋다.

1. MLflow를 이용해 모델 학습 실험과 버전을 관리한다.
2. Amazon S3에 승인된 모델과 관련 아티팩트를 저장한다.
3. Airflow와 Python으로 30분 단위 Batch Serving을 먼저 구현한다.
4. 필요하면 FastAPI + ECS 또는 SageMaker를 이용해 API 기반 서빙으로 확장한다.
5. 데이터 발생 빈도와 지연 요구사항이 커지면 Kafka + Spark Streaming을 도입한다.
6. ML 모델의 예측값을 Python Rule Engine에서 검증하고 최종 경고를 결정한다.
7. Grafana를 통해 예측 결과와 모델 성능을 지속적으로 모니터링한다.
8. 재학습과 신규 모델 배포는 평가 및 승인 절차를 거쳐 수행한다.

**가장 중요한 설계 원칙은 모델 학습, 모델 관리, 모델 서빙, 최종 운영 판정을 서로 구분하는 것이다.**

모델은 예측을 담당하고, MLflow는 모델을 관리하며, AWS는 모델이 실제로 실행될 인프라를 제공하고, Rule Engine은 최종 업무 판단을 담당한다.
