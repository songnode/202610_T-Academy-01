
# IJmond 연기 분석 프로젝트: U-Net 기반 Smoke Segmentation 개발 가이드

## 1. 프로젝트 개요

### 1.1 프로젝트 목적

IJmond 산업시설 이미지 데이터셋을 활용하여 굴뚝에서 발생하는 연기 영역을 탐지하고, 연기의 면적과 불투명도 및 시간에 따른 변화를 분석하는 시스템을 개발한다.

이번 프로젝트의 핵심은 단순히 이미지에 연기가 있는지 없는지를 판단하는 것이 아니다.

이미지에서 연기가 차지하는 영역을 픽셀 단위로 분할(Segmentation)하고, 해당 영역의 외곽선(Boundary)을 추출하여 연기의 형태와 크기를 분석하는 것이다.

최종적으로 다음 기능을 구현한다.

1. 산업시설 이미지에서 연기 영역 추출
2. 저불투명도 및 고불투명도 연기 구분
3. 연기 영역의 외곽선(Boundary) 추출
4. 전체 이미지 대비 연기 면적 비율 계산
5. 시간대별 연기 면적 및 불투명도 변화 분석
6. Streamlit 기반 분석 대시보드 구현
7. LangGraph Multi-Agent 기반 자연어 질의 및 분석 결과 설명

### 1.2 전체 개발 방향

| 구분 | 주요 기술 | 개발 목적 |
|---|---|---|
| 이미지 데이터 | IJmond Dataset | 산업시설 이미지 및 정답 마스크 확보 |
| 이미지 전처리 | Python / OpenCV | 이미지 및 마스크 전처리 |
| AI 모델 | U-Net | 연기 영역 Segmentation |
| 특징 추출 | ResNet34 | 이미지 특징 추출 |
| Boundary | OpenCV | 연기 영역의 외곽선 추출 |
| 분석 | Python / Pandas | 연기 면적 및 시간대별 변화 분석 |
| 대시보드 | Streamlit | 분석 결과 시각화 |
| AI Agent | LangGraph | 분석 결과 질의 및 자연어 설명 |

---

## 2. 핵심 개념 이해: Detection, Segmentation, Boundary

이번 프로젝트를 이해하려면 Detection, Segmentation, Boundary의 차이를 먼저 알아야 한다.

### 2.1 Object Detection

Object Detection은 이미지에서 특정 객체가 어디에 있는지 찾아내는 기술이다.

예를 들어 굴뚝에서 연기가 발생하면 연기가 있는 위치를 사각형(Bounding Box)으로 표시한다.

```text
+--------------------------------+
|                                |
|         +-------------+        |
|         |             |        |
|         |    연기     |        |
|         |             |        |
|         +-------------+        |
|                                |
+--------------------------------+

        Bounding Box
```

주요 결과:
- 연기 존재 여부
- 연기 위치
- 연기 영역의 대략적인 크기

대표 모델: YOLO Detection

### 2.2 Segmentation

Segmentation은 이미지의 각 픽셀이 어떤 영역에 속하는지 구분하는 기술이다.

Object Detection이 사각형으로 객체의 위치를 찾는다면, Segmentation은 실제 연기의 모양을 따라 픽셀 단위로 영역을 분할한다.

```text
원본 이미지
    |
    v
Segmentation
    |
    v
연기 영역의 픽셀 단위 분할
```

주요 결과:
- 연기가 차지하는 실제 영역
- 연기 면적
- 연기의 불투명도 단계
- 배경과 연기의 구분

대표 모델: U-Net

### 2.3 Boundary

Boundary는 Segmentation으로 추출한 영역의 외곽 경계를 의미한다.

예를 들어 U-Net이 연기 영역을 분할하면, 해당 영역의 가장자리를 따라 외곽선을 추출할 수 있다.

```text
Segmentation 결과

       연기 영역
      ***********
    ***************
   *****************
    ***************
      ***********

Boundary 결과

       _______
     /         \
    /           \
    \           /
     \_________/
```

주요 분석 결과:
- 연기 외곽 형태
- 연기 영역의 크기
- 연기 확산 범위의 변화
- 연기 영역의 형태 변화

### 2.4 세 가지 기술의 차이

| 구분 | Detection | Segmentation | Boundary |
|---|---|---|---|
| 목적 | 객체 위치 탐지 | 객체 영역 분할 | 영역의 외곽선 추출 |
| 결과 | Bounding Box | Pixel Mask | Contour / Edge |
| 분석 단위 | 사각형 영역 | 픽셀 단위 | 외곽 경계 |
| 연기 면적 계산 | 정확도에 한계 | 적합 | 보조 지표 |
| 연기 형태 분석 | 제한적 | 가능 | 적합 |
| 이번 프로젝트 | 선택 사항 | 핵심 기능 | 핵심 분석 기능 |

**핵심 정리**

- Detection: 연기가 어디에 있는가?
- Segmentation: 연기가 실제로 차지하는 영역은 어디인가?
- Boundary: 연기 영역의 외곽 형태는 어떠한가?

이번 프로젝트에서는 Detection을 별도로 구현하지 않고 U-Net 기반 Segmentation을 핵심으로 개발한다.

---

## 3. U-Net을 선택하는 이유

### 3.1 U-Net이란?

U-Net은 이미지의 각 픽셀이 어떤 영역에 속하는지 구분하는 Semantic Segmentation 모델이다.

이미지를 입력하면 배경과 연기 영역을 픽셀 단위로 구분하여 결과 마스크를 생성한다.

U-Net은 Encoder와 Decoder 구조를 사용한다.

- Encoder: 이미지에서 특징을 추출한다.
- Decoder: 추출된 특징을 이용하여 픽셀 단위 분할 결과를 생성한다.
- Skip Connection: Encoder의 세부적인 위치 정보를 Decoder에 전달한다.

이러한 구조는 불규칙한 형태를 가진 연기 영역을 분할하는 데 활용할 수 있다.

### 3.2 U-Net과 YOLO Segmentation 비교

| 비교 항목 | U-Net | YOLO Segmentation |
|---|---|---|
| 주요 목적 | 이미지 영역 분할 | 객체 탐지 및 분할 |
| 연기 영역 추출 | 적합 | 적합 |
| 불규칙한 연기 형태 분석 | 적합 | 가능 |
| 픽셀 단위 분할 | 지원 | 지원 |
| 불투명도 단계 구분 | 다중 클래스 분할로 구현 | 별도 클래스 설계 필요 |
| 객체 위치 탐지 | 별도 구현 필요 | 기본 지원 |
| 실시간 추론 | 상대적으로 불리 | 상대적으로 유리 |
| 이번 프로젝트 적합성 | 높음 | 확장 개발에 적합 |

### 3.3 최종 모델 선택

이번 프로젝트에서는 연기 객체의 위치를 빠르게 탐지하는 것보다 연기 영역을 정밀하게 분할하고, 불투명도와 면적 변화를 분석하는 것이 중요하다.

따라서 다음 구성을 권장한다.

**U-Net + ResNet34 사전학습 Encoder + IJmond 데이터 Fine-tuning**

YOLO Segmentation은 추후 실시간 객체 탐지나 다른 객체와의 구분이 필요한 경우 확장 모델로 검토한다.

---

## 4. U-Net은 별도 학습이 필요한가?

### 4.1 결론

U-Net을 사용하더라도 IJmond 데이터셋을 이용한 별도 학습이 필요하다.

U-Net은 모델의 구조를 의미하며, 모델 자체가 연기를 이미 알고 있는 것은 아니다.

IJmond 이미지와 정답 마스크를 이용하여 연기 영역을 학습해야 한다.

다만 U-Net 구조를 처음부터 직접 개발할 필요는 없다.

공개된 U-Net 구현체와 사전학습 Encoder를 활용하면 개발 시간을 줄일 수 있다.

### 4.2 사전학습 모델이란?

사전학습(Pretrained Model)은 대량의 데이터로 미리 학습한 모델을 의미한다.

예를 들어 ResNet34는 ImageNet이라는 대규모 이미지 데이터셋으로 사전학습된 모델을 제공한다.

이 모델은 이미지의 기본적인 특징을 추출하는 능력을 가지고 있다.

이번 프로젝트에서는 이러한 특징 추출 능력을 활용하고, IJmond 데이터로 연기 영역을 추가 학습한다.

```text
ImageNet 데이터
       |
       v
ResNet34 사전학습
       |
       v
일반적인 이미지 특징 추출 능력 확보
       |
       v
U-Net Encoder로 활용
       |
       v
IJmond 이미지 및 정답 마스크 학습
       |
       v
연기 분할에 최적화된 모델
```

주의할 점은 ImageNet 사전학습 모델이 연기를 이미 학습한 것은 아니라는 것이다.

연기 분할 기능은 IJmond 데이터로 별도 학습해야 한다.

### 4.3 학습 방식 비교

| 구분 | 학습 방식 | 특징 |
|---|---|---|
| 처음부터 학습 | 무작위 가중치로 모델 학습 | 학습 데이터와 시간이 더 많이 필요할 수 있음 |
| 사전학습 Encoder 활용 | ImageNet 사전학습 가중치 사용 | 학습 부담 감소 |
| Fine-tuning | IJmond 데이터로 추가 학습 | 프로젝트 데이터에 맞게 모델 조정 |

**권장 방식**

U-Net + ResNet34 사전학습 Encoder를 구성하고 IJmond 데이터로 Fine-tuning한다.

초기에는 Encoder를 일부 고정(Freeze)하고 Decoder를 학습한 후, 필요하면 Encoder를 일부 또는 전체 Unfreeze하여 추가 학습할 수 있다.

---

## 5. IJmond 데이터셋 이해

### 5.1 데이터셋 특징

IJmond Industrial Smoke Segmentation Dataset은 산업시설에서 촬영한 이미지와 연기 영역에 대한 Segmentation 정보를 제공한다.

이번 프로젝트에서는 이미지와 해당 이미지에 대응하는 정답 마스크를 사용한다.

정답 마스크는 모델이 학습할 때 비교 기준으로 사용한다.

데이터셋의 이미지와 Crop은 원본 이미지와의 관계를 유지해야 한다.

Crop 이미지가 여러 개 존재하더라도 각각을 독립적인 원본 이미지로 취급해서는 안 된다.

### 5.2 마스크 픽셀 값의 의미

IJmond 마스크는 픽셀 값에 따라 연기의 영역과 불투명도를 구분한다.

| 마스크 픽셀 값 | 의미 | 학습 클래스 |
|---|---|---|
| 0 | 배경 | Background |
| 155 | 저불투명도 연기 | Low-opacity Smoke |
| 255 | 고불투명도 연기 | High-opacity Smoke |

따라서 이번 프로젝트에서는 3개 클래스를 출력하도록 U-Net을 구성한다.

```python
NUM_CLASSES = 3

CLASS_MAPPING = {
    0: "Background",
    155: "Low-opacity Smoke",
    255: "High-opacity Smoke"
}
```

### 5.3 마스크 클래스 변환

모델 학습 시 원본 마스크의 0, 155, 255를 각각 0, 1, 2의 클래스 인덱스로 변환한다.

```python
import numpy as np

def convert_mask(mask):
    """
    IJmond 원본 마스크를
    3개 클래스 인덱스로 변환한다.
    """

    result = np.zeros(mask.shape, dtype=np.uint8)

    result[mask == 0] = 0
    result[mask == 155] = 1
    result[mask == 255] = 2

    return result
```

원본 마스크에 예상하지 못한 픽셀 값이 있는지 먼저 확인해야 한다.

예상하지 못한 값은 임의로 클래스를 지정하지 말고 데이터 검증 단계에서 확인한다.

---

## 6. U-Net 모델 구성

### 6.1 전체 구조

U-Net은 Encoder와 Decoder를 중심으로 구성된다.

```text
                 Input Image
                      |
                      v
              +---------------+
              |    Encoder    |
              |   ResNet34    |
              +---------------+
                      |
                      v
              +---------------+
              |  Bottleneck   |
              +---------------+
                      |
                      v
              +---------------+
              |    Decoder    |
              | Feature Fusion|
              +---------------+
                      |
                      v
              +---------------+
              | Segmentation  |
              |     Head      |
              +---------------+
                      |
                      v
                 Output Mask
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
        배경       저불투명도    고불투명도
```

### 6.2 Encoder

Encoder는 입력 이미지에서 특징을 추출한다.

예를 들어 다음과 같은 특징을 추출하는 데 활용된다.

- 연기의 색상과 명암
- 연기의 경계
- 연기의 형태
- 배경과 연기의 차이
- 연기의 공간적인 패턴

이번 프로젝트에서는 ResNet34를 Encoder로 사용한다.

### 6.3 Decoder

Decoder는 Encoder에서 추출한 특징을 이용하여 이미지의 원래 해상도에 가까운 Segmentation 결과를 생성한다.

최종적으로 각 픽셀에 대해 다음 세 가지 클래스 중 하나를 예측한다.

- Background
- Low-opacity Smoke
- High-opacity Smoke

### 6.4 Skip Connection

U-Net의 중요한 특징 중 하나는 Skip Connection이다.

Encoder에서 추출한 세부적인 위치 정보를 Decoder에 전달하여 연기 영역의 경계를 보다 정밀하게 복원하는 데 도움을 준다.

---

## 7. Segmentation과 Boundary 구현 방법

이번 프로젝트에서 중요한 개발 설계 중 하나다.

**Segmentation과 Boundary를 각각 별도의 AI 모델로 만들 필요는 없다.**

초기 개발에서는 U-Net으로 Segmentation을 수행하고, 그 결과에서 OpenCV로 Boundary를 추출하는 방식으로 구현한다.

### 7.1 전체 처리 흐름

```text
       산업시설 원본 이미지
                 |
                 v
          U-Net Segmentation
                 |
                 v
           Segmentation Mask
                 |
          +------+------+
          |             |
          v             v
    연기 영역 분석    Boundary 추출
          |             |
          v             v
      연기 면적       연기 외곽선
      불투명도        외곽 형태
          |             |
          +------+------+
                 |
                 v
            종합 분석
```

### 7.2 1단계: U-Net으로 Segmentation

U-Net이 입력 이미지를 분석하여 각 픽셀을 세 가지 클래스로 분류한다.

| 클래스 | 결과 |
|---|---|
| 0 | 배경 |
| 1 | 저불투명도 연기 |
| 2 | 고불투명도 연기 |

이 결과를 Segmentation Mask라고 한다.

### 7.3 2단계: 연기 영역 통합

Boundary를 추출하려면 먼저 연기 영역을 하나의 마스크로 통합한다.

저불투명도 연기와 고불투명도 연기를 모두 연기 영역으로 처리한다.

```python
import numpy as np

# prediction:
# 0 = Background
# 1 = Low-opacity Smoke
# 2 = High-opacity Smoke

smoke_mask = np.where(
    prediction > 0,
    255,
    0
).astype(np.uint8)
```

이제 연기 영역은 255, 배경은 0으로 표현된다.

### 7.4 3단계: OpenCV로 Boundary 추출

OpenCV의 findContours()를 이용하여 연기 영역의 외곽선을 추출한다.

```python
import cv2

contours, hierarchy = cv2.findContours(
    smoke_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 원본 이미지에 외곽선 표시
result = original_image.copy()

cv2.drawContours(
    result,
    contours,
    -1,
    (0, 255, 0),
    2
)
```

이렇게 하면 U-Net이 추출한 연기 영역의 외곽선을 이미지에 표시할 수 있다.

### 7.5 Boundary를 별도 학습해야 하는가?

초기 프로젝트에서는 별도 학습이 필요하지 않다.

다음과 같은 경우에만 확장 개발을 검토한다.

- 연기의 아주 미세한 외곽 경계까지 정밀하게 추출해야 하는 경우
- 연기와 구름, 수증기 등의 경계 구분 성능을 높여야 하는 경우
- 경계 자체의 정확도를 별도 지표로 평가해야 하는 경우
- Boundary Loss 등을 적용해 분할 경계 품질을 개선해야 하는 경우

이러한 요구사항은 초기 개발 범위에 포함하지 않는다.

### 7.6 최종 구현 방향

| 기능 | 구현 방법 | 우선순위 |
|---|---|---|
| 연기 영역 분할 | U-Net | 필수 |
| 연기 불투명도 분할 | U-Net 다중 클래스 | 필수 |
| 연기 Boundary 추출 | OpenCV 후처리 | 필수 |
| Boundary 전용 학습 | Boundary-aware 모델 | 확장 |
| 연기 면적 계산 | Python | 필수 |
| 연기 형태 변화 분석 | OpenCV / Pandas | 필수 |

---

## 8. 학습 데이터 구성 및 전처리

### 8.1 입력 데이터 구성

모델 학습에는 다음 두 가지 데이터가 필요하다.

| 데이터 | 설명 |
|---|---|
| Input Image | IJmond 원본 이미지 또는 Crop 이미지 |
| Ground Truth Mask | 해당 이미지에 대응하는 정답 마스크 |

이미지와 마스크는 반드시 동일한 파일 쌍으로 관리해야 한다.

### 8.2 주요 전처리

| 전처리 항목 | 목적 |
|---|---|
| 이미지 크기 조정 | 모델 입력 크기 통일 |
| 이미지 정규화 | 학습 안정성 개선 |
| 마스크 클래스 변환 | 0, 1, 2의 클래스 인덱스로 변환 |
| 데이터 증강 | 다양한 이미지 형태에 대한 일반화 |
| 이미지 검증 | 손상된 이미지 및 마스크 확인 |

주의사항:

- 이미지와 마스크에는 동일한 기하학적 변환을 적용한다.
- 마스크에는 최근접 이웃 보간(Nearest Neighbor)을 사용한다.
- 마스크에는 일반적인 이미지 색상 변환을 적용하지 않는다.
- 학습 전에 이미지와 마스크의 크기 및 파일 매칭을 확인한다.

### 8.3 데이터 증강

데이터 증강(Data Augmentation)은 기존 이미지에 변형을 적용하여 학습 데이터를 다양하게 만드는 방법이다.

사용 가능한 예시는 다음과 같다.

- 좌우 반전
- 상하 반전(촬영 환경에 적합한 경우)
- 회전
- 밝기 및 대비 조정
- 작은 범위의 크기 조정

이미지와 마스크가 서로 다른 형태로 변형되지 않도록 주의한다.

### 8.4 데이터 분할 시 주의사항

IJmond 데이터셋은 원본 이미지에서 여러 Crop 이미지가 만들어질 수 있다.

Crop 이미지를 무작위로 분할하면 동일한 원본 이미지에서 만들어진 Crop이 Train과 Test에 동시에 포함될 수 있다.

이 경우 데이터 누수(Data Leakage)가 발생한다.

반드시 원본 이미지 기준으로 그룹을 나누어야 한다.

```text
Original Image A
 ├── Crop A-1
 ├── Crop A-2
 └── Crop A-3
       |
       +---- 모두 Train

Original Image B
 ├── Crop B-1
 ├── Crop B-2
 └── Crop B-3
       |
       +---- 모두 Validation
```

가능하면 IJmond 데이터셋의 공식 Timestamp Split 또는 Camera Split 기준을 우선 검토한다.

공식 분할 기준을 사용할 경우에도 원본 이미지와 Crop의 관계를 확인하고 동일한 원본에서 파생된 데이터가 서로 다른 분할에 들어가지 않도록 관리한다.

---

## 9. U-Net 모델 학습

### 9.1 전체 학습 과정

```text
1. IJmond 이미지 및 마스크 로딩
                 |
                 v
2. 이미지 전처리 및 클래스 변환
                 |
                 v
3. Train / Validation / Test 분리
                 |
                 v
4. U-Net + ResNet34 모델 생성
                 |
                 v
5. ImageNet 사전학습 가중치 적용
                 |
                 v
6. IJmond 데이터로 모델 학습
                 |
                 v
7. Validation 성능 확인
                 |
                 v
8. 최적 모델 저장
                 |
                 v
9. Test 데이터 최종 평가
                 |
                 v
10. 신규 이미지 추론
```

### 9.2 학습 구성

| 항목 | 권장 구성 | 설명 |
|---|---|---|
| Model | U-Net | Segmentation 모델 |
| Encoder | ResNet34 | 이미지 특징 추출 |
| Pretrained | ImageNet | 사전학습 가중치 |
| Classes | 3 | 배경, 저불투명도, 고불투명도 |
| Loss | Cross Entropy + Dice Loss | 클래스 분류 및 영역 중첩 개선 |
| Optimizer | AdamW | 가중치 최적화 |
| Evaluation | IoU, Dice | 분할 성능 평가 |

Loss 함수는 클래스 불균형과 실제 데이터 특성을 고려해 조정한다.

### 9.3 학습과 Fine-tuning의 차이

이번 프로젝트의 Fine-tuning은 기존 ImageNet 사전학습 Encoder를 IJmond 데이터에 맞게 추가 학습하는 과정이다.

```text
ImageNet 사전학습 ResNet34
                 |
                 v
             U-Net 구성
                 |
                 v
           IJmond 데이터 학습
                 |
                 v
          연기 분할 모델 완성
```

초기에는 Encoder를 일부 Freeze하여 Decoder 중심으로 학습할 수 있다.

이후 성능 개선이 필요하면 Encoder를 Unfreeze하여 추가 학습한다.

---

## 10. 모델 성능 평가

학습이 끝나면 Test 데이터로 모델 성능을 평가한다.

단순히 Accuracy만 확인해서는 안 된다.

연기 영역은 배경에 비해 픽셀 수가 적을 수 있으므로 영역 분할 성능을 직접 평가하는 지표가 필요하다.

### 10.1 평가 지표

| 평가 지표 | 의미 |
|---|---|
| IoU | 예측 영역과 정답 영역의 교집합 비율 |
| Dice Score | 예측 영역과 정답 영역의 중첩 정도 |
| Precision | 연기로 예측한 영역 중 실제 연기인 비율 |
| Recall | 실제 연기 영역 중 모델이 찾아낸 비율 |
| Class-wise IoU | 클래스별 분할 성능 |

### 10.2 클래스별 평가

다음 세 가지 클래스를 구분하여 평가한다.

- Background
- Low-opacity Smoke
- High-opacity Smoke

특히 저불투명도 연기와 고불투명도 연기의 성능을 별도로 확인해야 한다.

전체 평균 점수만 높고 특정 연기 클래스의 성능이 낮을 수 있기 때문이다.

### 10.3 Boundary 평가

초기에는 Boundary를 별도로 학습하지 않으므로 Boundary 전용 학습 지표를 핵심 평가 기준으로 삼지 않는다.

우선 Segmentation의 IoU와 Dice를 평가하고, 필요하면 추출된 외곽선과 정답 마스크에서 계산한 경계를 비교하는 Boundary F1 Score 등을 확장 지표로 추가한다.

---

## 11. 학습된 모델을 활용한 연기 분석

U-Net의 최종 결과는 이미지와 동일한 공간 구조를 가진 Segmentation Mask다.

이 마스크를 활용하면 연기 면적과 불투명도를 계산할 수 있다.

### 11.1 Smoke Ratio

전체 이미지에서 연기가 차지하는 비율이다.

```text
Smoke Ratio (%)
=
(저불투명도 연기 픽셀 수 + 고불투명도 연기 픽셀 수)
/
전체 유효 이미지 픽셀 수
× 100
```

### 11.2 High Opacity Ratio

전체 이미지에서 고불투명도 연기가 차지하는 비율이다.

```text
High Opacity Ratio (%)
=
고불투명도 연기 픽셀 수
/
전체 유효 이미지 픽셀 수
× 100
```

### 11.3 High Opacity Share

전체 연기 영역 중 고불투명도 연기가 차지하는 비중이다.

```text
High Opacity Share (%)
=
고불투명도 연기 픽셀 수
/
전체 연기 픽셀 수
× 100
```

연기 영역이 없는 경우 High Opacity Share는 0으로 처리하지 않고 NULL 또는 별도 상태로 처리한다.

### 11.4 Boundary 기반 분석

추출한 외곽선을 이용해 다음 항목을 분석할 수 있다.

| 분석 항목 | 설명 |
|---|---|
| Contour Area | 연기 외곽선으로 둘러싸인 영역의 면적 |
| Perimeter | 연기 영역의 외곽선 길이 |
| Bounding Box | 연기 영역의 최소 사각형 영역 |
| Aspect Ratio | 연기 영역의 가로·세로 비율 |
| Shape Change | 시간에 따른 연기 형태 변화 |

주의사항:

이미지 픽셀로 계산한 면적은 실제 물리적인 연기 면적과 동일하지 않다.

카메라와 굴뚝 사이의 거리, 촬영 각도, 해상도, 원근 효과 등이 영향을 주기 때문이다.

따라서 별도의 카메라 보정이나 거리 정보가 없다면 픽셀 기반 면적 및 비율로 표현한다.

### 11.5 시간대별 연기 변화 분석

시간 정보가 있는 이미지들을 순서대로 비교하여 연기 영역의 변화를 분석한다.

예시:

| 촬영 시점 | Smoke Ratio | High Opacity Ratio |
|---|---:|---:|
| 09:00 | 12% | 3% |
| 09:30 | 18% | 6% |
| 10:00 | 28% | 15% |

위 수치는 설명을 위한 가상 데이터다.

이러한 분석을 통해 다음을 확인할 수 있다.

- 연기 영역이 증가하는지 감소하는지
- 고불투명도 연기의 비중이 증가하는지
- 특정 시간대에 연기 영역이 크게 변하는지
- 카메라별 연기 관측 패턴이 어떠한지

단, IJmond 데이터는 연속 동영상이 아니라 시간 정보가 있는 이미지 데이터다.

따라서 실제 연기 지속시간이나 프레임 간 움직임을 직접 측정한다고 해석해서는 안 된다.

정확한 표현은 **Timestamp 기반 이미지 시계열 분석**이다.

---

## 12. 전체 시스템 아키텍처

```text
                   IJmond Dataset
                         |
                         v
                Image / Mask Loading
                         |
                         v
                Image Preprocessing
                         |
                         v
                 U-Net Model Training
                  (ResNet34 Encoder)
                         |
                         v
                   Trained Model
                         |
                         v
                   Image Inference
                         |
                         v
                  Segmentation Mask
                         |
               +---------+---------+
               |                   |
               v                   v
        Smoke Area Analysis   Boundary Extraction
               |                   |
               v                   v
          Smoke Ratio        Contour / Perimeter
          Opacity Ratio      Shape Analysis
               |                   |
               +---------+---------+
                         |
                         v
                 Timestamp Analysis
                         |
                         v
                   Analysis Results
                         |
               +---------+---------+
               |                   |
               v                   v
            Streamlit           LangGraph
            Dashboard          Multi-Agent
               |                   |
               v                   v
          결과 시각화         자연어 질의 및 설명
```

### 12.1 구성 요소별 역할

| 구성 요소 | 주요 역할 |
|---|---|
| IJmond Dataset | 산업시설 이미지와 정답 마스크 제공 |
| Preprocessing | 이미지 및 마스크 전처리 |
| U-Net | 연기 영역 및 불투명도 분할 |
| ResNet34 | U-Net Encoder 특징 추출 |
| OpenCV | Boundary 및 외곽선 추출 |
| Python | 연기 면적 및 비율 계산 |
| Pandas | 시간대별 변화 분석 |
| Streamlit | 분석 결과 시각화 |
| LangGraph | 분석 결과 질의 및 자연어 설명 |

---

## 13. GitHub 프로젝트 개발 단계

| 단계 | 개발 내용 | 주요 산출물 |
|---|---|---|
| 1단계 | IJmond 데이터 다운로드 및 구조 확인 | 데이터셋 Manifest |
| 2단계 | 이미지 및 마스크 검증 | 데이터 검증 결과 |
| 3단계 | 이미지 전처리 및 클래스 변환 | 전처리 데이터 |
| 4단계 | Train / Validation / Test 분리 | 데이터 분할 목록 |
| 5단계 | U-Net + ResNet34 구성 | 모델 코드 |
| 6단계 | IJmond 데이터 학습 | 학습된 모델 |
| 7단계 | IoU, Dice 등 성능 평가 | 모델 평가 보고서 |
| 8단계 | Segmentation 및 Boundary 구현 | 마스크 및 외곽선 결과 |
| 9단계 | 연기 비율 및 불투명도 계산 | 분석 결과 데이터 |
| 10단계 | Timestamp 기반 변화 분석 | 시계열 분석 결과 |
| 11단계 | Streamlit 대시보드 구현 | 분석 화면 |
| 12단계 | LangGraph 연동 | 자연어 질의 기능 |

---

## 14. 최종 개발 범위

### 14.1 핵심 개발 범위

이번 프로젝트에서 반드시 구현할 기능이다.

| 구분 | 기능 | 적용 기술 |
|---|---|---|
| 데이터 | IJmond 이미지 및 마스크 로딩 | Python |
| 전처리 | 이미지 정규화 및 마스크 변환 | OpenCV |
| 모델 | 연기 영역 Segmentation | U-Net |
| 모델 | 사전학습 Encoder 활용 | ResNet34 / ImageNet |
| 학습 | IJmond 데이터 Fine-tuning | PyTorch |
| 평가 | 모델 성능 평가 | IoU / Dice |
| 후처리 | Boundary 추출 | OpenCV |
| 분석 | 연기 면적 및 불투명도 | Python |
| 분석 | 시간대별 변화 | Pandas |
| 시각화 | 분석 결과 대시보드 | Streamlit |

### 14.2 확장 개발 범위

핵심 기능이 안정화된 이후 추가할 기능이다.

| 구분 | 기능 | 적용 기술 |
|---|---|---|
| 모델 | Boundary-aware Segmentation | U-Net 확장 |
| 모델 | YOLO Segmentation 비교 | YOLO |
| 분석 | 연기 형태의 정밀 분석 | OpenCV |
| 분석 | 카메라별 패턴 비교 | Pandas |
| AI Agent | 분석 결과 자연어 질의 | LangGraph |
| AI Agent | 이상 패턴 요약 및 설명 | LLM |
| 통합 | TMS NOx/SO2 데이터와 나란히 비교 | Python / Dashboard |

TMS 데이터와 IJmond 이미지가 동일 사업장 및 동일 시각으로 매칭되지 않는다면, 두 데이터의 직접적인 상관관계나 융합 모델 성능을 주장하지 않는다.

우선 별도의 분석 결과를 나란히 보여주는 비교 대시보드로 구현한다.

---

## 15. 최종 권장 기술 구성

| 영역 | 최종 선택 | 적용 목적 |
|---|---|---|
| Segmentation | U-Net | 연기 영역 분할 |
| Encoder | ResNet34 | 이미지 특징 추출 |
| Pretrained | ImageNet | 사전학습 가중치 활용 |
| 추가 학습 | IJmond Fine-tuning | 연기 분할 학습 |
| 출력 클래스 | 3개 | 배경 / 저불투명도 / 고불투명도 |
| Loss | Cross Entropy + Dice Loss | 분할 성능 개선 |
| Optimizer | AdamW | 모델 최적화 |
| 평가 | IoU / Dice / Precision / Recall | 모델 성능 검증 |
| Boundary | OpenCV | 연기 외곽선 추출 |
| 이미지 분석 | Python / OpenCV | 연기 면적 및 형태 분석 |
| 시계열 분석 | Pandas | 시간대별 변화 분석 |
| 대시보드 | Streamlit | 분석 결과 시각화 |
| AI Agent | LangGraph | 자연어 질의 및 분석 설명 |

---

## 16. 최종 정리

이번 프로젝트의 핵심 개발 방향은 다음과 같다.

1. **U-Net으로 연기 영역을 Segmentation한다.**
   - 배경, 저불투명도 연기, 고불투명도 연기를 픽셀 단위로 분류한다.

2. **ResNet34 사전학습 Encoder를 활용한다.**
   - ImageNet에서 사전학습된 가중치를 사용한다.
   - IJmond 데이터로 추가 학습(Fine-tuning)한다.

3. **Boundary는 별도 모델 없이 추출한다.**
   - U-Net의 Segmentation 결과를 이용한다.
   - OpenCV의 findContours()로 연기 외곽선을 추출한다.

4. **연기 영역과 외곽선을 분석한다.**
   - Smoke Ratio
   - High Opacity Ratio
   - High Opacity Share
   - Contour Area
   - Perimeter

5. **시간대별 연기 변화를 분석한다.**
   - Timestamp 기반 이미지 시계열 분석을 수행한다.
   - 연기 면적과 불투명도 변화 추이를 확인한다.

6. **Streamlit과 LangGraph를 연동한다.**
   - Streamlit은 분석 결과를 시각화한다.
   - LangGraph는 분석 결과를 조회하고 자연어로 설명한다.

### 최종 개발 방향

**U-Net + ResNet34 사전학습 Encoder를 기반으로 IJmond 데이터에 맞게 Fine-tuning하고, Segmentation 결과에서 OpenCV로 Boundary를 추출한다. 이후 연기 면적·불투명도·외곽 형태·시간대별 변화를 분석하여 산업시설 연기 분석 시스템을 구축한다.**

핵심은 AI 모델이 연기 영역을 분할하고, Python/OpenCV가 수치와 외곽선을 계산하며, LangGraph Agent가 분석 결과를 설명하도록 역할을 명확히 구분하는 것이다.
