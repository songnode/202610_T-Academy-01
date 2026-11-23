# IJmond 연기 분석 
## U-Net 기반 Smoke Segmentation 전담 Agent 개발 가이드

---

# 1. 프로젝트 개요

## 1.1 프로젝트 목적

본 프로젝트는 산업시설에서 촬영한 이미지를 활용하여 연기 영역을 자동으로 분할하고, 연기의 면적과 불투명도 등을 정량적으로 분석하는 AI 시스템을 구축하는 것을 목표로 한다.

IJmond Industrial Smoke Segmentation Dataset을 활용하여 U-Net 기반 Smoke Segmentation 모델을 학습하고, 학습된 모델을 Smoke Segmentation 전담 Agent에 연결한다.

사용자가 TMS 시스템에 산업시설 사진과 사건 발생일자를 등록하면, Smoke Segmentation Agent가 해당 정보를 전달받아 이미지 분석을 수행하고 결과를 TMS에 전달하는 구조로 개발한다.

핵심은 단순히 이미지에서 연기를 찾아내는 것이 아니라, 연기 영역을 픽셀 단위로 분할하고 분석 결과를 정량적인 데이터로 만들어 관리하는 것이다.

### 핵심 개발 목표

1. IJmond 데이터셋을 활용한 U-Net 기반 연기 분할 모델 개발
2. 연기 영역을 Background / Low Opacity / High Opacity로 구분
3. 연기 영역의 면적, 비율, 불투명도 및 경계 정보 추출
4. Smoke Segmentation 전담 Agent 개발
5. TMS에서 등록한 사진과 사건 발생일자를 Agent에 전달하는 연계 구조 구현
6. 분석 결과를 저장하고 조회할 수 있는 데이터 구조 마련
7. 사건 발생일자 기준으로 이미지 분석 결과의 시간적 변화 확인

## 1.2 전체 서비스 개념

전체 시스템은 TMS와 Smoke Segmentation Agent가 역할을 분담하는 구조로 설계한다.

- TMS: 사진과 사건 발생일자 등록, 분석 요청, 분석 결과 조회
- Smoke Segmentation Agent: 이미지 분석 요청 수신, 모델 추론, 연기 영역 분할, 분석 지표 생성
- U-Net Model: 이미지의 각 픽셀을 연기 유형별로 분류
- Analysis Engine: 분할 결과를 이용하여 연기 면적과 불투명도 등 계산
- Data Storage: 원본 이미지, 분석 결과, 마스크 및 분석 이력 저장

### 전체 처리 흐름

```text
┌──────────────────────────────┐
│             TMS              │
│                              │
│  1. 산업시설 사진 등록        │
│  2. 사건 발생일자 등록        │
│  3. 분석 요청                 │
└──────────────┬───────────────┘
               │
               │ 이미지 + 사건 발생일자
               ▼
┌──────────────────────────────┐
│   Smoke Segmentation Agent   │
│                              │
│  1. 분석 요청 확인            │
│  2. 입력 데이터 검증          │
│  3. 이미지 전처리             │
│  4. U-Net 모델 호출           │
│  5. 연기 영역 Segmentation    │
│  6. Boundary 추출             │
│  7. 분석 지표 계산            │
│  8. 결과 저장 및 전달         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       U-Net Segmentation     │
│                              │
│  Background                  │
│  Low Opacity Smoke            │
│  High Opacity Smoke           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Analysis Engine        │
│                              │
│  Smoke Ratio                 │
│  High Opacity Ratio          │
│  High Opacity Share          │
│  Contour Area                │
│  Perimeter                   │
│  Bounding Box                │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Analysis Storage       │
│                              │
│  원본 이미지 정보             │
│  Segmentation Mask           │
│  분석 지표                    │
│  사건 발생일자                │
│  Agent 실행 이력              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│             TMS              │
│                              │
│  분석 결과 조회 및 활용       │
└──────────────────────────────┘
```

---

# 2. Smoke Segmentation의 기본 개념

## 2.1 Detection, Segmentation, Boundary의 차이

연기 분석 시스템을 개발하기 전에 Detection, Segmentation, Boundary의 차이를 이해해야 한다.

| 구분 | 설명 | 결과 |
|---|---|---|
| Detection | 이미지에서 연기가 존재하는 위치를 찾음 | Bounding Box |
| Segmentation | 이미지의 모든 픽셀을 분석하여 연기 영역을 구분 | Segmentation Mask |
| Boundary | 분할된 연기 영역의 외곽선을 추출 | Contour |

### Detection

Detection은 이미지에서 연기가 존재하는 위치를 사각형 영역으로 표시하는 기술이다.

예를 들어 연기가 이미지의 오른쪽 상단에 존재한다면 해당 영역을 Bounding Box로 표시한다.

다만 Bounding Box 내부에 연기가 아닌 배경이 포함될 수 있다.

### Segmentation

Segmentation은 이미지의 모든 픽셀을 분석하여 각 픽셀이 어떤 영역에 해당하는지 분류하는 기술이다.

예를 들어 다음과 같이 구분할 수 있다.

- Background: 연기가 아닌 영역
- Low Opacity Smoke: 불투명도가 낮은 연기
- High Opacity Smoke: 불투명도가 높은 연기

Segmentation을 적용하면 연기의 정확한 형태와 면적을 픽셀 단위로 계산할 수 있다.

### Boundary

Boundary는 Segmentation 결과로 만들어진 연기 영역의 외곽선을 의미한다.

연기 영역의 형태와 크기를 분석하거나 이미지 위에 연기 경계를 시각적으로 표시할 때 활용한다.

**이번 프로젝트에서는 U-Net으로 Segmentation을 수행하고, OpenCV를 이용하여 Boundary를 후처리하는 방식으로 개발한다.**

초기 개발 단계에서는 Boundary를 별도의 AI 모델로 학습하지 않는다.

---

# 3. U-Net을 사용하는 이유

## 3.1 U-Net이란?

U-Net은 이미지의 각 픽셀을 분류하는 Semantic Segmentation 모델이다.

이미지를 입력받아 픽셀 단위로 영역을 구분한 결과를 출력한다.

대표적인 활용 분야는 다음과 같다.

- 의료 영상에서 장기 및 병변 분할
- 위성 이미지에서 도로 및 건물 분할
- 제조업에서 제품 결함 영역 분할
- 산업시설 이미지에서 연기 영역 분할

이번 프로젝트는 연기의 위치를 대략적으로 찾는 것보다 연기 영역을 정확하게 구분하고 불투명도별로 분석하는 것이 중요하다.

따라서 U-Net을 기본 모델로 선정한다.

## 3.2 U-Net과 YOLO Segmentation 비교

| 비교 항목 | U-Net | YOLO Segmentation |
|---|---|---|
| 주요 목적 | 픽셀 단위 영역 분할 | 객체 탐지와 분할 |
| 출력 결과 | 픽셀별 클래스 | 객체별 마스크 및 탐지 정보 |
| 장점 | 세밀한 영역 분할에 적합 | 탐지와 분할을 함께 수행 |
| 활용 분야 | 의료 영상, 연기, 결함 분석 | 실시간 객체 탐지 |
| 이번 프로젝트 적합성 | 높음 | 확장 기능으로 검토 |

### 모델 선정 방향

**1차 개발 모델은 U-Net으로 진행한다.**

YOLO Segmentation은 향후 실시간 영상 분석이나 여러 객체를 개별적으로 탐지해야 하는 경우 확장 모델로 검토할 수 있다.

## 3.3 추천 모델 구성

이번 프로젝트에서는 다음 구성을 권장한다.

```text
U-Net
 └── ResNet34 Encoder (ImageNet Pretrained)
      └── U-Net Decoder
           └── 3-Class Segmentation Output
```

| 구성 요소 | 역할 |
|---|---|
| U-Net | 전체 Segmentation 구조 |
| ResNet34 Encoder | 이미지 특징 추출 |
| ImageNet Pretrained | 사전 학습된 이미지 특징 활용 |
| U-Net Decoder | 픽셀 단위 분할 결과 복원 |
| Output Layer | 3개 클래스별 픽셀 확률 출력 |

### Pretrained 모델에 대한 이해

ResNet34 Encoder를 ImageNet으로 사전 학습된 모델로 사용한다.

여기서 주의할 점은 ImageNet 사전 학습 모델이 이미 연기를 인식하도록 학습된 모델은 아니라는 것이다.

ImageNet Pretrained는 일반적인 이미지 특징을 추출하는 능력을 제공한다.

이후 IJmond 데이터셋의 이미지와 정답 마스크를 이용하여 연기 영역을 구분하도록 Fine-tuning을 수행해야 한다.

---

# 4. IJmond 데이터셋 이해

## 4.1 데이터셋 소개

프로젝트에서는 IJmond Industrial Smoke Segmentation Dataset을 활용한다.

이 데이터셋은 산업시설에서 촬영한 이미지와 연기 영역에 대한 Segmentation Mask를 제공한다.

데이터셋에는 연기가 포함된 이미지와 연기가 없는 이미지가 있으며, 연기의 불투명도에 따라 구분된 마스크 정보가 포함되어 있다.

- 데이터셋: IJmond Industrial Smoke Segmentation Dataset
- 데이터 출처: Figshare
- 논문: arXiv
- 라이선스: CC BY 4.0

참고 링크:

- Dataset: https://uvaauas.figshare.com/articles/dataset/IJmond_Industrial_Smoke_Segmentation_Dataset/31847188
- Paper: https://arxiv.org/abs/2603.23754

## 4.2 데이터셋 구성

현재 확인된 데이터셋의 주요 구성은 다음과 같다.

| 항목 | 수량 |
|---|---:|
| 원본 이미지 | 900장 |
| Polygon Annotation | 1,209개 |
| 연기가 없는 원본 이미지 | 7장 |
| Crop 이미지 | 2,074장 |
| Smoke Mask가 포함된 Crop | 1,109장 |

주의할 점은 원본 이미지와 Crop 이미지가 서로 독립적인 데이터가 아니라는 것이다.

Crop 이미지는 원본 이미지에서 잘라낸 데이터이므로 두 데이터를 합산하여 전체 이미지 수로 표현하면 안 된다.

## 4.3 카메라 정보

데이터셋에는 다음과 같은 카메라 정보가 포함되어 있다.

- kooks_1
- kooks_2
- hoogovens_6_7

카메라별로 촬영 환경과 연기 형태가 다를 수 있다.

따라서 모델이 특정 카메라의 이미지에만 과도하게 적응하지 않는지 확인해야 한다.

## 4.4 Mask 데이터 이해

Segmentation 학습에서 가장 중요한 데이터는 이미지와 정답 Mask이다.

Mask는 원본 이미지의 각 픽셀이 어떤 영역에 해당하는지를 표시한 정답 데이터다.

현재 데이터셋의 Mask 값은 다음과 같다.

| 원본 Mask 값 | 의미 | 학습용 Class |
|---:|---|---:|
| 0 | Background | 0 |
| 155 | Low Opacity Smoke | 1 |
| 255 | High Opacity Smoke | 2 |

모델은 원본 이미지와 Mask를 함께 학습한다.

예를 들어 다음과 같은 구조다.

```text
Original Image
      │
      ▼
   U-Net
      │
      ▼
Predicted Mask

Ground Truth Mask
      │
      ▼
Loss Calculation
      │
      ▼
Model Training
```

모델이 예측한 Mask와 실제 정답 Mask를 비교하면서 학습한다.

## 4.5 with_masks와 without_masks 폴더 확인

폴더 이름만 보고 데이터의 의미를 판단하면 안 된다.

특히 다음 사항을 확인해야 한다.

- with_masks 폴더에 실제 정답 Mask가 존재하는가?
- without_masks 폴더의 이미지에 대응하는 Mask가 있는가?
- 연기가 없는 이미지는 검은색 Mask로 표현되어 있는가?
- Mask가 없다는 것과 연기가 없다는 것이 구분되어 있는가?

연기가 없는 이미지는 Background로 학습할 수 있지만, 정답 Mask 자체가 없는 이미지는 지도학습에 그대로 사용할 수 없다.

따라서 학습 전에 이미지와 Mask의 파일명 및 대응 관계를 확인해야 한다.

---

# 5. 데이터 전처리 및 데이터 증강

## 5.1 전체 전처리 순서

Segmentation 모델은 이미지와 Mask의 대응 관계가 매우 중요하다.

따라서 데이터 전처리는 다음 순서로 진행한다.

```text
원본 이미지 및 Mask 확보
        │
        ▼
이미지와 Mask 파일 매핑
        │
        ▼
원본 이미지 기준 Train / Validation / Test 분리
        │
        ▼
이미지 크기 조정
        │
        ▼
Mask 클래스 변환
        │
        ▼
Train 데이터 증강
        │
        ▼
이미지 정규화
        │
        ▼
U-Net 학습
```

## 5.2 Train / Validation / Test 분리

데이터를 무작위로 나누기 전에 반드시 원본 이미지 기준으로 분리해야 한다.

예를 들어 원본 이미지 A에서 만들어진 Crop A-1, A-2, A-3이 있다고 가정한다.

이 Crop들이 서로 다른 Train과 Test에 들어가면 동일한 원본 이미지의 특징이 학습과 평가에 동시에 포함된다.

이러한 문제를 Data Leakage라고 한다.

따라서 다음 원칙을 적용한다.

1. 원본 이미지 단위로 Train / Validation / Test를 분리한다.
2. 원본 이미지에서 파생된 Crop은 모두 같은 데이터 그룹에 포함한다.
3. Train에 포함된 원본 이미지의 Crop만 학습에 사용한다.
4. Test에 포함된 원본 이미지의 Crop은 최종 평가에만 사용한다.

### 권장 데이터 분리

데이터셋에서 제공하는 공식 분할 기준이 있다면 이를 우선 확인하고 활용한다.

기본적으로 다음과 같은 시간 순서 분할을 고려할 수 있다.

| 구분 | 데이터 비율 | 목적 |
|---|---:|---|
| Train | 70% | 모델 학습 |
| Validation | 10% | 모델 성능 확인 및 설정 조정 |
| Test | 20% | 최종 성능 평가 |

시간 순서 분할을 적용할 때는 원본 이미지의 촬영 시간 기준으로 정렬한다.

공식 카메라 분할 기준도 제공되므로 별도로 확인해야 한다.

예를 들어 kooks_2의 시간 순서상 앞 80%를 학습에 사용하고 나머지 20%를 검증에 사용하며, hoogovens_6_7과 kooks_1을 테스트에 사용하는 방식이 제시되어 있다.

시간 순서 분할과 카메라 분할은 평가 목적이 다르므로 임의로 혼합하지 말고, 데이터셋의 공식 분할 기준을 확인하여 실험 조건을 명확히 기록한다.

## 5.3 이미지 크기 조정

원본 이미지의 크기가 서로 다르면 모델 학습을 위해 동일한 크기로 변환해야 한다.

예를 들어 다음과 같이 설정할 수 있다.

```python
IMAGE_SIZE = 512

image = cv2.resize(
    image,
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=cv2.INTER_LINEAR
)

mask = cv2.resize(
    mask,
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=cv2.INTER_NEAREST
)
```

이미지와 Mask는 동일한 크기로 변환해야 한다.

특히 Mask는 클래스 값이 유지되어야 하므로 반드시 Nearest Neighbor 보간법을 사용한다.

## 5.4 Mask 클래스 변환

모델 학습에 사용하기 쉽도록 원본 Mask 값을 0, 1, 2로 변환한다.

```python
import numpy as np

def convert_mask(mask):
    converted = np.zeros(mask.shape, dtype=np.uint8)

    converted[mask == 155] = 1
    converted[mask == 255] = 2

    return converted
```

변환 후 클래스가 정상적으로 유지되는지 확인한다.

```python
print(np.unique(converted))
```

정상적인 결과는 다음과 같다.

```text
[0 1 2]
```

## 5.5 데이터 증강(Data Augmentation)

데이터 증강은 기존 이미지에 다양한 변화를 주어 모델이 더 다양한 이미지 환경에 대응할 수 있도록 학습 데이터를 확장하는 방법이다.

산업시설의 연기는 촬영 시간, 조명, 날씨, 카메라 환경에 따라 형태와 색상이 달라질 수 있다.

따라서 적절한 데이터 증강을 적용하는 것이 좋다.

단, 증강을 많이 한다고 반드시 모델 성능이 좋아지는 것은 아니다.

연기의 실제 특징을 훼손하지 않는 범위에서 적용해야 한다.

### 권장 데이터 증강

| 증강 기법 | 적용 내용 | 권장 여부 |
|---|---|---|
| Horizontal Flip | 좌우 반전 | 권장 |
| Rotation | 약한 회전 | 권장 |
| Brightness | 밝기 조정 | 권장 |
| Contrast | 대비 조정 | 권장 |
| Gaussian Blur | 약한 흐림 효과 | 선택 |
| Random Crop | 일부 영역 자르기 | 신중하게 적용 |
| 강한 Color 변환 | 색상 과도 변경 | 비권장 |
| 강한 Rotation | 큰 각도 회전 | 비권장 |

### 증강 적용 예시

```python
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),

    A.Rotate(
        limit=10,
        p=0.3,
        border_mode=0,
        mask_interpolation=0
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.3
    ),

    A.GaussianBlur(
        blur_limit=(3, 3),
        p=0.1
    )
])
```

실제 적용 시에는 설치된 Albumentations 버전에 맞춰 API를 확인한다.

### 이미지와 Mask를 함께 증강해야 하는 이유

Segmentation에서는 이미지와 Mask가 정확하게 대응해야 한다.

예를 들어 원본 이미지에서 연기가 오른쪽에 있다면 Mask에서도 동일한 위치에 연기 영역이 있어야 한다.

이미지만 좌우 반전하고 Mask를 그대로 두면 이미지와 정답 데이터가 서로 맞지 않게 된다.

따라서 기하학적 변환은 이미지와 Mask에 동일하게 적용해야 한다.

```python
augmented = train_transform(
    image=image,
    mask=mask
)

train_image = augmented["image"]
train_mask = augmented["mask"]
```

주의사항:

- Flip, Rotation, Crop 등은 이미지와 Mask에 동일하게 적용한다.
- Brightness, Contrast 등은 이미지에만 적용한다.
- Mask는 보간 과정에서 클래스 값이 변하지 않도록 Nearest Neighbor를 사용한다.
- 증강 후 Mask 클래스 값이 0, 1, 2로 유지되는지 확인한다.

### 데이터 증강 적용 원칙

**Train에만 랜덤 데이터 증강을 적용한다.**

| 데이터 | 랜덤 증강 | 일반 전처리 |
|---|---|---|
| Train | 적용 | 적용 |
| Validation | 미적용 | 적용 |
| Test | 미적용 | 적용 |

Validation과 Test에 랜덤 증강을 적용하면 실제 평가 조건이 달라질 수 있다.

초기에는 데이터 증강을 적용하지 않은 Baseline 모델과 증강을 적용한 모델을 비교하는 것이 좋다.

최종적으로 Validation IoU와 Dice가 개선되는지 확인하여 증강 방법을 결정한다.

## 5.6 이미지 정규화

이미지 픽셀 값은 일반적으로 0~255 범위를 사용한다.

모델 학습을 위해 0~1 범위로 변환할 수 있다.

```python
image = image.astype("float32") / 255.0
```

단, ImageNet Pretrained Encoder를 사용할 경우에는 해당 모델의 학습 기준에 맞는 평균과 표준편차를 사용해야 한다.

정규화 방식은 학습과 추론에서 동일하게 적용해야 한다.

---

# 6. U-Net 모델 구조

## 6.1 U-Net 전체 구조

U-Net은 크게 Encoder와 Decoder로 구성된다.

```text
             Input Image
                  │
                  ▼
             Encoder
          (Feature Extraction)
                  │
                  ▼
              Bottleneck
                  │
                  ▼
              Decoder
          (Feature Restoration)
                  │
                  ▼
            Output Mask
                  │
                  ▼
       Background / Low / High
```

## 6.2 Encoder

Encoder는 이미지에서 특징을 추출하는 역할을 수행한다.

초기 이미지에서는 색상, 선, 형태 등을 분석하고 점차 복잡한 특징을 학습한다.

이번 프로젝트에서는 ResNet34를 Encoder로 사용한다.

## 6.3 Bottleneck

Encoder에서 추출한 특징을 압축하고 중요한 정보를 학습하는 구간이다.

## 6.4 Decoder

Decoder는 Encoder에서 추출한 특징을 활용하여 원래 이미지의 공간 정보에 가까운 형태로 복원한다.

최종적으로 각 픽셀의 클래스별 확률을 출력한다.

## 6.5 Skip Connection

U-Net의 중요한 특징 중 하나다.

Encoder에서 추출한 공간 정보를 Decoder에 전달하여 이미지의 세부적인 특징을 복원하는 데 도움을 준다.

연기 영역의 경계가 흐릿하거나 형태가 복잡한 경우에도 공간 정보를 활용할 수 있다.

## 6.6 Output Layer

최종 출력은 3개 클래스로 구성한다.

| 클래스 | 의미 |
|---:|---|
| 0 | Background |
| 1 | Low Opacity Smoke |
| 2 | High Opacity Smoke |

모델 출력은 픽셀마다 3개 클래스의 확률을 갖는다.

예를 들어 특정 픽셀의 출력이 다음과 같다고 가정한다.

```text
Background        : 0.05
Low Opacity Smoke : 0.80
High Opacity Smoke: 0.15
```

가장 높은 확률인 Low Opacity Smoke로 분류한다.

---

# 7. U-Net 모델 학습

## 7.1 학습 환경

권장 개발 환경은 다음과 같다.

| 구분 | 기술 |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Segmentation | U-Net |
| Encoder | ResNet34 |
| Pretrained | ImageNet |
| Image Processing | OpenCV |
| Augmentation | Albumentations |
| Evaluation | TorchMetrics 또는 직접 구현 |

## 7.2 주요 라이브러리

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install numpy pandas matplotlib
pip install scikit-learn
```

## 7.3 모델 생성 예시

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
```

각 설정의 의미는 다음과 같다.

| 설정 | 의미 |
|---|---|
| encoder_name | ResNet34 사용 |
| encoder_weights | ImageNet Pretrained 사용 |
| in_channels | RGB 이미지 입력 |
| classes | 3개 클래스 출력 |

이 모델은 ImageNet으로 사전 학습된 Encoder를 사용하지만, 연기 Segmentation에 대해서는 IJmond 데이터셋으로 별도 학습해야 한다.

## 7.4 Loss Function

모델 학습에서는 예측 결과와 정답 Mask의 차이를 계산하는 Loss Function이 필요하다.

이번 프로젝트에서는 다음 조합을 권장한다.

**Cross Entropy Loss + Dice Loss**

| Loss | 역할 |
|---|---|
| Cross Entropy | 픽셀별 클래스 분류 오류 계산 |
| Dice Loss | 예측 영역과 실제 영역의 겹치는 정도 개선 |

연기 영역은 Background보다 차지하는 픽셀 수가 적을 수 있다.

이러한 클래스 불균형 문제를 고려하여 Dice Loss를 함께 사용하는 방식을 검토한다.

초기에는 두 Loss의 가중치를 동일하게 시작하고 Validation 결과를 보면서 조정한다.

## 7.5 Optimizer

AdamW를 기본 Optimizer로 사용한다.

```python
import torch

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)
```

학습률은 데이터 크기와 모델 성능에 따라 조정한다.

## 7.6 Fine-tuning

Pretrained Encoder를 활용하더라도 IJmond 데이터에 맞게 학습해야 한다.

초기 학습에서는 Encoder의 일부를 고정하고 Decoder 중심으로 학습하는 방법을 사용할 수 있다.

이후 Validation 결과에 따라 Encoder 일부 또는 전체를 Unfreeze하여 Fine-tuning을 수행한다.

권장 실험 순서는 다음과 같다.

1. Pretrained Encoder를 고정하고 Decoder 학습
2. Validation 성능 확인
3. Encoder 일부 Unfreeze
4. 낮은 Learning Rate로 Fine-tuning
5. 성능 비교 후 최종 모델 선택

---

# 8. 모델 성능 평가

## 8.1 Accuracy만으로 평가하면 안 되는 이유

연기 이미지에서 Background 픽셀은 연기 픽셀보다 훨씬 많을 수 있다.

예를 들어 이미지의 95%가 Background이고 5%만 연기라고 가정한다.

모델이 모든 픽셀을 Background로 예측해도 Accuracy는 95%가 된다.

하지만 실제 연기는 하나도 찾지 못했으므로 좋은 모델이라고 할 수 없다.

따라서 Segmentation에서는 IoU, Dice, Precision, Recall 등을 함께 확인해야 한다.

## 8.2 주요 평가 지표

| 지표 | 의미 |
|---|---|
| IoU | 예측 영역과 정답 영역의 교집합 / 합집합 |
| Dice | 예측 영역과 정답 영역의 겹침 정도 |
| Precision | 연기로 예측한 영역 중 실제 연기 비율 |
| Recall | 실제 연기 영역 중 찾아낸 비율 |
| F1-Score | Precision과 Recall의 조화 평균 |

### IoU

IoU는 예측한 연기 영역과 실제 연기 영역이 얼마나 겹치는지를 평가한다.

```text
IoU = Intersection / Union
```

값이 1에 가까울수록 좋은 결과다.

### Dice

Dice는 예측 영역과 실제 영역의 겹치는 정도를 평가한다.

```text
Dice = 2 × Intersection / (Prediction + Ground Truth)
```

IoU와 Dice는 모두 Segmentation 모델의 영역 일치도를 확인하는 대표적인 지표다.

## 8.3 클래스별 평가

전체 성능뿐만 아니라 클래스별 성능도 확인해야 한다.

| 클래스 | 주요 평가 항목 |
|---|---|
| Background | IoU, Precision, Recall |
| Low Opacity Smoke | IoU, Dice, Recall |
| High Opacity Smoke | IoU, Dice, Recall |

특히 High Opacity Smoke는 실제 연기 분석에서 중요한 정보이므로 별도로 성능을 확인한다.

---

# 9. Segmentation 후처리 및 Boundary 추출

## 9.1 Segmentation 결과

U-Net은 각 픽셀을 다음 3개 클래스로 분류한다.

```text
0 = Background
1 = Low Opacity Smoke
2 = High Opacity Smoke
```

예측 결과에서 연기에 해당하는 클래스만 추출하면 전체 연기 영역을 확인할 수 있다.

## 9.2 Smoke Mask 생성

```python
import numpy as np

# prediction: 0=Background, 1=Low, 2=High

smoke_mask = np.where(
    prediction > 0,
    255,
    0
).astype(np.uint8)
```

이렇게 생성한 Mask는 Low와 High Opacity를 모두 포함한 연기 영역이다.

## 9.3 Boundary 추출

Boundary는 OpenCV의 Contour 기능을 활용하여 추출한다.

```python
import cv2

contours, hierarchy = cv2.findContours(
    smoke_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = original_image.copy()

cv2.drawContours(
    result,
    contours,
    -1,
    (0, 255, 0),
    2
)
```

이를 통해 원본 이미지에 연기 영역의 외곽선을 표시할 수 있다.

## 9.4 클래스별 Boundary 추출

전체 연기 영역뿐 아니라 불투명도별 경계를 추출할 수도 있다.

```python
low_mask = np.where(
    prediction == 1,
    255,
    0
).astype(np.uint8)

high_mask = np.where(
    prediction == 2,
    255,
    0
).astype(np.uint8)
```

이렇게 하면 Low Opacity와 High Opacity 영역을 각각 분석할 수 있다.

단, 초기 개발에서는 전체 연기 영역의 Boundary 추출을 우선 구현하고 클래스별 경계는 확장 기능으로 고려한다.

---

# 10. 연기 분석 지표 계산

Segmentation 결과를 활용하면 연기 영역을 정량적으로 분석할 수 있다.

단순히 연기가 존재하는지 여부만 확인하는 것이 아니라, 연기 면적과 불투명도 비율을 계산할 수 있다.

## 10.1 Smoke Ratio

전체 이미지에서 연기 영역이 차지하는 비율이다.

```text
Smoke Ratio
= (Low Opacity Pixels + High Opacity Pixels)
  / Valid Image Pixels × 100
```

예를 들어 전체 유효 픽셀이 100만 개이고 연기 픽셀이 10만 개라면 Smoke Ratio는 10%다.

## 10.2 High Opacity Ratio

전체 이미지에서 High Opacity Smoke가 차지하는 비율이다.

```text
High Opacity Ratio
= High Opacity Pixels
  / Valid Image Pixels × 100
```

이 지표를 통해 이미지 전체에서 고불투명도 연기가 얼마나 차지하는지 확인할 수 있다.

## 10.3 High Opacity Share

전체 연기 영역 중 High Opacity Smoke가 차지하는 비율이다.

```text
High Opacity Share
= High Opacity Pixels
  / Total Smoke Pixels × 100
```

Smoke Ratio와 달리 연기 영역 내부에서 고불투명도 연기의 비중을 확인하는 지표다.

전체 연기 픽셀이 0인 경우에는 0으로 나누는 문제가 발생하므로 NULL 또는 해당 없음 상태로 처리한다.

## 10.4 Boundary 분석 지표

Contour를 활용하여 다음 정보를 계산할 수 있다.

| 지표 | 설명 |
|---|---|
| Contour Area | 연기 영역의 픽셀 면적 |
| Perimeter | 연기 영역 외곽선의 길이 |
| Bounding Box | 연기 영역을 포함하는 사각형 |
| Aspect Ratio | Bounding Box의 가로와 세로 비율 |
| Contour Count | 검출된 연기 영역의 개수 |

주의할 점은 카메라 보정, 촬영 거리 및 원근 보정 정보가 없으면 픽셀 면적을 실제 물리적 연기 면적으로 환산할 수 없다는 것이다.

따라서 초기 프로젝트에서는 픽셀 면적과 이미지 대비 비율로 표현한다.

---

# 11. 사건 발생일자 기반 이미지 시계열 분석

## 11.1 분석 목적

TMS에서 사진과 사건 발생일자를 등록하면 Smoke Segmentation Agent가 해당 이미지의 연기 영역을 분석한다.

여러 장의 이미지가 서로 다른 시간에 등록되면 사건 발생일자 또는 촬영일자 기준으로 분석 결과를 비교할 수 있다.

예를 들어 다음과 같은 분석이 가능하다.

| 사건 발생일자 | Smoke Ratio | High Opacity Ratio |
|---|---:|---:|
| 2026-11-20 09:00 | 5.2% | 1.1% |
| 2026-11-20 10:00 | 8.7% | 2.5% |
| 2026-11-20 11:00 | 12.3% | 4.8% |
| 2026-11-20 12:00 | 7.1% | 2.0% |

이 데이터를 이용하면 시간에 따른 연기 영역과 불투명도 변화 추이를 확인할 수 있다.

## 11.2 분석 가능한 항목

- 시간대별 Smoke Ratio 변화
- 시간대별 High Opacity Ratio 변화
- 연기 영역의 픽셀 면적 변화
- 연기 형태 및 Contour 변화
- 카메라별 분석 결과 비교
- 사건 발생일자별 분석 이력

## 11.3 분석 시 주의사항

IJmond 데이터셋은 연속 동영상이 아니라 시간 정보가 있는 이미지 데이터다.

따라서 이미지의 시간 순서를 비교하여 변화 추이를 분석할 수는 있지만, 이미지 사이에 실제로 연기가 얼마나 지속되었는지 또는 연기가 어떤 경로로 이동했는지를 직접 측정할 수는 없다.

실제 연기 지속시간과 움직임을 분석하려면 연속 영상 또는 충분히 촘촘한 시간 간격의 이미지가 필요하다.

또한 TMS에서 등록하는 사건 발생일자와 이미지 촬영일자는 서로 다른 정보일 수 있으므로 구분하여 저장하는 것이 좋다.

- 사건 발생일자: 사용자가 TMS에 등록하는 사건 기준 시간
- 촬영일자: 이미지가 실제 촬영된 시간
- 등록일자: TMS에 이미지가 등록된 시간
- 분석일자: Agent가 분석을 수행한 시간

---

# 12. Smoke Segmentation 전담 Agent 개발

## 12.1 Agent를 별도로 구성하는 이유

이번 프로젝트에서는 Smoke Segmentation 기능을 별도의 전담 Agent로 구성한다.

TMS는 사진과 사건 발생일자를 등록하고 분석을 요청하는 역할을 담당한다.

Smoke Segmentation Agent는 이미지 분석 업무를 전담한다.

이렇게 역할을 분리하면 향후 다른 AI 분석 기능을 추가하더라도 기존 TMS 업무와 이미지 분석 기능을 독립적으로 확장할 수 있다.

### 역할 구분

| 구성 요소 | 주요 역할 |
|---|---|
| TMS | 사진 등록, 사건 발생일자 등록, 분석 요청 및 결과 조회 |
| Smoke Segmentation Agent | 이미지 분석 작업 관리 및 실행 |
| U-Net Model | 연기 영역 픽셀 단위 분할 |
| Analysis Engine | 연기 면적, 불투명도, 경계 지표 계산 |
| Storage | 원본 이미지, Mask, 분석 결과 저장 |

중요한 점은 Agent가 직접 모든 수치를 판단하거나 계산하는 구조로 만들지 않는 것이다.

**AI 모델은 연기 영역을 분할하고, 정량적인 수치는 Python 분석 코드가 계산하며, Agent는 전체 작업을 관리하고 결과를 전달하는 역할을 담당한다.**

## 12.2 Agent의 주요 기능

| 기능 | 상세 설명 |
|---|---|
| 분석 요청 수신 | TMS에서 전달한 이미지 분석 요청 수신 |
| 입력 데이터 검증 | 이미지 파일, 사건 발생일자, 식별 정보 확인 |
| 이미지 전처리 | 이미지 크기 조정 및 정규화 |
| 모델 추론 | U-Net 모델 실행 |
| Smoke Segmentation | 연기 영역을 3개 클래스로 분류 |
| Boundary Extraction | 연기 영역의 외곽선 추출 |
| Metric Calculation | 연기 면적 및 불투명도 지표 계산 |
| 결과 저장 | Mask와 분석 결과 저장 |
| 결과 전달 | 분석 결과를 TMS에 전달 |
| 실행 이력 관리 | 분석 상태, 처리 시간, 오류 이력 관리 |

## 12.3 Agent 처리 흐름

```text
TMS 사진 및 사건 발생일자 등록
              │
              ▼
     분석 요청 생성 및 전달
              │
              ▼
   Smoke Segmentation Agent
              │
              ▼
       입력 데이터 검증
              │
              ▼
         이미지 전처리
              │
              ▼
          U-Net 추론
              │
              ▼
       Smoke Segmentation
              │
              ▼
        Boundary 추출
              │
              ▼
        분석 지표 계산
              │
              ▼
       결과 및 Mask 저장
              │
              ▼
        TMS에 결과 전달
              │
              ▼
          분석 완료
```

## 12.4 Agent의 Tool 구성

Agent가 업무를 수행할 때 호출하는 기능을 Tool 형태로 분리한다.

| Tool | 기능 |
|---|---|
| validate_input | 이미지 및 사건 정보 검증 |
| preprocess_image | 이미지 전처리 |
| run_unet_inference | U-Net 모델 추론 |
| extract_smoke_mask | 연기 Mask 생성 |
| extract_boundary | 연기 Boundary 추출 |
| calculate_smoke_metrics | 연기 분석 지표 계산 |
| save_analysis_result | 분석 결과 저장 |
| update_tms_result | TMS에 결과 전달 |

이러한 기능을 Tool로 분리하면 Agent가 각 단계를 순서대로 실행하고, 오류가 발생했을 때 어느 단계에서 문제가 발생했는지 확인하기 쉽다.

## 12.5 Agent와 Python 코드의 역할 구분

모든 기능을 LLM이 직접 수행하도록 만들면 안 된다.

이번 프로젝트는 모델 추론과 수치 계산이 중요하므로 각 역할을 명확하게 분리한다.

| 작업 | 담당 |
|---|---|
| 이미지 분석 요청 이해 | Agent |
| 입력 데이터 검증 | Python Tool |
| 이미지 전처리 | Python Tool |
| 연기 영역 분할 | U-Net |
| Boundary 추출 | OpenCV |
| 분석 지표 계산 | Python |
| 분석 결과 저장 | Python Tool |
| 작업 흐름 관리 | Agent |
| 분석 결과 설명 | LLM |

특히 Smoke Ratio, High Opacity Ratio 등의 값은 LLM이 추정하지 않고 Python 코드가 계산한 결과를 사용한다.

LLM은 계산된 결과를 바탕으로 자연어 설명을 생성하는 역할을 담당한다.

## 12.6 Agent 실행 상태 관리

분석 요청이 들어오면 작업 상태를 관리하는 것이 좋다.

| 상태 | 설명 |
|---|---|
| REQUESTED | 분석 요청 접수 |
| VALIDATING | 입력 데이터 검증 |
| PROCESSING | 이미지 전처리 및 모델 추론 |
| ANALYZING | 분석 지표 계산 |
| COMPLETED | 분석 완료 |
| FAILED | 분석 실패 |

실패한 경우에는 실패 단계와 오류 메시지를 저장하여 재처리할 수 있도록 한다.

---

# 13. TMS와 Smoke Segmentation Agent 연계

## 13.1 TMS에서 등록하는 데이터

사용자는 TMS 화면에서 사진과 사건 발생일자를 등록한다.

기본적으로 다음 정보가 필요하다.

| 항목 | 설명 |
|---|---|
| event_id | 사건 식별자 |
| facility_id | 사업장 식별자 |
| image_file | 등록 이미지 |
| event_datetime | 사건 발생일자 및 시간 |
| captured_at | 이미지 촬영일자 및 시간 |
| camera_id | 촬영 카메라 식별자 |
| description | 사건 설명 |
| registered_at | TMS 등록일시 |

초기 개발에서는 event_id, 이미지 파일, 사건 발생일자를 필수 항목으로 구성하고 나머지는 데이터 확보 상황에 따라 적용한다.

사건 발생일자와 촬영일자는 반드시 구분하는 것이 좋다.

## 13.2 TMS에서 Agent로 전달하는 정보

TMS에서 이미지 분석을 요청하면 다음과 같은 정보를 Agent에 전달한다.

```json
{
  "event_id": "EVT-20261120-001",
  "facility_id": "FACILITY-001",
  "event_datetime": "2026-11-20T09:00:00",
  "captured_at": "2026-11-20T08:58:00",
  "camera_id": "CAM-001",
  "image_uri": "storage/events/EVT-20261120-001/image.jpg",
  "description": "산업시설 연기 발생 확인"
}
```

실제 구현에서는 이미지 파일 자체를 전달하거나 Object Storage에 저장한 이미지의 접근 가능한 URI를 전달할 수 있다.

대용량 이미지인 경우에는 Object Storage에 먼저 저장한 후 URI를 전달하는 방식을 권장한다.

## 13.3 Agent 분석 결과 예시

Agent는 분석을 수행한 후 다음과 같은 결과를 생성한다.

```json
{
  "event_id": "EVT-20261120-001",
  "analysis_status": "COMPLETED",
  "model_name": "unet-resnet34",
  "model_version": "v1.0",
  "event_datetime": "2026-11-20T09:00:00",
  "analysis_datetime": "2026-11-20T09:01:20",
  "smoke_detected": true,
  "smoke_ratio": 12.3,
  "high_opacity_ratio": 4.8,
  "high_opacity_share": 39.0,
  "smoke_pixel_area": 125000,
  "contour_count": 2,
  "mask_uri": "storage/analysis/EVT-20261120-001/mask.png",
  "boundary_uri": "storage/analysis/EVT-20261120-001/boundary.png"
}
```

위 수치는 데이터 구조를 설명하기 위한 예시이며 실제 분석 결과가 아니다.

## 13.4 연계 시 고려사항

### 1. 비동기 처리

이미지 분석에는 전처리, 모델 추론, 후처리 시간이 필요하다.

따라서 TMS가 분석이 완료될 때까지 계속 기다리는 동기 방식보다는 비동기 방식이 적합하다.

```text
TMS
 │
 ├── 이미지 등록
 │
 ├── 분석 요청
 │
 └── 요청 접수 응답
          │
          ▼
   Agent 분석 작업
          │
          ▼
      결과 저장
          │
          ▼
   TMS 결과 업데이트
```

TMS는 요청이 접수되었다는 응답을 먼저 받고, Agent는 별도로 분석을 진행한다.

### 2. 중복 요청 방지

동일한 사건과 이미지가 여러 번 전달되더라도 불필요하게 분석을 반복하지 않도록 요청 식별자와 이미지 식별 정보를 관리한다.

필요하면 이미지 해시값을 이용하여 동일 이미지 여부를 확인할 수 있다.

### 3. 실패 및 재처리

분석 도중 오류가 발생하면 실패 상태와 오류 내용을 저장한다.

재처리 시에는 처음부터 모든 작업을 반복할지, 실패한 단계부터 다시 수행할지 정책을 정한다.

### 4. 모델 버전 관리

동일한 이미지라도 모델 버전에 따라 분석 결과가 달라질 수 있다.

따라서 분석 결과에 모델명과 모델 버전을 함께 저장해야 한다.

---

# 14. 데이터 저장 구조

분석 결과를 재사용하고 시간에 따른 변화를 비교하기 위해 데이터 저장 구조를 설계한다.

## 14.1 원본 이미지 정보

| 필드 | 설명 |
|---|---|
| image_id | 이미지 식별자 |
| event_id | 사건 식별자 |
| facility_id | 사업장 식별자 |
| image_uri | 원본 이미지 경로 |
| captured_at | 촬영일시 |
| registered_at | 등록일시 |
| camera_id | 카메라 식별자 |

## 14.2 Segmentation 결과

| 필드 | 설명 |
|---|---|
| analysis_id | 분석 식별자 |
| image_id | 이미지 식별자 |
| model_name | 모델명 |
| model_version | 모델 버전 |
| mask_uri | Segmentation Mask 경로 |
| boundary_uri | Boundary 이미지 경로 |
| analysis_status | 분석 상태 |
| analyzed_at | 분석 완료 시간 |

## 14.3 연기 분석 지표

| 필드 | 설명 |
|---|---|
| smoke_detected | 연기 존재 여부 |
| smoke_pixel_area | 전체 연기 픽셀 수 |
| smoke_ratio | 전체 이미지 대비 연기 비율 |
| low_opacity_pixels | 저불투명도 연기 픽셀 수 |
| high_opacity_pixels | 고불투명도 연기 픽셀 수 |
| high_opacity_ratio | 전체 이미지 대비 고불투명도 연기 비율 |
| high_opacity_share | 전체 연기 중 고불투명도 연기 비율 |
| contour_count | 연기 영역의 외곽선 개수 |
| perimeter | 외곽선 길이 |

분석 결과는 원본 이미지와 연결되어야 하며, 모델 버전이 변경되더라도 기존 결과를 비교할 수 있도록 설계한다.

---

# 15. TMS 데이터와 연기 분석 결과의 연계

## 15.1 연계 목적

TMS에서 관리하는 환경 측정 데이터와 이미지 분석 결과를 함께 조회하면 사건 발생 시점의 정량적인 환경 데이터와 시각적인 연기 정보를 함께 확인할 수 있다.

예를 들어 다음과 같은 정보를 비교할 수 있다.

| 구분 | 분석 항목 |
|---|---|
| TMS | NOx 측정값 |
| TMS | SO2 측정값 |
| Image Analysis | Smoke Ratio |
| Image Analysis | High Opacity Ratio |
| Image Analysis | Contour Area |
| Event | 사건 발생일자 |

## 15.2 주의사항

TMS 측정 데이터와 IJmond 이미지 데이터가 동일한 사업장, 동일한 배출구, 동일한 시간에 수집된 데이터가 아니라면 직접적인 상관관계가 있다고 판단하면 안 된다.

따라서 프로젝트에서는 다음과 같이 구분한다.

- TMS 데이터: 측정 장비에서 수집된 환경 측정값
- IJmond 이미지 데이터: 산업시설 이미지에서 추출한 연기 분석 결과
- 통합 분석: 두 데이터의 분석 항목과 활용 가능성을 비교하는 수준

실제 동일 사업장 및 동일 시간 기준으로 데이터가 연결되는 경우에만 통합 상관분석을 수행한다.

---

# 16. 전체 시스템 아키텍처

이번 프로젝트의 핵심은 TMS와 Smoke Segmentation Agent를 분리하고, Agent가 U-Net 모델 및 분석 Tool을 호출하여 작업을 수행하는 구조다.

```text
┌──────────────────────────────────────────────┐
│                     TMS                      │
│                                              │
│  사진 등록 / 사건 발생일자 등록 / 분석 요청   │
│  분석 결과 조회                               │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│          Smoke Segmentation Agent            │
│                                              │
│  Request Management                          │
│  Input Validation                            │
│  Workflow Orchestration                      │
│  Result Management                           │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │              Agent Tools               │  │
│  │                                        │  │
│  │  Image Preprocessing                   │  │
│  │  U-Net Inference                       │  │
│  │  Smoke Mask Extraction                 │  │
│  │  Boundary Extraction                   │  │
│  │  Metric Calculation                    │  │
│  │  Result Storage                        │  │
│  └────────────────────────────────────────┘  │
│                      │                       │
│                      ▼                       │
│  ┌────────────────────────────────────────┐  │
│  │              U-Net Model               │  │
│  │                                        │  │
│  │  ResNet34 Encoder                      │  │
│  │  U-Net Decoder                         │  │
│  │  3-Class Segmentation                  │  │
│  └────────────────────────────────────────┘  │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│                 Data Storage                 │
│                                              │
│  Original Images                             │
│  Segmentation Masks                          │
│  Boundary Images                             │
│  Analysis Metrics                            │
│  Agent Execution Logs                        │
│  Model Version                               │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│                     TMS                      │
│                                              │
│  분석 결과 수신 및 조회                       │
│  사건별 / 시간별 분석                         │
└──────────────────────────────────────────────┘
```

---

# 17. 개발 단계 및 구현 순서

개발 기간을 고려하여 핵심 기능부터 구현한다.

## 1단계. IJmond 데이터셋 분석

목표: 학습 가능한 이미지와 Mask를 확보한다.

주요 작업:

1. IJmond 데이터셋 다운로드
2. 원본 이미지와 Crop 이미지 구조 확인
3. 이미지와 Mask 파일 매핑
4. Mask 값 및 클래스 확인
5. 카메라별 데이터 분포 확인
6. 촬영 시간 및 메타데이터 확인
7. Train / Validation / Test 분리

완료 기준:

- 원본 이미지와 Mask 매핑 완료
- 클래스별 픽셀 분포 확인
- 데이터 누수 방지 분리 기준 확정

## 2단계. 이미지 전처리 및 데이터 증강

목표: U-Net 학습에 적합한 데이터셋을 준비한다.

주요 작업:

1. 이미지 크기 통일
2. Mask 클래스 변환
3. 이미지 정규화
4. Train 데이터 증강
5. 이미지와 Mask 동시 변환 검증
6. 증강 결과 시각화

완료 기준:

- 이미지와 Mask 크기 일치
- Mask 클래스 값 유지
- Train에만 랜덤 증강 적용
- Validation 및 Test 전처리 기준 확정

## 3단계. U-Net 모델 학습

목표: 연기 영역을 분할하는 AI 모델을 개발한다.

주요 작업:

1. U-Net 모델 구성
2. ResNet34 Pretrained Encoder 적용
3. Cross Entropy + Dice Loss 적용
4. AdamW Optimizer 적용
5. 모델 학습
6. Fine-tuning
7. Validation 성능 평가
8. 모델 저장

완료 기준:

- 3개 클래스 Segmentation 수행
- IoU, Dice, Precision, Recall 산출
- 클래스별 성능 확인
- 학습 모델 저장

## 4단계. Segmentation 및 분석 기능 개발

목표: 모델 예측 결과를 정량적인 분석 데이터로 변환한다.

주요 작업:

1. 이미지 입력 및 전처리
2. U-Net 추론
3. Segmentation Mask 생성
4. Low / High Opacity 영역 분리
5. Boundary 추출
6. Smoke Ratio 계산
7. High Opacity Ratio 계산
8. High Opacity Share 계산
9. 분석 결과 시각화 및 저장

완료 기준:

- 원본 이미지와 Mask 결과 비교 가능
- 연기 영역의 픽셀 수 계산
- 주요 분석 지표 산출
- 분석 결과 저장

## 5단계. Smoke Segmentation Agent 개발

목표: 이미지 분석 작업을 수행하는 전담 Agent를 구현한다.

주요 작업:

1. Agent 입력 데이터 정의
2. 분석 요청 수신 기능 구현
3. 입력 데이터 검증 Tool 개발
4. 이미지 전처리 Tool 개발
5. U-Net 추론 Tool 연결
6. Boundary 추출 Tool 연결
7. 분석 지표 계산 Tool 연결
8. 결과 저장 Tool 연결
9. Agent 실행 상태 관리
10. 실패 및 재처리 기능 구현

완료 기준:

- Agent가 이미지 분석 요청을 수신
- 정의된 Tool을 순서대로 실행
- 분석 결과 저장
- 분석 상태 및 오류 이력 관리

## 6단계. TMS 연계 및 통합 테스트

목표: TMS에서 등록한 이미지가 Agent를 통해 분석되고 결과가 다시 TMS로 전달되는 흐름을 완성한다.

주요 작업:

1. TMS 이미지 등록 데이터 정의
2. 사건 발생일자 전달
3. Agent 분석 요청 API 연계
4. 비동기 분석 처리
5. 분석 결과 전달
6. TMS 결과 조회 연계
7. 중복 요청 및 오류 테스트
8. 전체 E2E 테스트

완료 기준:

- TMS에서 사진 등록 가능
- 사건 발생일자 전달 가능
- Agent 분석 실행 가능
- 분석 결과 TMS 전달 가능
- 분석 실패 및 재처리 확인

---

# 18. 핵심 개발 범위와 확장 개발 범위

개발 기간과 프로젝트 목표를 고려하여 필수 기능과 확장 기능을 구분한다.

## 18.1 핵심 개발 범위

| 영역 | 핵심 개발 기능 |
|---|---|
| Dataset | IJmond 이미지 및 Mask 분석 |
| Preprocessing | 이미지 크기 조정, 정규화, Mask 변환 |
| Augmentation | Train 데이터 증강 |
| Model | U-Net + ResNet34 |
| Training | 3-Class Segmentation 학습 |
| Evaluation | IoU, Dice, Precision, Recall |
| Segmentation | 연기 영역 Mask 생성 |
| Boundary | OpenCV Contour 추출 |
| Analytics | Smoke Ratio 및 불투명도 지표 계산 |
| Agent | Smoke Segmentation 전담 Agent |
| TMS 연계 | 사진 및 사건 발생일자 전달 |
| Result | 분석 결과 저장 및 TMS 전달 |

## 18.2 확장 개발 범위

### 1. YOLO Segmentation 비교

U-Net과 YOLO Segmentation을 비교하여 모델별 성능과 처리 시간을 확인한다.

비교 항목:

- Segmentation IoU
- Dice
- 추론 시간
- 모델 크기
- 클래스별 성능

### 2. Boundary 품질 개선

초기에는 OpenCV Contour를 활용하지만, 더 정밀한 경계 분석이 필요하면 Boundary-aware Loss 등을 검토할 수 있다.

### 3. 카메라별 모델 성능 분석

카메라별로 연기 형태와 촬영 환경이 다를 수 있으므로 카메라별 성능을 비교한다.

### 4. 이미지 시계열 분석

사건 발생일자와 촬영일자를 기준으로 연기 분석 결과를 정렬하고 변화 추이를 확인한다.

### 5. Agent Observability

Agent의 각 단계별 실행 시간과 성공률을 기록한다.

| 항목 | 설명 |
|---|---|
| Request Count | 분석 요청 수 |
| Success Rate | 분석 성공률 |
| Failure Rate | 분석 실패율 |
| Preprocessing Time | 전처리 시간 |
| Inference Time | 모델 추론 시간 |
| Analysis Time | 지표 계산 시간 |
| Total Processing Time | 전체 처리 시간 |
| Model Version | 사용 모델 버전 |

### 6. 자연어 기반 분석 결과 조회

LLM을 활용하여 사용자가 자연어로 분석 결과를 조회할 수 있도록 확장한다.

예시:

- 특정 사건의 연기 비율은 얼마인가?
- 고불투명도 연기가 가장 높았던 시간은 언제인가?
- 특정 기간의 Smoke Ratio 변화 추이를 보여줘.
- 카메라별 연기 분석 결과를 비교해 줘.

이 기능은 저장된 분석 데이터를 조회하고 설명하는 역할로 구성한다.

LLM이 원본 Mask를 임의로 해석하여 수치를 생성하지 않도록 하고, 실제 계산된 분석 결과를 기반으로 응답하도록 한다.

---

# 19. 프로젝트 최종 기술 구성

| 구분 | 기술 | 용도 |
|---|---|---|
| Language | Python | 전체 AI 개발 |
| Deep Learning | PyTorch | 모델 학습 및 추론 |
| Segmentation | U-Net | 연기 영역 분할 |
| Encoder | ResNet34 | 이미지 특징 추출 |
| Pretrained | ImageNet | 사전 학습 가중치 |
| Image Processing | OpenCV | 이미지 처리 및 Boundary |
| Augmentation | Albumentations | 데이터 증강 |
| Data Processing | NumPy / Pandas | 데이터 처리 및 지표 계산 |
| Agent | LangGraph 등 | Agent 실행 흐름 관리 |
| LLM | 프로젝트 환경에 적합한 모델 | 자연어 분석 및 결과 설명 |
| Storage | 프로젝트 환경에 맞게 선택 | 이미지, Mask, 분석 결과 저장 |
| Integration | API | TMS와 Agent 연계 |

특정 대시보드 또는 프론트엔드 기술은 본 가이드에서 지정하지 않는다. 해당 영역은 프로젝트 개발 환경과 멘토의 판단에 따라 선택한다.

---

# 20. 최종 정리

이번 프로젝트에서 가장 중요한 것은 U-Net 모델을 학습하는 것만이 아니다.

TMS에서 등록한 이미지와 사건 발생일자를 Smoke Segmentation 전담 Agent가 전달받아 실제 분석을 수행하고, 그 결과를 정량적인 데이터로 만들어 TMS에 다시 전달하는 전체 흐름을 구현하는 것이다.

핵심 역할은 다음과 같이 구분한다.

| 구성 요소 | 핵심 역할 |
|---|---|
| TMS | 사진 및 사건 발생일자 등록 |
| Smoke Segmentation Agent | 분석 작업 관리 및 실행 |
| U-Net | 연기 영역 픽셀 단위 분할 |
| OpenCV | Boundary 추출 |
| Python Analysis | 연기 면적 및 불투명도 계산 |
| Storage | 이미지와 분석 결과 저장 |
| LLM | 분석 결과 자연어 설명 |

### 최종 개발 목표

1. IJmond 데이터셋을 활용하여 U-Net 모델을 학습한다.
2. 연기 영역을 Background / Low Opacity / High Opacity로 분할한다.
3. Segmentation 결과에서 Boundary를 추출한다.
4. Smoke Ratio 및 High Opacity Ratio 등 주요 지표를 계산한다.
5. Smoke Segmentation 전담 Agent를 개발한다.
6. TMS에서 등록한 사진과 사건 발생일자를 Agent에 전달한다.
7. Agent가 이미지 분석부터 결과 저장까지 수행한다.
8. 분석 결과를 TMS에 전달하고 사건별·시간별로 조회할 수 있도록 한다.

**최종적으로 TMS는 사진과 사건 정보를 등록하고, Smoke Segmentation Agent는 해당 정보를 기반으로 연기 영역을 분석하는 역할을 담당한다.**

이처럼 업무 시스템(TMS)과 AI 분석 Agent를 분리하면 향후 다른 이미지 분석 모델이나 추가 AI Agent를 도입하더라도 기존 시스템을 크게 변경하지 않고 확장할 수 있다.
