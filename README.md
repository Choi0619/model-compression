# LoRA Rank에 따른 학습 성능 비교 🔍

## 프로젝트 설명 🎯

이번 프로젝트는 **LoRA (Low-Rank Adaptation)**에서 rank를 조정할 때 모델 성능과 메모리 사용량에 미치는 영향을 분석하는 것이 목표입니다. Rank는 LoRA의 핵심 파라미터 중 하나로, 모델의 성능과 자원 사용에 영향을 줍니다. 실험에서는 rank를 `[8, 128, 256]`으로 변경하여, 학습 손실(Loss), 학습 속도(Runtime), 그리고 메모리 사용량을 관찰했습니다.

## 실험 설정 ⚙️

- **모델**: `facebook/opt-350m`
- **데이터셋**: `lucasmccabe-lmi/CodeAlpaca-20k`
- **Rank 값**: `[8, 128, 256]`
- **평가 항목**:
  - 학습 손실 (Loss)
  - 평균 CPU 및 GPU 메모리 사용량
  - 학습 소요 시간 (Runtime)

## 결과 분석 📊

### 학습 손실 (Loss)
- [Loss 그래프 보기](https://api.wandb.ai/links/wrtyu0603-illinois-institute-of-technology/630d7aty)
- Rank가 높을수록 미세하게 낮은 Loss를 보여, **Rank 256과 128**이 **Rank 8**에 비해 약간 더 안정적인 학습을 제공했습니다.

### 평균 CPU 메모리 사용량
- [CPU 메모리 사용량 그래프 보기](https://api.wandb.ai/links/wrtyu0603-illinois-institute-of-technology/1ocbxayx)
- Rank 8이 가장 적은 CPU 메모리를 사용하였고, 128과 256의 메모리 사용량은 비슷하게 나왔습니다.

### 평균 GPU 메모리 사용량
- [GPU 메모리 사용량 그래프 보기](https://api.wandb.ai/links/wrtyu0603-illinois-institute-of-technology/wyhl2pom)
- GPU 메모리 사용량은 Rank가 높아질수록 증가했으며, Rank 8이 가장 적은 메모리를 소비했습니다.

### 학습 소요 시간 (Runtime)
- [Runtime 그래프 보기](https://api.wandb.ai/links/wrtyu0603-illinois-institute-of-technology/m2u59eri)
- Rank가 높을수록 학습 시간이 길어졌으며, Rank 8이 가장 빠른 학습 속도를 보였습니다.

## 결론 및 요약 ✍️

- **LoRA Rank**가 높을수록 GPU 메모리와 학습 시간이 증가하지만, 학습 손실이 약간 더 낮아졌습니다.
- **Rank 8**: 적은 자원 소모와 빠른 학습 속도를 보여, **저사양 환경에서 효율적**으로 보입니다.
- **Rank 256**: 메모리와 시간이 더 소요되지만, **학습 성능**이 미미하게 좋아졌습니다.
