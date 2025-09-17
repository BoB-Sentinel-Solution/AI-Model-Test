# AI-Model-Test
프로젝트 수행 중 Model을 선정하기 위한 엔티티 인식 테스트

## Requirment
```
pip install -U transformers accelerate torch sentencepiece
pip install huggingface_hub[hf_xet]
```

## Test
**1. Qwen2.5-3B-Instruct**

강점: 멀티링구얼(한국어 포함), 128K 컨텍스트, 최신 스피드 벤치 페이지 제공(양자화/FP16별 속도·메모리 지표 있음). 지연 시간·처리량 최적화 자료가 잘 정리되어 있어 실환경 튜닝이 쉬움.

**2. Llama-3.2-Korean-Bllossom-3B**

강점: 3B급 초경량 + 한국어 특화 튜닝(한/영 이중언어). 한국어 분류·엔티티 감지에서 작은 용량 대비 안정적인 정밀도/지연 시간 기대. (Llama 3.2 자체는 저지연 최적화가 많이 되어 있음.)

**3. Mistral-7B-Instruct-v0.3**

강점: 7B급 모델 중 낮은 TTFT와 빠른 출력 속도로 유명. vLLM/llama.cpp 생태계 지원이 매우 탄탄하고, 함수호출 등 최신 기능 호환.

**4. Phi-3.5-Mini-Instruct (≈3.8B)**

강점: 소형 모델 중 지연 시간/비용 대비 성능이 우수. MLPerf Client 벤치 범주에 포함되어 있어(지표 정의 명확) 내부 지연·속도 측정 기준 수립에 유리.

**5. Gemma 2-9B-IT**

강점: 9B급이지만 GQA·로컬/글로벌 어텐션 등 추론 효율 개선 아키텍처로 설계. 경량 양자화(FP8/INT4) 및 vLLM 호환성이 좋아 속도·안정성 균형형 후보.

## Test2
서버 사양 확정에 따른 재분류 및 50개 프롬프트 실험<br>
GPU : NVIDIA GeForce RTX 4070 SUPER<BR>
메모리 : 60
<ul>
  <li>mistralai/Mistral-7B-Instruct-v0.3</li>
  <li>meta-llama/Llama-3.1-8B-Instruct</li>
  <li>Qwen/Qwen2.5-7B-Instruct</li>
</ul>
<img width="1098" height="174" alt="image" src="https://github.com/user-attachments/assets/93ef0011-44b4-4a86-892d-52127df7e164" />
python -c "import torch, torchvision, torchaudio; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torchaudio', torchaudio.__version__); print('CUDA:', getattr(torch.version, 'cuda', None)); print('GPU?', torch.cuda.is_available())"
torch 2.5.1+cu121
torchvision 0.20.1+cu121
torchaudio 2.5.1+cu121
CUDA: 12.1
GPU? True

## Test3
정밀 테스트

지표 산출:

JSON 유효성률(파싱 성공 비율)

Positive rate(has_sensitive=True 비율)

엔티티 개수 평균/타입 분포

TTFT p50/p95(첫 토큰까지 시간, ms)

tok/s p50/p95(토큰 생성 속도)

정상 프롬프트 FPR(정상인데 민감이라고 판단한 비율)

```
# Qwen 7B, 기본 벤치 전부 실행
python evaluator.py --model Qwen/Qwen2.5-7B-Instruct

# Qwen 3B
python evaluator.py --model Qwen/Qwen2.5-3B-Instruct

python evaluator.py --model MLP-KTLim/Llama-3.2-Korean-Bllossom-3B
python evaluator.py --model MLP-KTLim/llama-3-Korean-Bllossom-8B

python evaluator.py --model mistralai/Mistral-7B-Instruct-v0.3

```
