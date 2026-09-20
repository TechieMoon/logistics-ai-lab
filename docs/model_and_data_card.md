# 모델 · 데이터 카드

## 소유권과 기여 표시

이 저장소의 코드·가상 데이터·설명 문서는 **Codex를 활용해 제작**했다. 사용자에게 확인되지 않은 직접 구현·연구·팀 협업·대회 실적으로 표현하지 않는다. 사용자의 실제 학습, 코드 수정, 재실험은 향후 변경 기록으로 남길 수 있다.

특정 기업의 공식 프로젝트가 아니며 기업의 로고·내부 정책·고객 데이터는 사용하지 않는다. 개인 연락처·지원서·성적 증빙은 이 저장소의 공개 대상이 아니다.

## 데이터

| 데이터 | 출처 | 사용 범위 | 제한 |
|---|---|---|---|
| 고객 좌표·수요 | seed로 생성한 합성 데이터 | CVRP 학습/검증/평가 | 실제 물류 분포나 도로 거리와 다름 |
| 한국어 정책 | 이 프로젝트를 위해 작성한 가상 문서 | BM25 / Dense / Hybrid 검색 | 실제 CJ 정책으로 사용할 수 없음 |
| 개발·평가 질문 | 이 프로젝트를 위해 작성한 합성 질문 | threshold 선택 / 최종 평가 | 외부 독립 벤치마크가 아니며 표본이 작음 |

같은 정책 집합에서 질문을 만든 폐쇄형 실험이다. 질문 분리와 seed 분리는 과적합 위험을 줄이는 방법이지만 외부 데이터에서의 일반화 증거가 아니다.

## 외부 모델

| 모델 | 고정 revision | 라이선스 | 용도 |
|---|---|---|---|
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | MIT (모델 카드 기준) | 다국어 텍스트 임베딩 |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 (모델 카드 기준) | 검색 근거를 입력받는 로컬 답변 초안 |

모델은 공식 Hugging Face 모델 저장소에서 내려받고 `trust_remote_code=False`, safetensors 형식으로 로드한다. 모델 가중치는 본 저장소 MIT 라이선스의 재라이선스 대상이 아니다. 각 원본 라이선스를 따른다. 파인튜닝을 수행하지 않았으며 사전학습 모델을 추론에 사용한다.

## 직접 학습한 정책

`scripts/train_routing.py`에서 작은 그래프 메시지 전달 정책을 PyTorch로 학습한다. 학습 설정·seed·장치·시간·성능은 `artifacts/routing/metrics.json`에 남긴다. 로컬 `checkpoint.pt`는 재학습 스크립트로 생성하며 공개 저장소에서 제외한다. 다운로드 받은 출처 불명의 PyTorch 체크포인트를 대신 로드하지 않는다.

## 재현 환경

실제 실행 환경은 `artifacts/environment.json`, 설치 버전은 `requirements-lock-windows.txt`에 기록한다. PyTorch CUDA wheel 자체에 런타임이 포함되므로 이 실험을 위해 별도 CUDA Toolkit이나 그래픽 드라이버를 설치하지 않았다. 최초 검증 장치는 RTX 3060 8GB / 드라이버 576.02 / PyTorch 2.8.0+cu128이다.

공식 설치 기준: [PyTorch 이전 버전 설치 안내](https://pytorch.org/get-started/previous-versions/). 다른 장치에서는 GPU·드라이버에 맞는 wheel을 선택해야 하며 CPU 실행도 지원한다.
