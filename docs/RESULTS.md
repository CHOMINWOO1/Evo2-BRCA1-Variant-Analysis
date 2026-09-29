# 결과와 해석

![기록된 결과](figures/research-results.png)

## 집합과 계산 방식의 구분

원래 13,455 SNV 스크리닝은 native 추론 기록이다. 이후 동일한 prefix에서 나타난 REF–ALT 표현 차이 중 수치적 성분이 확인되었다. 전체 native 집합을 교정 완료로 표시하지 않는다. [native_screen_status.json](results/native_screen_status.json)은 분석 완료 수를, [native_group_summary.csv](results/native_group_summary.csv)는 당시 기술 통계를 보존한다.

별도 SeqSplice 집합은 193 SNV, 136개 고유 좌표, 12개 실험 엑손으로 구성되었다. 193개 × 2방향의 수치 검증된 signed 표현을 추출했고 6개 선택 레이어에서 2,316개 prefix 검사가 통과했다. [numerical_qc.json](results/numerical_qc.json)은 원본 audit의 공개 집계 필드만 보존한 projection이다. 전체 audit의 경로·개별 artifact는 포함하지 않았다.

원래 결과와 정확히 겹친 183개 정방향 사례의 prefix 평균 상대 L2 평균은 native 0.0036756860962534536에서 검증된 계산 0.0으로 바뀌었다. 이는 선택된 조건의 동일 prefix 불변성을 확인한 결과다. 모델 출력 전체의 오차가 제거되거나 13,455개 전체가 교정되었다는 뜻은 아니다. 후방 위치의 변이 반응도 함께 평가해야 한다.

## Signed 표현의 주 기능 비교

사전에 새 signed-vector/outcome 결합 이전에 고정한 탐색 프로토콜을 사용했다. 기존 scalar 결과는 이미 알려진 상태였고 외부 사전등록은 아니다. [프로토콜](results/signed_protocol.json)에 입력 hash, layer/region, baseline, split, lambda, bootstrap 정의를 보존했다.

MDA-MB-231의 norm / assayed exon, n=193에서 matched scalar baseline의 macro-exon MAE는 13.237234042391426 pp, scalar + signed는 13.877791324357814 pp였다. 주 효과는 combined minus baseline이므로 +0.6405572819663893 pp는 개선이 아니다. 95% 구간은 [0.14342398031848713, 1.1032661643511483]이다.

HS578T는 n=191이고 같은 변이를 공유한다. 별도 독립 변이 검증 집합이 아니다. [모든 metric](results/signed_metrics.csv)과 [paired comparisons](results/signed_paired_comparisons.csv)를 함께 공개하여 주 결과만 선택적으로 제시하지 않는다. 구간은 12개 엑손의 고정 OOF 예측에 조건부이며, 5,000회의 공통 bootstrap draw를 사용했다. 여러 layer/region의 탐색적 비교를 다중검정 보정된 발견으로 해석하지 않는다.

## 겹치는 문맥과 Evo2-only 대조

32k 문맥은 이웃 변이 사이에서 크게 겹칠 수 있다. context-purged outer LOEO는 test와 겹치는 train window를 제거한다. [원래 특징 구성의 결과](results/context_purged_metrics.csv), [point comparisons](results/context_purged_comparisons.csv), [검증 기록](results/context_purged_validation.json)을 공개했다. outer overlap은 0이지만 inner LOEO는 일반 방식이며, 모든 것이 단일 유전자 안에서 수행된 기술적 stress test다.

[Evo2-only 일반 LOEO](results/evo2_only_metrics.csv)와 [Evo2-only context purge](results/evo2_only_context_purged_metrics.csv)는 사후 민감도 분석이다. SpliceAI를 새 특징에 넣지 않는 scalar·sequence·signed·REF 대조를 구분한다. purged point estimate에 bootstrap 구간이나 p-value를 새로 만들지 않았다. 원래 exploratory 주 결과와 다른 분석을 섞어 성능 향상을 주장하지 않는다.

## 이번 공개 준비에서 한 일

원래 수치·CSV·protocol을 유지하고 설명과 시각화를 새로 정리했다. 과장된 성능·신규성 주장을 추가하지 않았다. 새로 계산한 것은 기존 집계값을 표시하는 그림과 공개 파일·수학 구현 검증이며, 대규모 GPU 추론이나 새로운 생물학적 실험을 수행한 결과는 아니다.
