# 재현 수준과 필요한 입력

## 이 공개본에서 바로 재현하는 범위

`requirements-analysis.txt` 설치 후 `pytest`와 `brca1_signed_kernel.py`의 합성 self-check를 실행할 수 있다. 공개 CSV와 JSON은 모델 없이 읽을 수 있고 `render_portfolio_results.py`로 결과 그림을 재생성한다. `docs/SOURCE_FILES.json`의 공개 SHA-256을 검사하면 선택된 원본 파일이 바뀌지 않았는지 확인할 수 있다. 해시는 인증이나 전자서명을 대신하지 않는다.

`requirements.lock.txt`는 당시 GPU 추론 환경의 기록이다. CPU 분석 환경 설치 파일과 용도가 다르다. Linux 전용 `fcntl`을 사용하는 추론·스크리닝 모듈은 Windows에서 직접 실행하지 않는다. Linux CI에서 기존 두 analysis unit test를 실행한다.

## 전체 GPU 실험에 추가로 필요한 자료

| 단계 | 추가 입력·조건 |
|---|---|
| 기본 Evo2 inference | Linux / CUDA / 호환 GPU / evo2_7b 가중치 |
| GRCh38 catalog | ClinVar GRCh38 VCF fileDate=2026-09-05, MANE v1.5 RefSeq GTF, hg38 chromosome FASTA |
| native screen | BRCA1 SNV 목록, splice saturation 목록, REF/ALT 서열, extraction manifest |
| corrected signed cohort | 193개 cohort manifest, forward/RC region mapping, 수치 pilot, 고정 실행 config와 matching fingerprint |
| 기능 probe | 원 연구의 RNA 측정값, 지정 baseline 특징, 193개 수치 audit, 저장된 signed activations와 REF controls |
| context purge | 고정 geometry audit, 원래 OOF 예측, 완료된 prior protocol과 결과 |

원본 스크립트는 `results/brca1_grch38/` 등의 기존 상대 경로와 완료 gate를 유지한다. 공개 집계표는 `docs/results/`에 두었으며 원시 activation을 대신하지 않는다. 필요한 입력이 없으면 full pipeline을 실행할 수 없다. 원본 GPU 실행을 이번 배포 준비에서 다시 수행하지 않았으며, 최초 사용자는 공개 출처의 이용 조건에 따라 입력을 확보하고 cohort와 해시를 다시 검증해야 한다.

## 입력과 평가의 핵심 제약

- BRCA1 MANE transcript: NM_007294.4. 다른 transcript/assembly의 좌표를 그대로 혼합하지 않는다.
- 정방향 변이 index는 16,384, 동일 창의 역상보 index는 16,383 (0-based)다. 역상보를 별도로 재중앙화하지 않는다.
- selected layers: blocks.0, blocks.7, blocks.14, blocks.21, blocks.28, norm.
- pooling: variant ±20, assayed exon, donor ±20, acceptor ±20, whole 32k.
- 같은 genomic position의 ALT는 같은 fold를 유지한다. 중심화·스케일·튜닝은 train fold에서만 계산한다.
- numerical audit, artifact completeness, phenotype prediction quality는 서로 다른 gate다. 하나의 PASS로 나머지까지 통과했다고 선언하지 않는다.

## 코드와 자료의 출처 추적

선택된 연구 소스는 원본 byte를 보존했다. `docs/results/numerical_qc.json`만 지정된 집계 키를 추린 projection이며 manifest에 변환을 명시했다. 일부 결과 CSV의 CRLF도 `.gitattributes`로 보존하여 원래 해시를 유지한다. 새 README·검증 테스트·그림 renderer는 이번 공개본에 추가한 파일이다.

원래 작업 폴더의 캐시·환경·실험 결과와 비공개 설정은 수정하지 않는다. GitHub에는 새 이력을 만들며 기존 로컬 Git·세션 이력은 복사하지 않는다.
