# 학생 생활기록부 분석 시스템

이 프로젝트는 고등학생의 생활기록부 데이터를 분석하고 시각화하여 종합적인 학생 프로필을 생성하는 시스템입니다.

## 주요 기능

- CSV 형식의 생활기록부 데이터 업로드 및 분석
- Google Gemini API를 활용한 학생 프로필 분석
- 교과별 성취도, 활동 내역, 진로 적합성 등의 시각화
- 종합 분석 보고서 생성 및 다운로드

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. GitHub 환경변수 설정:
   - GitHub 저장소의 Settings > Secrets and variables > Actions로 이동
   - "New repository secret" 클릭
   - Name: `GEMINI_API_KEY`
   - Value: Google Gemini API 키 입력

## 실행 방법

```bash
streamlit run app.py
```

## CSV 파일 형식

다음 열이 포함되어야 합니다:

### 교과별 세특
- 국어
- 수학
- 영어
- 한국사
- 사회
- 과학
- 과학탐구실험
- 정보
- 체육
- 음악
- 미술

### 활동 내역
- 자율활동
- 동아리활동
- 진로활동
- 행동특성 및 발달사항

## 시각화 기능

1. 교과별 성취도 분석
   - 교과 등급 분석


2. 세특 기반 진로 적합성 분석

