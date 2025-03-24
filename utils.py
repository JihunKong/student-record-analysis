import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any

def preprocess_csv(file):
    """CSV 파일을 전처리하여 DataFrame으로 변환합니다."""
    try:
        # 여러 인코딩 시도
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("파일을 읽을 수 없습니다. 인코딩 문제가 있을 수 있습니다.")
        
        # 빈 행과 열 제거
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # 첫 번째 유효한 행을 찾아 컬럼명으로 설정
        first_valid_row = df.iloc[0]
        if not first_valid_row.isna().all():
            df.columns = first_valid_row
            df = df.iloc[1:]
        
        # 인덱스 재설정
        df = df.reset_index(drop=True)
        
        # 성적 데이터 섹션 찾기
        try:
            grade_section = df[df.iloc[:, 0] == '학 기'].index[0]
            grade_data = df.iloc[grade_section:].copy()
            
            # 성적 데이터 컬럼명 설정
            grade_columns = ['학 기', '교 과', '과 목', '학점수', '원점수/과목평균\n (표준편차)', '성취도\n (수강자수)', '석차등급']
            grade_data = grade_data.iloc[:, :len(grade_columns)]
            grade_data.columns = grade_columns
            
            # 빈 행 제거
            grade_data = grade_data.dropna(subset=['학 기', '교 과', '과 목'], how='all')
            
            # 성적 데이터를 제외한 나머지 데이터
            main_data = df.iloc[:grade_section].copy()
            
            return main_data, grade_data
            
        except IndexError:
            # 성적 데이터 섹션을 찾지 못한 경우
            return df, pd.DataFrame()
            
    except Exception as e:
        raise Exception(f"CSV 파일 전처리 중 오류 발생: {str(e)}")

def convert_to_python_type(obj):
    """NumPy 타입을 Python 기본 타입으로 변환합니다."""
    if obj is None:
        return None
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if pd.isna(obj):
        return None
    if isinstance(obj, dict):
        return {key: convert_to_python_type(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_python_type(item) for item in obj]
    return obj

def extract_student_info(df_tuple):
    """DataFrame에서 학생 정보를 추출합니다."""
    try:
        main_data, grade_data = df_tuple
        
        student_info = {}
        
        # 교과별 세부능력 및 특기사항
        academic_performance = {}
        subjects = ['국어', '수학', '영어', '한국사', '사회', '과학', '과학탐구실험', '정보', '체육', '음악', '미술']
        
        for subject in subjects:
            if subject in main_data.columns:
                content = main_data[main_data.columns[list(main_data.columns).index(subject)]].iloc[0]
                if pd.notna(content):
                    # 문자열로 변환하여 저장
                    academic_performance[subject] = str(content) if not isinstance(content, str) else content
        
        # 활동 내역
        activities = {}
        activity_types = ['자율', '동아리', '진로', '행특', '개인']
        
        for activity_type in activity_types:
            if activity_type in main_data.columns:
                content = main_data[main_data.columns[list(main_data.columns).index(activity_type)]].iloc[0]
                if pd.notna(content):
                    # 문자열로 변환하여 저장
                    activities[activity_type] = str(content) if not isinstance(content, str) else content
        
        # 진로 희망
        career_aspiration = ""
        if '진로희망' in main_data.columns:
            career_aspiration = main_data[main_data.columns[list(main_data.columns).index('진로희망')]].iloc[0]
            career_aspiration = str(career_aspiration) if pd.notna(career_aspiration) else ""
        
        # 학기별 성적
        grades = []
        if not grade_data.empty:
            # 컬럼명에서 개행 문자 제거
            grade_data.columns = [col.replace('\n', ' ').strip() for col in grade_data.columns]
            
            # 성적 데이터 정리
            for _, row in grade_data.iterrows():
                if pd.notna(row['학 기']):  # 유효한 행만 처리
                    grade_entry = {
                        '학기': str(row['학 기']),
                        '교과': str(row['교 과']),
                        '과목': str(row['과 목']),
                        '학점수': str(row['학점수']) if pd.notna(row['학점수']) else "",
                        '원점수/과목평균': str(row['원점수/과목평균 (표준편차)']) if pd.notna(row['원점수/과목평균 (표준편차)']) else "",
                        '성취도': str(row['성취도 (수강자수)']) if pd.notna(row['성취도 (수강자수)']) else "",
                        '석차등급': str(row['석차등급']) if '석차등급' in row and pd.notna(row['석차등급']) else ""
                    }
                    grades.append(grade_entry)
        
        # 추출한 정보를 student_info에 저장
        student_info['academic_performance'] = academic_performance
        student_info['activities'] = activities
        student_info['career_aspiration'] = career_aspiration
        student_info['grades'] = grades
        
        return student_info
        
    except Exception as e:
        print(f"학생 정보 추출 중 오류 발생: {str(e)}")  # 디버깅을 위한 출력 추가
        raise Exception(f"학생 정보 추출 중 오류 발생: {str(e)}")

def create_downloadable_report(content: Dict[str, Any], filename: str = "분석_보고서.md") -> str:
    """다운로드 가능한 보고서를 생성합니다."""
    report = f"""# 학생 생활기록부 분석 보고서

## 1. 학생 프로필
{content['학생_프로필']['기본_정보']}

### 강점
{chr(10).join(f"- {strength}" for strength in content['학생_프로필']['강점'])}

### 학업 패턴
{content['학생_프로필']['학업_패턴']}

## 2. 진로 적합성 분석
{content['진로_적합성']['분석_결과']}

### 추천 진로
{chr(10).join(f"- {option}" for option in content['진로_적합성']['추천_진로'])}

### 진로 로드맵
{content['진로_적합성']['진로_로드맵']}

## 3. 학업 발전 전략
{content['학업_발전_전략']['분석_결과']}

### 개선 전략
{chr(10).join(f"- {strategy}" for strategy in content['학업_발전_전략']['개선_전략'])}

## 4. 학부모 상담 가이드
{content['학부모_상담_가이드']['분석_결과']}

### 상담 포인트
{chr(10).join(f"- {point}" for point in content['학부모_상담_가이드']['상담_포인트'])}

### 지원 방안
{chr(10).join(f"- {support}" for support in content['학부모_상담_가이드']['지원_방안'])}

## 5. 진로 로드맵
### 단기 목표
{chr(10).join(f"- {goal}" for goal in content['진로_로드맵']['단기_목표'])}

### 중기 목표
{chr(10).join(f"- {goal}" for goal in content['진로_로드맵']['중기_목표'])}

### 장기 목표
{chr(10).join(f"- {goal}" for goal in content['진로_로드맵']['장기_목표'])}
"""
    return report

def create_subject_comparison_chart(subject_data: Dict[str, Any]) -> go.Figure:
    """교과별 성취도를 비교하는 차트를 생성합니다."""
    subjects = list(subject_data.keys())
    scores = [float(subject_data[subject]['성취도']) for subject in subjects]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=subjects,
        y=scores,
        text=scores,
        textposition='auto',
        name='성취도'
    ))
    
    fig.update_layout(
        title="교과별 성취도 비교",
        xaxis_title="교과",
        yaxis_title="성취도",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_activity_heatmap(activities: List[Dict[str, Any]]) -> go.Figure:
    """활동 내역을 히트맵으로 시각화합니다."""
    # 활동 유형별 빈도 계산
    activity_types = [activity['활동명'] for activity in activities]
    unique_types = list(set(activity_types))
    type_counts = [activity_types.count(type_) for type_ in unique_types]
    
    fig = go.Figure(data=go.Heatmap(
        z=[type_counts],
        x=unique_types,
        y=['활동 빈도'],
        text=[[f"{count}회" for count in type_counts]],
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title="활동 유형별 참여 빈도",
        xaxis_title="활동 유형",
        yaxis_title=""
    )
    
    return fig

def create_career_radar_chart(career_data: Dict[str, Any]) -> go.Figure:
    """진로 적합성을 레이더 차트로 시각화합니다."""
    categories = list(career_data.keys())
    values = list(career_data.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='적합도'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="진로 적합성 분석"
    )
    
    return fig

def plot_timeline(events: List[Dict[str, Any]]) -> plt.Figure:
    """시간순 이벤트를 타임라인 차트로 시각화합니다."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_positions = range(len(events))
    labels = [event['title'] for event in events]
    
    ax.scatter([pd.to_datetime(event['date']) for event in events], y_positions, s=80, color='skyblue')
    
    for i, event in enumerate(events):
        ax.annotate(event['title'], 
                   (pd.to_datetime(event['date']), i),
                   xytext=(10, 0), 
                   textcoords='offset points',
                   va='center')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([''] * len(y_positions))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.title('학생 발전 타임라인')
    plt.tight_layout()
    
    return fig

def create_radar_chart(categories: Dict[str, float]) -> plt.Figure:
    """능력치 레이더 차트를 생성합니다."""
    categories_list = list(categories.keys())
    values = list(categories.values())
    
    # 레이더 차트 생성
    angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
    values += values[:1]  # 첫번째 값을 마지막에 추가하여 폐곡선 만들기
    angles += angles[:1]  # 첫번째 각도를 마지막에 추가
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_list)
    
    plt.title('학생 능력 프로필', size=15, pad=20)
    
    return fig

def process_csv_file(file):
    """CSV 파일을 처리하여 문자열로 반환합니다."""
    try:
        # 파일 내용을 문자열로 읽기
        content = file.read().decode('utf-8')
        
        # 파일 내용을 Gemini에게 전달하기 위한 형식으로 정리
        formatted_content = (
            "아래는 학생 생활기록부 데이터입니다. 이 데이터를 분석해주세요:\n\n"
            f"{content}\n"
        )
        
        return formatted_content
        
    except UnicodeDecodeError:
        try:
            # UTF-8로 읽기 실패시 CP949로 시도
            file.seek(0)  # 파일 포인터를 처음으로 되돌림
            content = file.read().decode('cp949')
            formatted_content = (
                "아래는 학생 생활기록부 데이터입니다. 이 데이터를 분석해주세요:\n\n"
                f"{content}\n"
            )
            return formatted_content
        except Exception as e:
            raise Exception(f"파일 인코딩 처리 중 오류 발생: {str(e)}")
    except Exception as e:
        raise Exception(f"CSV 파일 처리 중 오류 발생: {str(e)}")

def create_analysis_prompt(csv_content: str) -> str:
    """분석을 위한 프롬프트를 생성합니다."""
    prompt = f"""
다음은 학생 생활기록부 데이터입니다. 이 데이터를 분석하여 다음 형식의 JSON으로 응답해주세요.
또한 성적 데이터를 시각화하기 위한 React 컴포넌트 코드도 함께 제공해주세요.

{csv_content}

응답은 다음 세 부분으로 구성해주세요:

1. React 컴포넌트 코드 (```jsx 또는 ```react로 시작):
```jsx
// 성적 데이터 시각화를 위한 React 컴포넌트
// Chart.js 또는 Recharts 사용
```

2. CSS 스타일 코드 (```css로 시작):
```css
/* 시각화 컴포넌트를 위한 스타일 */
```

3. JSON 형식의 분석 결과 (```json으로 시작):
```json
{{
    "학생_프로필": {{
        "기본_정보_요약": "학년, 반, 번호, 이름 정보를 포함한 한 줄 요약",
        "진로희망": "학생의 진로 희망 분야",
        "강점": ["주요 강점 1", "주요 강점 2", ...],
        "약점": ["개선이 필요한 부분 1", "개선이 필요한 부분 2", ...]
    }},
    "교과_성취도": {{
        "과목별_분석": {{
            "과목명": "성취도 및 특징 분석",
            ...
        }},
        "성적_데이터": {{
            "과목명": {{
                "석차등급": "1~9 사이의 값",
                "원점수": "0~100 사이의 값",
                "과목평균": "0~100 사이의 값",
                "표준편차": "값",
                "성취도": "A~E 사이의 값",
                "수강자수": "정수값"
            }},
            ...
        }}
    }},
    "활동_내역": {{
        "자율활동": "활동 내용 분석",
        "동아리활동": "활동 내용 분석",
        "진로활동": "활동 내용 분석",
        "봉사활동": "활동 내용 분석"
    }},
    "진로_적합성": {{
        "일치도": "현재 진로희망과 역량의 일치도 분석",
        "적합_진로_옵션": ["추천 진로 1", "추천 진로 2", ...]
    }},
    "학업_발전_전략": {{
        "교과목_분석": {{
            "과목명": "구체적인 학습 전략",
            ...
        }},
        "권장_전략": ["학습 전략 1", "학습 전략 2", ...]
    }},
    "진로_로드맵": {{
        "단기_목표": ["목표 1", "목표 2", ...],
        "중기_목표": ["목표 1", "목표 2", ...],
        "장기_목표": ["목표 1", "목표 2", ...],
        "추천_활동": {{
            "교과_활동": ["추천 활동 1", "추천 활동 2", ...],
            "비교과_활동": ["추천 활동 1", "추천 활동 2", ...]
        }}
    }}
}}
```

주의사항:
1. React 컴포넌트는 반드시 ```jsx 또는 ```react로 시작하는 코드 블록으로 제공해주세요.
2. CSS 스타일은 반드시 ```css로 시작하는 코드 블록으로 제공해주세요.
3. JSON 분석 결과는 반드시 ```json으로 시작하는 코드 블록으로 제공해주세요.
4. React 컴포넌트는 다음 차트들을 포함해야 합니다:
   - 과목별 석차등급 비교 차트
   - 원점수/과목평균 비교 차트
   - 성취도 분포 차트
   - 학기별 성적 추이 차트
5. 모든 분석은 객관적인 데이터를 기반으로 해주세요.
6. 구체적이고 실행 가능한 제안을 해주세요.
7. 학생의 강점을 최대한 살리는 방향으로 분석해주세요.
8. 모든 값은 문자열 형태로 반환해주세요.
"""
    return prompt 