import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple

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

def extract_student_info(df):
    """DataFrame에서 학생 정보를 추출합니다."""
    try:
        student_info = {}
        
        # 교과별 세부능력 및 특기사항
        academic_performance = {}
        subjects = ['국어', '수학', '영어', '한국사', '사회', '과학', '과학탐구실험', '정보', '체육', '음악', '미술']
        
        # 세특이 있는 행 찾기
        setech_rows = df[df.iloc[:, 0].str.contains('세부능력 및 특기사항', na=False)].index
        if len(setech_rows) > 0:
            setech_start = setech_rows[0]
            setech_end = df[df.iloc[:, 0].str.contains('^학.?기$', na=False, regex=True)].index[0] if len(df[df.iloc[:, 0].str.contains('^학.?기$', na=False, regex=True)].index) > 0 else len(df)
            
            # 세특 데이터 처리
            setech_data = df.iloc[setech_start+1:setech_end].copy()
            
            # 첫 번째 행이 과목명을 포함하고 있는지 확인
            if any(subject in str(cell) for cell in setech_data.iloc[0] for subject in subjects):
                setech_data.columns = setech_data.iloc[0]
                setech_data = setech_data.iloc[1:]
            
            # 각 과목별 세특 내용 추출
            for subject in subjects:
                matching_columns = [col for col in setech_data.columns if str(subject) in str(col)]
                for col in matching_columns:
                    content = setech_data[col].iloc[0] if len(setech_data) > 0 else None
                    if pd.notna(content):
                        academic_performance[subject] = str(content)
        
        # 활동 내역
        activities = {}
        activity_types = ['자율', '동아리', '진로', '행특', '개인']
        
        for activity_type in activity_types:
            matching_columns = [col for col in df.columns if str(activity_type) in str(col)]
            for col in matching_columns:
                content = df[col].iloc[0] if len(df) > 0 else None
                if pd.notna(content):
                    activities[activity_type] = str(content)
        
        # 진로 희망
        career_aspiration = ""
        career_columns = [col for col in df.columns if '진로' in str(col) and '희망' in str(col)]
        if career_columns:
            career_aspiration = df[career_columns[0]].iloc[0]
            career_aspiration = str(career_aspiration) if pd.notna(career_aspiration) else ""
        
        # 성적 데이터 섹션 찾기
        grades = []
        main_subjects = ['국어', '수학', '영어', '사회', '과학', '한국사', '정보']  # 주요 과목 리스트
        grade_section = df[df.iloc[:, 0].str.contains('^학.?기$', na=False, regex=True)].index
        if len(grade_section) > 0:
            grade_start = grade_section[0]
            grade_data = df.iloc[grade_start:].copy()
            
            # 과목별 학점수 저장
            subject_credits = {}
            
            for _, row in grade_data.iterrows():
                try:
                    if pd.notna(row.iloc[0]) and not str(row.iloc[0]).startswith('학'):  # 학기 정보가 있고 헤더가 아닌 행만 처리
                        semester = str(row.iloc[0])
                        subject = str(row.iloc[2]) if len(row) > 2 else ""
                        grade = str(row.iloc[6]) if len(row) > 6 and pd.notna(row.iloc[6]) else "0"
                        
                        # 학점수가 숫자인 경우에만 처리
                        try:
                            credit = float(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else 1.0
                        except (ValueError, TypeError):
                            credit = 1.0  # 학점수가 숫자가 아닌 경우 기본값 1.0 사용
                        
                        if grade != "0" and grade.replace('.', '').isdigit():  # 유효한 등급인 경우만 처리
                            subject_credits[subject] = credit
                            is_main = any(main_subj in subject for main_subj in main_subjects)
                            grades.append({
                                'semester': semester,
                                'subject': subject,
                                'grade': grade,
                                'credit': credit,
                                'is_main': is_main
                            })
                except Exception as e:
                    print(f"행 처리 중 오류 발생: {str(e)}, 행 데이터: {row}")
                    continue
        
        # 전체 과목 평균 계산
        first_semester = [float(g['grade']) for g in grades if g['semester'] == '1' and g['grade'] != '0']
        second_semester = [float(g['grade']) for g in grades if g['semester'] == '2' and g['grade'] != '0']
        
        # 주요 과목 평균 계산
        first_semester_main = [float(g['grade']) for g in grades if g['semester'] == '1' and g['grade'] != '0' and g['is_main']]
        second_semester_main = [float(g['grade']) for g in grades if g['semester'] == '2' and g['grade'] != '0' and g['is_main']]
        
        # 전체 과목 단순 평균 계산
        first_semester_avg = sum(first_semester) / len(first_semester) if first_semester else 0
        second_semester_avg = sum(second_semester) / len(second_semester) if second_semester else 0
        total_avg = (first_semester_avg + second_semester_avg) / 2 if first_semester and second_semester else 0
        
        # 주요 과목 단순 평균 계산
        first_semester_main_avg = sum(first_semester_main) / len(first_semester_main) if first_semester_main else 0
        second_semester_main_avg = sum(second_semester_main) / len(second_semester_main) if second_semester_main else 0
        total_main_avg = (first_semester_main_avg + second_semester_main_avg) / 2 if first_semester_main and second_semester_main else 0
        
        # 전체 과목 가중 평균 계산
        first_semester_weighted = [(float(g['grade']) * float(g['credit'])) for g in grades if g['semester'] == '1' and g['grade'] != '0']
        first_semester_credits = [float(g['credit']) for g in grades if g['semester'] == '1' and g['grade'] != '0']
        second_semester_weighted = [(float(g['grade']) * float(g['credit'])) for g in grades if g['semester'] == '2' and g['grade'] != '0']
        second_semester_credits = [float(g['credit']) for g in grades if g['semester'] == '2' and g['grade'] != '0']
        
        # 주요 과목 가중 평균 계산
        first_semester_main_weighted = [(float(g['grade']) * float(g['credit'])) for g in grades if g['semester'] == '1' and g['grade'] != '0' and g['is_main']]
        first_semester_main_credits = [float(g['credit']) for g in grades if g['semester'] == '1' and g['grade'] != '0' and g['is_main']]
        second_semester_main_weighted = [(float(g['grade']) * float(g['credit'])) for g in grades if g['semester'] == '2' and g['grade'] != '0' and g['is_main']]
        second_semester_main_credits = [float(g['credit']) for g in grades if g['semester'] == '2' and g['grade'] != '0' and g['is_main']]
        
        first_semester_weighted_avg = sum(first_semester_weighted) / sum(first_semester_credits) if first_semester_credits else 0
        second_semester_weighted_avg = sum(second_semester_weighted) / sum(second_semester_credits) if second_semester_credits else 0
        total_weighted_avg = (first_semester_weighted_avg + second_semester_weighted_avg) / 2 if first_semester_credits and second_semester_credits else 0
        
        first_semester_main_weighted_avg = sum(first_semester_main_weighted) / sum(first_semester_main_credits) if first_semester_main_credits else 0
        second_semester_main_weighted_avg = sum(second_semester_main_weighted) / sum(second_semester_main_credits) if second_semester_main_credits else 0
        total_main_weighted_avg = (first_semester_main_weighted_avg + second_semester_main_weighted_avg) / 2 if first_semester_main_credits and second_semester_main_credits else 0
        
        # 결과 저장
        student_info.update({
            'academic_performance': academic_performance,
            'activities': activities,
            'career_aspiration': career_aspiration,
            'grades': grades,
            
            # 전체 과목 평균
            'first_semester_average': first_semester_avg,
            'second_semester_average': second_semester_avg,
            'total_average': total_avg,
            'first_semester_weighted_average': first_semester_weighted_avg,
            'second_semester_weighted_average': second_semester_weighted_avg,
            'total_weighted_average': total_weighted_avg,
            
            # 주요 과목 평균
            'first_semester_main_average': first_semester_main_avg,
            'second_semester_main_average': second_semester_main_avg,
            'total_main_average': total_main_avg,
            'first_semester_main_weighted_average': first_semester_main_weighted_avg,
            'second_semester_main_weighted_average': second_semester_main_weighted_avg,
            'total_main_weighted_average': total_main_weighted_avg
        })
        
        return student_info
        
    except Exception as e:
        print(f"학생 정보 추출 중 오류 발생: {str(e)}")  # 디버깅을 위한 출력 추가
        raise Exception(f"학생 정보 추출 중 오류 발생: {str(e)}")

def create_downloadable_report(content: Dict[str, Any], original_data: str, filename: str = "분석_보고서.md") -> str:
    """다운로드 가능한 보고서를 생성합니다."""
    report = f"""# 학생 생활기록부 분석 보고서

<details>
<summary>원본 데이터 보기</summary>

```csv
{original_data}
```

</details>

## 1. 성적 분석
### 학기별 과목 등급 비교
{content['성적_분석'].get('1학기', {}).get('과목별_등급', {})}
{content['성적_분석'].get('2학기', {}).get('과목별_등급', {})}

### 평균 등급 분석
- 1학기 가중평균: {content['성적_분석'].get('1학기', {}).get('가중_평균', '-')}
- 1학기 단순평균: {content['성적_분석'].get('1학기', {}).get('단순_평균', '-')}
- 2학기 가중평균: {content['성적_분석'].get('2학기', {}).get('가중_평균', '-')}
- 2학기 단순평균: {content['성적_분석'].get('2학기', {}).get('단순_평균', '-')}

### 전체 평균
- 주요과목(국영수사과) 평균: {content['성적_분석'].get('전체', {}).get('주요과목_평균', '-')}
- 전체과목 평균: {content['성적_분석'].get('전체', {}).get('전체과목_평균', '-')}

## 2. 학생 프로필
{content['학생_프로필']['기본_정보']}

### 강점
{chr(10).join(f"- {strength}" for strength in content['학생_프로필']['강점'])}

### 학업 패턴
{content['학생_프로필']['학업_패턴']}

## 3. 진로 적합성 분석
{content['진로_적합성']['분석_결과']}

### 추천 진로
{chr(10).join(f"- {option}" for option in content['진로_적합성']['추천_진로'])}

### 진로 로드맵
{content['진로_적합성']['진로_로드맵']}

## 4. 학업 발전 전략
{content['학업_발전_전략']['분석_결과']}

### 개선 전략
{chr(10).join(f"- {strategy}" for strategy in content['학업_발전_전략']['개선_전략'])}

## 5. 학부모 상담 가이드
{content['학부모_상담_가이드']['분석_결과']}

### 상담 포인트
{chr(10).join(f"- {point}" for point in content['학부모_상담_가이드']['상담_포인트'])}

### 지원 방안
{chr(10).join(f"- {support}" for support in content['학부모_상담_가이드']['지원_방안'])}

## 6. 진로 로드맵
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

def process_csv_file(file) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """CSV 파일을 처리하여 세특 데이터와 성적 데이터를 분리합니다."""
    try:
        # 전체 파일 읽기 (헤더 없이)
        df = pd.read_csv(file, header=None, encoding='utf-8')
        
        # 빈 행과 열 제거
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # 성적 데이터 시작점 찾기 ('학 기' 또는 '학기'가 있는 행)
        grade_start_idx = None
        for idx, row in df.iterrows():
            if any(str(cell).strip() in ['학 기', '학기'] for cell in row):
                grade_start_idx = idx
                break
                
        if grade_start_idx is None:
            raise Exception("성적 데이터를 찾을 수 없습니다.")
        
        # 세특 데이터와 성적 데이터 분리
        special_notes = df.iloc[:grade_start_idx].copy()
        grades = df.iloc[grade_start_idx:].copy()
        
        # 성적 데이터의 첫 번째 행을 헤더로 설정
        grades.columns = grades.iloc[0]
        grades = grades.iloc[1:]
        
        return special_notes, grades
        
    except Exception as e:
        raise Exception(f"CSV 파일 처리 중 오류가 발생했습니다: {str(e)}")

def extract_student_info(special_notes: pd.DataFrame, grades: pd.DataFrame) -> Dict[str, Any]:
    """처리된 데이터에서 학생 정보를 추출합니다."""
    try:
        # 컬럼명 가져오기
        columns = special_notes.columns.tolist()
        
        # 교과별 세특
        subjects = columns[:11]  # 처음 11개 컬럼은 교과목
        subject_notes = {subject: str(special_notes.iloc[1][subject]) 
                        for subject in subjects if pd.notna(special_notes.iloc[1][subject])}
        
        # 활동 내역
        activity_columns = columns[11:-1]  # 마지막 컬럼 제외한 나머지
        activities = {col: str(special_notes.iloc[1][col]) 
                     for col in activity_columns if pd.notna(special_notes.iloc[1][col])}
        
        # 진로 희망
        career = str(special_notes.iloc[1][columns[-1]]) if pd.notna(special_notes.iloc[1][columns[-1]]) else "미정"
        
        # 성적 데이터 처리
        semester_grades = {'semester1': {'grades': {}, 'average': {'total': 0.0, 'main_subjects': 0.0}},
                         'semester2': {'grades': {}, 'average': {'total': 0.0, 'main_subjects': 0.0}},
                         'total': {'average': {'total': 0.0, 'main_subjects': 0.0}}}
        
        main_subjects = ['국어', '수학', '영어', '한국사', '사회', '과학', '정보']
        
        for _, row in grades.iterrows():
            semester = '1' if row['학기'] == 1 else '2'
            subject = row['과목']
            
            if pd.notna(row['석차등급']):
                grade_info = {
                    'raw_score': float(row['원점수/과목평균(표준편차)'].split('/')[0]),
                    'rank': int(row['석차등급'])
                }
                semester_grades[f'semester{semester}']['grades'][subject] = grade_info
        
        # 평균 계산
        for semester in ['semester1', 'semester2']:
            grades_dict = semester_grades[semester]['grades']
            if grades_dict:
                # 전체 평균
                total_scores = [g['raw_score'] for g in grades_dict.values()]
                semester_grades[semester]['average']['total'] = sum(total_scores) / len(total_scores)
                
                # 주요 과목 평균
                main_scores = [g['raw_score'] for s, g in grades_dict.items() if s in main_subjects]
                if main_scores:
                    semester_grades[semester]['average']['main_subjects'] = sum(main_scores) / len(main_scores)
        
        # 전체 평균 계산
        all_scores = []
        main_subject_scores = []
        for semester in ['semester1', 'semester2']:
            grades_dict = semester_grades[semester]['grades']
            all_scores.extend([g['raw_score'] for g in grades_dict.values()])
            main_subject_scores.extend([g['raw_score'] for s, g in grades_dict.items() if s in main_subjects])
        
        if all_scores:
            semester_grades['total']['average']['total'] = sum(all_scores) / len(all_scores)
        if main_subject_scores:
            semester_grades['total']['average']['main_subjects'] = sum(main_subject_scores) / len(main_subject_scores)
        
        return {
            'special_notes': {
                'subjects': subject_notes,
                'activities': activities
            },
            'academic_records': semester_grades,
            'career_aspiration': career
        }
        
    except Exception as e:
        raise Exception(f"학생 정보 추출 중 오류가 발생했습니다: {str(e)}")

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

def analyze_grades(grade_data: pd.DataFrame) -> Dict[str, Any]:
    """성적 데이터를 분석하여 다양한 통계를 생성합니다."""
    # 분석할 주요 과목 리스트
    main_subjects = ['국어', '영어', '수학', '사회', '과학']
    all_subjects = main_subjects + ['한국사', '정보']
    
    # 학기별 데이터 분리
    semester1_data = grade_data[grade_data['학 기'] == 1]
    semester2_data = grade_data[grade_data['학 기'] == 2]
    
    # 결과 저장을 위한 딕셔너리
    analysis_result = {
        '1학기': {'과목별_등급': {}, '가중_평균': 0, '단순_평균': 0},
        '2학기': {'과목별_등급': {}, '가중_평균': 0, '단순_평균': 0},
        '전체': {'주요과목_평균': 0, '전체과목_평균': 0}
    }
    
    # 학기별 분석
    for semester, data in [('1학기', semester1_data), ('2학기', semester2_data)]:
        total_credits = 0
        weighted_sum = 0
        grades_sum = 0
        subject_count = 0
        
        for _, row in data.iterrows():
            subject = row['과 목']
            if pd.notna(row['석차등급']) and subject in all_subjects:
                grade = float(row['석차등급'])
                credits = float(row['학점수']) if pd.notna(row['학점수']) else 1
                
                analysis_result[semester]['과목별_등급'][subject] = {
                    '등급': grade,
                    '학점수': credits
                }
                
                weighted_sum += grade * credits
                total_credits += credits
                grades_sum += grade
                subject_count += 1
        
        if total_credits > 0:
            analysis_result[semester]['가중_평균'] = round(weighted_sum / total_credits, 2)
        if subject_count > 0:
            analysis_result[semester]['단순_평균'] = round(grades_sum / subject_count, 2)
    
    # 전체 평균 계산
    main_subject_grades = []
    all_subject_grades = []
    
    for semester in ['1학기', '2학기']:
        for subject, data in analysis_result[semester]['과목별_등급'].items():
            if subject in main_subjects:
                main_subject_grades.append(data['등급'])
            if subject in all_subjects:
                all_subject_grades.append(data['등급'])
    
    if main_subject_grades:
        analysis_result['전체']['주요과목_평균'] = round(sum(main_subject_grades) / len(main_subject_grades), 2)
    if all_subject_grades:
        analysis_result['전체']['전체과목_평균'] = round(sum(all_subject_grades) / len(all_subject_grades), 2)
    
    return analysis_result

def create_grade_comparison_chart(grade_analysis: Dict[str, Any]) -> go.Figure:
    """학기별 과목 등급을 비교하는 차트를 생성합니다."""
    fig = go.Figure()
    
    # 1학기 데이터
    subjects_1 = list(grade_analysis['1학기']['과목별_등급'].keys())
    grades_1 = [grade_analysis['1학기']['과목별_등급'][subject]['등급'] for subject in subjects_1]
    
    # 2학기 데이터
    subjects_2 = list(grade_analysis['2학기']['과목별_등급'].keys())
    grades_2 = [grade_analysis['2학기']['과목별_등급'][subject]['등급'] for subject in subjects_2]
    
    # 1학기 막대 그래프
    fig.add_trace(go.Bar(
        name='1학기',
        x=subjects_1,
        y=grades_1,
        text=grades_1,
        textposition='auto',
    ))
    
    # 2학기 막대 그래프
    fig.add_trace(go.Bar(
        name='2학기',
        x=subjects_2,
        y=grades_2,
        text=grades_2,
        textposition='auto',
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title='학기별 과목 등급 비교',
        xaxis_title='과목',
        yaxis_title='등급',
        yaxis=dict(
            range=[9.5, 0.5],  # 1등급이 위로 가도록 y축 반전
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        barmode='group',
        showlegend=True
    )
    
    return fig

def create_average_comparison_chart(grade_analysis: Dict[str, Any]) -> go.Figure:
    """평균 등급을 비교하는 차트를 생성합니다."""
    fig = go.Figure()
    
    categories = ['1학기 가중평균', '1학기 단순평균', '2학기 가중평균', '2학기 단순평균', '주요과목 평균', '전체과목 평균']
    values = [
        grade_analysis['1학기']['가중_평균'],
        grade_analysis['1학기']['단순_평균'],
        grade_analysis['2학기']['가중_평균'],
        grade_analysis['2학기']['단순_평균'],
        grade_analysis['전체']['주요과목_평균'],
        grade_analysis['전체']['전체과목_평균']
    ]
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        text=[f'{v:.2f}' for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='등급 평균 비교',
        xaxis_title='구분',
        yaxis_title='등급',
        yaxis=dict(
            range=[9.5, 0.5],  # 1등급이 위로 가도록 y축 반전
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )
    
    return fig

def create_credit_weighted_chart(grade_analysis: Dict[str, Any]) -> go.Figure:
    """학점 가중치를 고려한 과목별 차트를 생성합니다."""
    fig = go.Figure()
    
    # 1학기 데이터
    subjects_1 = list(grade_analysis['1학기']['과목별_등급'].keys())
    grades_1 = []
    weighted_grades_1 = []
    
    for subject in subjects_1:
        data = grade_analysis['1학기']['과목별_등급'][subject]
        grades_1.append(data['등급'])
        weighted_grades_1.append(data['등급'] * data['학점수'])
    
    # 2학기 데이터
    subjects_2 = list(grade_analysis['2학기']['과목별_등급'].keys())
    grades_2 = []
    weighted_grades_2 = []
    
    for subject in subjects_2:
        data = grade_analysis['2학기']['과목별_등급'][subject]
        grades_2.append(data['등급'])
        weighted_grades_2.append(data['등급'] * data['학점수'])
    
    # 1학기 일반 등급
    fig.add_trace(go.Scatter(
        name='1학기 등급',
        x=subjects_1,
        y=grades_1,
        mode='lines+markers',
        line=dict(dash='solid')
    ))
    
    # 1학기 가중 등급
    fig.add_trace(go.Scatter(
        name='1학기 가중등급',
        x=subjects_1,
        y=weighted_grades_1,
        mode='lines+markers',
        line=dict(dash='dot')
    ))
    
    # 2학기 일반 등급
    fig.add_trace(go.Scatter(
        name='2학기 등급',
        x=subjects_2,
        y=grades_2,
        mode='lines+markers',
        line=dict(dash='solid')
    ))
    
    # 2학기 가중 등급
    fig.add_trace(go.Scatter(
        name='2학기 가중등급',
        x=subjects_2,
        y=weighted_grades_2,
        mode='lines+markers',
        line=dict(dash='dot')
    ))
    
    fig.update_layout(
        title='과목별 등급과 가중등급 비교',
        xaxis_title='과목',
        yaxis_title='등급',
        yaxis=dict(
            range=[max(max(weighted_grades_1), max(weighted_grades_2)) + 0.5, 0.5],
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        showlegend=True
    )
    
    return fig 