import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re
import os
import json
import anthropic
from typing import Dict, List, Any, Tuple
from datetime import datetime

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

def extract_student_info(special_notes: pd.DataFrame, grades: pd.DataFrame) -> Dict:
    """학생 정보를 추출합니다."""
    try:
        print("학생 정보 추출 시작...")
        student_info = {}
        
        # 초기 데이터 구조 설정
        subject_notes = {}
        activities = {}
        career = "미정"
        semester_grades = {
            'semester1': {'grades': {}, 'average': {'total': 0.0, 'main_subjects': 0.0}},
            'semester2': {'grades': {}, 'average': {'total': 0.0, 'main_subjects': 0.0}},
            'total': {'average': {'total': 0.0, 'main_subjects': 0.0}}
        }
        
        # 세특 데이터 처리
        if not special_notes.empty:
            print(f"세특 데이터 컬럼: {special_notes.columns.tolist()}")
            # 컬럼 구분
            subject_cols = ['국어', '수학', '영어', '한국사', '사회', '과학', '과학탐구실험', '정보', '체육', '음악', '미술']
            activity_cols = ['자율', '동아리', '진로', '행특', '개인']
            career_cols = ['진로희망']
            
            # 존재하는 컬럼만 필터링
            existing_subject_cols = [col for col in subject_cols if col in special_notes.columns]
            existing_activity_cols = [col for col in activity_cols if col in special_notes.columns]
            existing_career_cols = [col for col in career_cols if col in special_notes.columns]
            
            print(f"존재하는 교과 컬럼: {existing_subject_cols}")
            print(f"존재하는 활동 컬럼: {existing_activity_cols}")
            print(f"존재하는 진로 컬럼: {existing_career_cols}")
            
            # 세특 데이터 추출
            for col in special_notes.columns:
                try:
                    if col in subject_cols and col in special_notes.columns and special_notes[col].notna().any():
                        # 교과별 세특
                        val = special_notes[col].dropna().iloc[0] if len(special_notes[col].dropna()) > 0 else ""
                        if val:
                            subject_notes[col] = str(val)
                            print(f"교과 '{col}' 정보 추출 성공")
                    elif col in activity_cols or any(act in col for act in activity_cols):
                        # 활동 내역
                        val = special_notes[col].dropna().iloc[0] if len(special_notes[col].dropna()) > 0 else ""
                        if val:
                            activities[col] = str(val)
                            print(f"활동 '{col}' 정보 추출 성공")
                    elif col in career_cols:
                        # 진로 희망
                        val = special_notes[col].dropna().iloc[0] if len(special_notes[col].dropna()) > 0 else "미정"
                        if val:
                            career = str(val)
                            print(f"진로 희망 '{career}' 추출 성공")
                except Exception as e:
                    print(f"컬럼 '{col}' 처리 중 오류: {str(e)}")
        else:
            print("세특 데이터가 비어 있습니다.")
        
        # 성적 데이터 처리
        main_subjects = ['국어', '수학', '영어', '사회', '과학', '한국사', '정보']  # 주요 과목 리스트
        
        if not grades.empty:
            print(f"성적 데이터 컬럼: {grades.columns.tolist()}")
            # 성적 데이터 컬럼 찾기
            semester_col = None
            subject_col = None
            raw_score_col = None
            grade_col = None
            credit_col = None
            
            # 컬럼 이름 찾기
            for col in grades.columns:
                col_str = str(col).lower()
                if '학기' in col_str:
                    semester_col = col
                elif '과목' in col_str:
                    subject_col = col
                elif '원점수' in col_str:
                    raw_score_col = col
                elif '석차등급' in col_str:
                    grade_col = col
                elif '학점수' in col_str:
                    credit_col = col
            
            print(f"식별된 컬럼 - 학기: {semester_col}, 과목: {subject_col}, 원점수: {raw_score_col}, 등급: {grade_col}, 학점수: {credit_col}")
            
            # 필수 컬럼이 없는 경우 오류 메시지 출력
            if not (semester_col and subject_col):
                print("필수 컬럼(학기, 과목)을 찾을 수 없습니다.")
                
            # 성적 데이터 추출
            for _, row in grades.iterrows():
                try:
                    if semester_col and subject_col and pd.notna(row[semester_col]) and pd.notna(row[subject_col]):
                        semester = str(row[semester_col]).strip()
                        subject = str(row[subject_col]).strip()
                        
                        if semester in ['1', '2']:
                            # 석차등급
                            grade_value = 0
                            if grade_col and pd.notna(row[grade_col]):
                                try:
                                    grade_value = float(str(row[grade_col]).strip())
                                except ValueError:
                                    print(f"등급 변환 오류 - 원본값: '{row[grade_col]}'")
                                    grade_value = 0
                            
                            # 원점수
                            raw_score = 0
                            if raw_score_col and pd.notna(row[raw_score_col]):
                                try:
                                    raw_score_text = str(row[raw_score_col]).strip().split('/')[0]
                                    raw_score = float(raw_score_text)
                                except (ValueError, IndexError) as e:
                                    print(f"원점수 변환 오류 - 원본값: '{row[raw_score_col]}', 오류: {str(e)}")
                                    raw_score = 0
                            
                            # 학점수
                            credit = 1.0
                            if credit_col and pd.notna(row[credit_col]):
                                try:
                                    credit = float(str(row[credit_col]).strip())
                                except ValueError:
                                    print(f"학점수 변환 오류 - 원본값: '{row[credit_col]}'")
                                    credit = 1.0
                            
                            # 과목 정보 저장
                            if grade_value > 0:
                                grade_info = {
                                    'raw_score': raw_score,
                                    'rank': grade_value,
                                    'credit': credit
                                }
                                semester_grades[f'semester{semester}']['grades'][subject] = grade_info
                                print(f"{semester}학기 '{subject}' 성적 추출: 원점수={raw_score}, 등급={grade_value}, 학점={credit}")
                except Exception as e:
                    print(f"성적 행 처리 중 오류: {str(e)}")
                    continue
            
            # 평균 계산
            for semester in ['semester1', 'semester2']:
                grades_dict = semester_grades[semester]['grades']
                if grades_dict:
                    # 전체 평균 계산
                    total_scores = [g['raw_score'] for g in grades_dict.values()]
                    total_credits = [g['credit'] for g in grades_dict.values()]
                    
                    if total_scores and total_credits:
                        # 학점 가중 평균
                        weighted_sum = sum(score * credit for score, credit in zip(total_scores, total_credits))
                        total_credit = sum(total_credits)
                        semester_grades[semester]['average']['total'] = weighted_sum / total_credit if total_credit > 0 else 0
                        
                        # 주요 과목 평균 계산
                        main_subject_scores = [grades_dict[subject]['raw_score'] for subject in grades_dict if subject in main_subjects]
                        main_subject_credits = [grades_dict[subject]['credit'] for subject in grades_dict if subject in main_subjects]
                        
                        if main_subject_scores and main_subject_credits:
                            main_weighted_sum = sum(score * credit for score, credit in zip(main_subject_scores, main_subject_credits))
                            main_total_credit = sum(main_subject_credits)
                            semester_grades[semester]['average']['main_subjects'] = main_weighted_sum / main_total_credit if main_total_credit > 0 else 0
                    
                    print(f"{semester} 전체 평균: {semester_grades[semester]['average']['total']:.2f}")
                    print(f"{semester} 주요과목 평균: {semester_grades[semester]['average']['main_subjects']:.2f}")
        else:
            print("성적 데이터가 비어 있습니다.")
        
        # 전체 평균 계산
        total_semester1 = semester_grades['semester1']['average']['total']
        total_semester2 = semester_grades['semester2']['average']['total']
        
        if total_semester1 > 0 and total_semester2 > 0:
            semester_grades['total']['average']['total'] = (total_semester1 + total_semester2) / 2
        elif total_semester1 > 0:
            semester_grades['total']['average']['total'] = total_semester1
        elif total_semester2 > 0:
            semester_grades['total']['average']['total'] = total_semester2
            
        # 주요 과목 전체 평균 계산
        main_semester1 = semester_grades['semester1']['average']['main_subjects']
        main_semester2 = semester_grades['semester2']['average']['main_subjects']
        
        if main_semester1 > 0 and main_semester2 > 0:
            semester_grades['total']['average']['main_subjects'] = (main_semester1 + main_semester2) / 2
        elif main_semester1 > 0:
            semester_grades['total']['average']['main_subjects'] = main_semester1
        elif main_semester2 > 0:
            semester_grades['total']['average']['main_subjects'] = main_semester2
            
        print(f"전체 평균: {semester_grades['total']['average']['total']:.2f}")
        print(f"주요과목 전체 평균: {semester_grades['total']['average']['main_subjects']:.2f}")
        
        # 결과 조합
        student_info = {
            'special_notes': {
                'subjects': subject_notes,
                'activities': activities,
                'career': career
            },
            'academic_records': semester_grades,
            'career_aspiration': career
        }
        
        # 디버깅용 데이터 구조 출력
        print("학생 정보 추출 완료")
        print(f"추출된 교과 정보: {len(subject_notes)}개")
        print(f"추출된 활동 정보: {len(activities)}개")
        print(f"진로 희망: {career}")
        
        return student_info
        
    except Exception as e:
        print(f"학생 정보 추출 중 예외 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'special_notes': {
                'subjects': {},
                'activities': {},
                'career': "미정"
            },
            'academic_records': {
                'semester1': {'grades': {}, 'average': {'total': 0.0, 'main_subjects': 0.0}},
                'semester2': {'grades': {}, 'average': {'total': 0.0, 'main_subjects': 0.0}},
                'total': {'average': {'total': 0.0, 'main_subjects': 0.0}}
            },
            'career_aspiration': "미정"
        }

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

def process_csv_file(file_path: str) -> Dict:
    """CSV 파일을 처리하여 세특 데이터와 성적 데이터를 구분하여 반환"""
    try:
        # 파일 읽기 - 헤더 없이 읽기
        df = pd.read_csv(file_path, header=None, encoding='utf-8')
        
        # 세특 데이터 (1~2행)
        special_notes_row = df.iloc[1].tolist()  # 첫 번째 행이 헤더, 두 번째 행이 세특 데이터
        
        # 세특 및 활동 데이터 추출
        subjects = {}
        activities = {}
        career_aspiration = '미정'
        
        # 컬럼 이름 가져오기 (첫 번째 행)
        headers = df.iloc[0].tolist()
        
        # 세특 데이터 처리
        for i, header in enumerate(headers):
            if pd.notna(header) and pd.notna(special_notes_row[i]):
                if header in ['국어', '수학', '영어', '한국사', '사회', '과학', '과학탐구실험', '정보', '체육', '음악', '미술']:
                    subjects[header] = special_notes_row[i]
                elif header in ['자율', '동아리', '진로', '행특', '개인']:
                    activities[header] = special_notes_row[i]
                elif header == '진로희망' and pd.notna(special_notes_row[i]):
                    career_aspiration = special_notes_row[i]
        
        # 성적 데이터 (4행부터)
        grade_data_start = 3  # 4번째 행부터 성적 데이터 시작 (0-based index)
        grade_data = df.iloc[grade_data_start:].copy()
        
        # 성적 데이터는 처음 6개 열만 사용
        grade_data = grade_data.iloc[:, :6]
        
        # 성적 데이터 컬럼 이름 설정
        grade_data.columns = ['학기', '과목', '학점수', '원점수/과목평균', '성취도', '석차등급']
        
        # 빈 행 제거
        grade_data = grade_data.dropna(subset=['학기', '과목'], how='all')
        
        # 성적 데이터 처리
        semester1_grades = {}
        semester2_grades = {}
        
        for _, row in grade_data.iterrows():
            try:
                semester = str(row['학기']).strip()
                subject = str(row['과목']).strip()
                
                if pd.notna(row['석차등급']) and pd.notna(row['학점수']):
                    try:
                        rank = float(row['석차등급'])
                        credit = float(row['학점수'])
                        
                        # 원점수 추출
                        raw_score = 0
                        if pd.notna(row['원점수/과목평균']):
                            raw_score_str = str(row['원점수/과목평균']).split('/')[0].strip()
                            try:
                                raw_score = float(raw_score_str)
                            except:
                                # 원점수를 파싱할 수 없는 경우, 등급에 따른 근사치 계산
                                raw_score = 100 - ((rank - 1) * 10)
                        
                        grade_info = {
                            'rank': rank,
                            'raw_score': raw_score,
                            'credit': credit
                        }
                        
                        # 학기별 데이터 저장
                        if semester == '1':
                            semester1_grades[subject] = grade_info
                        elif semester == '2':
                            semester2_grades[subject] = grade_info
                    except Exception as e:
                        print(f"성적 변환 오류: {e}")
            except Exception as e:
                print(f"성적 행 처리 중 오류: {e}")
        
        # 평균 계산
        semester1_avg = calculate_semester_average(semester1_grades)
        semester2_avg = calculate_semester_average(semester2_grades)
        
        # 결과 데이터 구성
        student_data = {
            'special_notes': {
                'subjects': subjects,
                'activities': activities
            },
            'academic_records': {
                'semester1': {'grades': semester1_grades, 'average': semester1_avg},
                'semester2': {'grades': semester2_grades, 'average': semester2_avg}
            },
            'career_aspiration': career_aspiration
        }
        
        return student_data
    except Exception as e:
        print(f"CSV 파일 처리 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def calculate_semester_average(grades: Dict) -> Dict:
    """학기별 평균 계산"""
    if not grades:
        return {'total': 0.0, 'main_subjects': 0.0}
    
    # 전체 평균 계산
    total_rank_sum = sum([g['rank'] for g in grades.values()])
    total_rank_avg = total_rank_sum / len(grades) if grades else 0
    
    # 주요과목 평균 계산 (국어, 영어, 수학)
    main_subjects = [subj for subj in grades.keys() if any(key in subj for key in ['국어', '영어', '수학'])]
    if main_subjects:
        main_rank_sum = sum([grades[subj]['rank'] for subj in main_subjects])
        main_rank_avg = main_rank_sum / len(main_subjects)
    else:
        main_rank_avg = 0
    
    return {
        'total': total_rank_avg,
        'main_subjects': main_rank_avg
    }

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