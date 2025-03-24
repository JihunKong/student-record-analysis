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
        # CSV 파일 읽기
        df = pd.read_csv(file, encoding='utf-8')
        
        # 빈 열 제거
        df = df.dropna(axis=1, how='all')
        
        # 첫 번째 행을 컬럼명으로 설정
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        
        # 인덱스 재설정
        df = df.reset_index(drop=True)
        
        return df
    except Exception as e:
        raise Exception(f"CSV 파일 전처리 중 오류 발생: {str(e)}")

def extract_student_info(df):
    """DataFrame에서 학생 정보를 추출합니다."""
    try:
        student_info = {}
        
        # 교과별 세부능력 및 특기사항
        academic_performance = {}
        for subject in ['국어', '수학', '영어', '한국사', '사회', '과학', '과학탐구실험', '정보', '체육', '음악', '미술']:
            if subject in df.columns:
                content = df[subject].iloc[0]
                if pd.notna(content):
                    academic_performance[subject] = content
        
        # 활동 내역
        activities = {}
        for activity_type in ['자율', '동아리', '진로', '행특', '개인']:
            if activity_type in df.columns:
                content = df[activity_type].iloc[0]
                if pd.notna(content):
                    activities[activity_type] = content
        
        # 진로 희망
        career_aspiration = ""
        if '진로희망' in df.columns:
            career_aspiration = df['진로희망'].iloc[0]
        
        # 학기별 성적
        grades = []
        # 성적 데이터가 있는 행 찾기
        grade_section = df[df.iloc[:, 0] == '학기'].index[0]
        grade_data = df.iloc[grade_section+1:].copy()
        
        if not grade_data.empty:
            # 성적 데이터 컬럼명 설정
            grade_columns = ['학기', '교과', '과목', '학점수', '원점수/과목평균', '성취도', '석차등급']
            grade_data = grade_data.iloc[:, :len(grade_columns)]
            grade_data.columns = grade_columns
            
            # 빈 행 제거
            grade_data = grade_data.dropna(subset=['학기', '교과', '과목'])
            grades = grade_data.to_dict('records')
        
        # 추출한 정보를 student_info에 저장
        student_info['academic_performance'] = academic_performance
        student_info['activities'] = activities
        student_info['career_aspiration'] = career_aspiration
        student_info['grades'] = grades
        
        return student_info
    except Exception as e:
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