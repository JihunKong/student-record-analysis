import os
import google.generativeai as genai
from typing import Dict, Any, List
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from dotenv import load_dotenv
import re
import anthropic
import streamlit as st

# .env 파일 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-002')

def create_prompt_for_student_profile(student_data: Dict[str, Any]) -> str:
    """학생 프로필 분석을 위한 프롬프트를 생성합니다."""
    prompt = f"""
    다음은 고등학교 학생의 생활기록부 정보입니다:
    
    {json.dumps(student_data, ensure_ascii=False, indent=2)}
    
    위 데이터를 분석하여 학생 프로필 요약을 작성해주세요. 다음 내용을 포함해야 합니다:
    1. 학생의 기본 정보 요약
    2. 학생의 강점과 약점
    3. 학생의 학업 성취도 패턴
    4. 학생의 인성 및 생활 태도
    
    실제 데이터에 기반한 현실적인 분석을 제공해주세요.
    결과는 JSON 포맷으로 반환해주세요.
    """
    return prompt

def create_prompt_for_career_analysis(student_data: Dict[str, Any]) -> str:
    """진로 적합성 분석을 위한 프롬프트를 생성합니다."""
    prompt = f"""
    다음은 고등학교 학생의 생활기록부 정보입니다:
    
    {json.dumps(student_data, ensure_ascii=False, indent=2)}
    
    위 데이터를 분석하여 학생의 진로 적합성을 평가해주세요. 다음 내용을 포함해야 합니다:
    1. 학생이 희망하는 진로와 현재 역량 사이의 일치도
    2. 학생에게 적합할 수 있는 다른 진로 옵션 3가지
    3. 각 진로 옵션의 적합 이유와 해당 진로를 위해 보완해야 할 부분
    4. 진로 성취를 위한 권장 단계별 계획
    
    결과는 JSON 포맷으로 반환해주세요.
    """
    return prompt

def create_prompt_for_academic_strategy(student_data: Dict[str, Any]) -> str:
    """학업 발전 전략 분석을 위한 프롬프트를 생성합니다."""
    prompt = f"""
    다음은 고등학교 학생의 생활기록부 정보입니다:
    
    {json.dumps(student_data, ensure_ascii=False, indent=2)}
    
    위 데이터를 분석하여 학생의 학업 발전 전략을 제안해주세요. 다음 내용을 포함해야 합니다:
    1. 각 교과목별 현재 성취도와 발전 가능성
    2. 학생의 학습 스타일과 효과적인 학습 방법 제안
    3. 취약 과목에 대한 개선 전략
    4. 강점 과목을 더욱 발전시키기 위한 방안
    5. 대학 입시를 고려한 학업 로드맵
    
    결과는 JSON 포맷으로 반환해주세요.
    """
    return prompt

def create_prompt_for_parent_consultation(student_data: Dict[str, Any]) -> str:
    """학부모 상담 가이드 생성을 위한 프롬프트를 생성합니다."""
    prompt = f"""
    다음은 고등학교 학생의 생활기록부 정보입니다:
    
    {json.dumps(student_data, ensure_ascii=False, indent=2)}
    
    위 데이터를 바탕으로 학부모 상담 가이드를 작성해주세요. 다음 내용을 포함해야 합니다:
    1. 학생의 현재 상황에 대한 객관적 평가
    2. 학부모가 가정에서 지원할 수 있는 구체적인 방법
    3. 학생의 성장을 위해 학부모가 주의해야 할 점
    4. 학생과 학부모 간 효과적인 소통 방법
    5. 학부모가 주목해야 할 학생의 특성과 그에 따른 교육적 접근법
    
    학생의 긍정적인 발전을 위한 건설적인 조언을 제공해주세요.
    결과는 JSON 포맷으로 반환해주세요.
    """
    return prompt

def create_prompt_for_career_roadmap(student_data: Dict[str, Any]) -> str:
    """진로 로드맵 생성을 위한 프롬프트를 생성합니다."""
    prompt = f"""
    다음은 고등학교 학생의 생활기록부 정보입니다:
    
    {json.dumps(student_data, ensure_ascii=False, indent=2)}
    
    위 데이터를 바탕으로 학생의 진로 로드맵을 작성해주세요. 다음 내용을 포함해야 합니다:
    1. 단기 목표 (고등학교 재학 중 달성할 목표)
    2. 중기 목표 (대학 진학 및 대학 생활 중 달성할 목표)
    3. 장기 목표 (졸업 후 커리어 목표)
    4. 각 단계별 추천 활동 및 경험
    5. 목표 달성을 위한 시간선 상의 주요 이정표
    
    학생의 현재 상황과 희망 진로를 고려하여 현실적이고 구체적인 로드맵을 제시해주세요.
    결과는 JSON 포맷으로 반환해주세요.
    """
    return prompt

def create_analysis_prompt(student_data: Dict[str, Any]) -> str:
    # 성적 데이터 요약
    grades_summary = []
    for semester in ['semester1', 'semester2']:
        semester_data = student_data['academic_records'][semester]
        grades = semester_data['grades']
        averages = semester_data.get('average', {})
        
        semester_summary = f"{semester.replace('semester', '')}학기:\n"
        semester_summary += f"- 전체 평균: {averages.get('total', 0):.1f}\n"
        semester_summary += f"- 주요과목 평균: {averages.get('main_subjects', 0):.1f}\n"
        semester_summary += "- 과목별 등급:\n"
        
        for subject, grade in grades.items():
            semester_summary += f"  * {subject}: {grade['rank']}등급 (원점수: {grade['raw_score']:.1f})\n"
        
        grades_summary.append(semester_summary)

    # 세특 데이터 요약
    special_notes = []
    for subject, content in student_data['special_notes']['subjects'].items():
        special_notes.append(f"[{subject}]\n{content}\n")

    # 진로 희망
    career = student_data['special_notes'].get('career', '미정')

    prompt = f"""
다음은 한 학생의 학업 데이터입니다. 이를 바탕으로 학생의 특성과 발전 가능성을 분석해주세요.

1. 성적 데이터
{'\n'.join(grades_summary)}

2. 세부능력 및 특기사항
{'\n'.join(special_notes)}

3. 진로 희망: {career}

위 데이터를 바탕으로 다음 항목들을 분석해주세요:

1. 학업 역량 분석
- 전반적인 학업 수준과 발전 추이
- 과목별 특징과 강점
- 학습 태도와 참여도

2. 진로 적합성 분석
- 희망 진로와 현재 역량의 연관성
- 진로 실현을 위한 준비 상태
- 발전 가능성과 보완이 필요한 부분

3. 종합 제언
- 학생의 주요 강점과 특징
- 향후 발전을 위한 구체적 조언
- 진로 실현을 위한 추천 활동

분석은 객관적 데이터를 기반으로 하되, 긍정적이고 발전적인 관점에서 작성해주세요.
"""
    return prompt

def analyze_with_claude(student_data: Dict[str, Any]) -> str:
    try:
        client = anthropic.Anthropic(
            api_key=st.secrets["ANTHROPIC_API_KEY"]
        )
        
        prompt = create_analysis_prompt(student_data)
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.7,
            system="당신은 학생 데이터를 분석하는 전문가입니다. 객관적인 데이터를 기반으로 학생의 강점을 발견하고 발전 가능성을 제시해주세요.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return message.content
        
    except Exception as e:
        return f"AI 분석 중 오류가 발생했습니다: {str(e)}"

def create_subject_radar_chart(subject_data: Dict[str, Any]) -> go.Figure:
    """교과별 성취도를 레이더 차트로 시각화합니다."""
    subjects = list(subject_data.keys())
    scores = [float(subject_data[subject]['성취도']) for subject in subjects]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=subjects,
        fill='toself',
        name='성취도'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="교과별 성취도 분석"
    )
    
    return fig

def create_activity_timeline(activities: List[Dict[str, Any]]) -> go.Figure:
    """활동 내역을 타임라인으로 시각화합니다."""
    fig = go.Figure()
    
    for activity in activities:
        fig.add_trace(go.Scatter(
            x=[activity['날짜']],
            y=[1],
            mode='markers+text',
            name=activity['활동명'],
            text=activity['활동명'],
            textposition="top center",
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="활동 타임라인",
        showlegend=False,
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(title="날짜")
    )
    
    return fig

def create_career_analysis_chart(career_data: Dict[str, Any]) -> go.Figure:
    """진로 적합성 분석을 시각화합니다."""
    fig = go.Figure()
    
    # 진로 옵션별 적합도 시각화
    options = career_data.get('적합_진로_옵션', [])
    if isinstance(options, list):
        for option in options:
            fig.add_trace(go.Bar(
                name=option.get('진로명', ''),
                x=['적합도'],
                y=[option.get('적합도', 0)],
                text=option.get('적합도', 0),
                textposition='auto',
            ))
    
    fig.update_layout(
        title="진로 옵션별 적합도 분석",
        barmode='group',
        yaxis=dict(title="적합도 (%)", range=[0, 100])
    )
    
    return fig

def create_learning_style_chart(style_data: Dict[str, Any]) -> go.Figure:
    """학습 스타일 분석을 시각화합니다."""
    fig = go.Figure()
    
    # 학습 스타일 요소별 점수 시각화
    styles = list(style_data.keys())
    scores = list(style_data.values())
    
    fig.add_trace(go.Bar(
        x=styles,
        y=scores,
        text=scores,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="학습 스타일 분석",
        yaxis=dict(title="점수", range=[0, 100])
    )
    
    return fig

def create_competency_chart(competency_data: Dict[str, Any]) -> go.Figure:
    """역량 분석을 시각화합니다."""
    fig = go.Figure()
    
    competencies = list(competency_data.keys())
    scores = list(competency_data.values())
    
    fig.add_trace(go.Bar(
        x=competencies,
        y=scores,
        text=scores,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="핵심 역량 분석",
        yaxis=dict(title="점수", range=[0, 100])
    )
    
    return fig

def analyze_student_record(student_data: Dict[str, Any], original_data: str = "") -> Dict[str, Any]:
    """학생 생활기록부를 분석하여 종합적인 결과를 반환합니다."""
    try:
        # 성적 데이터 분석
        grade_data = pd.DataFrame(student_data.get('grades', []))
        if not grade_data.empty:
            grade_analysis = analyze_grades(grade_data)
            
            # 차트 생성
            grade_charts = {
                '학기별_과목_비교': create_grade_comparison_chart(grade_analysis),
                '평균_비교': create_average_comparison_chart(grade_analysis),
                '가중치_비교': create_credit_weighted_chart(grade_analysis)
            }
        else:
            grade_analysis = {}
            grade_charts = {}
        
        # 기본 정보 추출
        basic_info = {
            "학년": str(student_data.get("학년", "")),
            "반": str(student_data.get("반", "")),
            "번호": str(student_data.get("번호", "")),
            "이름": str(student_data.get("이름", "")),
            "진로희망": str(student_data.get("진로희망", ""))
        }
        
        # 교과별 성취도 분석
        academic_performance = {}
        for subject in ["국어", "수학", "영어", "한국사", "사회", "과학", "과학탐구실험", "정보", "체육", "음악", "미술"]:
            if subject in student_data:
                academic_performance[subject] = str(student_data[subject]) if pd.notna(student_data[subject]) else ""
        
        # 활동 내역 분석
        activities = {}
        activity_types = ["자율", "동아리", "진로", "행특", "개인"]
        for activity_type in activity_types:
            if activity_type in student_data:
                activities[activity_type] = str(student_data[activity_type]) if pd.notna(student_data[activity_type]) else ""
        
        # 분석 결과 구성
        analysis_results = {
            "원본_데이터": original_data,
            "성적_분석": grade_analysis,
            "성적_차트": grade_charts,
            "학생_프로필": {
                "기본_정보_요약": f"{basic_info['학년']}학년 {basic_info['반']}반 {basic_info['번호']}번 {basic_info['이름']} 학생",
                "진로희망": basic_info["진로희망"],
                "강점": [],
                "약점": []
            },
            "교과_성취도": academic_performance,
            "활동_내역": activities,
            "진로_적합성": {
                "일치도": "",
                "적합_진로_옵션": []
            },
            "학업_발전_전략": {
                "교과목_분석": {},
                "권장_전략": []
            },
            "진로_로드맵": {
                "단기_목표": [],
                "중기_목표": [],
                "장기_목표": [],
                "추천_활동": {}
            }
        }
        
        # Gemini API를 통한 분석 수행
        profile_prompt = create_prompt_for_student_profile(student_data)
        career_prompt = create_prompt_for_career_analysis(student_data)
        academic_prompt = create_prompt_for_academic_strategy(student_data)
        roadmap_prompt = create_prompt_for_career_roadmap(student_data)
        
        # 각 분석 수행
        profile_analysis = analyze_with_claude(student_data)
        career_analysis = analyze_with_claude(student_data)
        academic_analysis = analyze_with_claude(student_data)
        roadmap_analysis = analyze_with_claude(student_data)
        
        # 분석 결과 통합
        if "error" not in profile_analysis:
            analysis_results["학생_프로필"].update(profile_analysis)
        if "error" not in career_analysis:
            analysis_results["진로_적합성"].update(career_analysis)
        if "error" not in academic_analysis:
            analysis_results["학업_발전_전략"].update(academic_analysis)
        if "error" not in roadmap_analysis:
            analysis_results["진로_로드맵"].update(roadmap_analysis)
        
        return analysis_results
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")  # 디버깅을 위한 출력 추가
        return {"error": str(e)} 