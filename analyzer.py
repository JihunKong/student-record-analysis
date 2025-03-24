import os
import google.generativeai as genai
from typing import Dict, Any, List
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from dotenv import load_dotenv

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

def analyze_with_gemini(prompt: str) -> Dict[str, Any]:
    """Gemini API를 사용하여 분석을 수행합니다."""
    response = model.generate_content(prompt)
    
    # JSON 문자열 추출 및 파싱
    try:
        # 응답에서 JSON 부분 추출
        json_str = response.text
        # 만약 마크다운 코드 블록으로 감싸진 경우 제거
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()
        
        result = json.loads(json_str)
        return result
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"원본 응답: {response.text}")
        return {"error": str(e), "raw_response": response.text}

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

def analyze_student_record(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """학생 생활기록부를 종합적으로 분석합니다."""
    try:
        # 기본 정보 추출
        basic_info = {
            "학년": student_data.get("학년", ""),
            "반": student_data.get("반", ""),
            "번호": student_data.get("번호", ""),
            "이름": student_data.get("이름", ""),
            "진로희망": student_data.get("진로희망", "")
        }
        
        # 교과별 성취도 분석
        academic_performance = {}
        for subject in ["국어", "수학", "영어", "한국사", "사회", "과학", "과학탐구실험", "정보", "체육", "음악", "미술"]:
            if subject in student_data:
                prompt = f"""
                다음은 {subject} 과목의 성취도 내용입니다. 
                이 내용을 바탕으로 학생의 성취 수준과 특징을 분석해주세요.
                
                내용: {student_data[subject]}
                """
                
                response = model.generate_content(prompt)
                academic_performance[subject] = response.text
        
        # 활동 내역 분석
        activities = {}
        for activity_type in ["자율", "동아리", "진로", "행특", "개인"]:
            if activity_type in student_data:
                prompt = f"""
                다음은 {activity_type} 활동 내용입니다.
                이 활동을 통해 보여진 학생의 특성과 역량을 분석해주세요.
                
                내용: {student_data[activity_type]}
                """
                
                response = model.generate_content(prompt)
                activities[activity_type] = response.text
        
        # 진로 적합성 분석
        career_prompt = f"""
        다음은 학생의 진로 희망과 활동 내역입니다.
        이를 바탕으로 진로 적합성을 분석해주세요.
        
        진로 희망: {basic_info["진로희망"]}
        활동 내역: {json.dumps(activities, ensure_ascii=False)}
        """
        
        career_response = model.generate_content(career_prompt)
        
        # 통합 분석 결과 생성
        analysis_results = {
            "학생_프로필": {
                "기본_정보": basic_info,
                "교과별_분석": academic_performance,
                "활동_분석": activities
            },
            "진로_적합성": career_response.text
        }
        
        # 시각화 데이터 추가
        analysis_results["시각화"] = {
            "교과_성취도": create_subject_radar_chart(student_data.get("교과_성취도", {})),
            "활동_타임라인": create_activity_timeline(student_data.get("활동_내역", [])),
            "진로_분석": create_career_analysis_chart({"적합_진로_옵션": []}),
            "학습_스타일": create_learning_style_chart(activities),
            "핵심_역량": create_competency_chart(activities)
        }
        
        return analysis_results
        
    except Exception as e:
        raise Exception(f"분석 중 오류가 발생했습니다: {str(e)}") 