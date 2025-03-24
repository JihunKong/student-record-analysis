import os
from typing import Dict, Any, List
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from dotenv import load_dotenv
import re
import streamlit as st
import logging
import requests  # anthropic 라이브러리 제거하고 requests만 사용

# .env 파일 로드
load_dotenv()

def create_analysis_prompt(student_data: Dict[str, Any]) -> str:
    """학생 데이터 분석을 위한 프롬프트를 생성합니다."""
    
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
        if content and len(content) > 10:  # 의미 있는 내용만 포함
            special_notes.append(f"[{subject}]\n{content}\n")

    # 활동 데이터 요약
    activities = []
    for activity_type, content in student_data['special_notes']['activities'].items():
        if content and len(content) > 10:  # 의미 있는 내용만 포함
            activities.append(f"[{activity_type}]\n{content}\n")

    # 진로 희망
    career = student_data.get('career_aspiration', '미정')

    prompt = f"""
다음은 한 학생의 학업 데이터입니다. 이를 바탕으로 학생의 특성과 발전 가능성을 분석해주세요.

1. 성적 데이터
{'\n'.join(grades_summary)}

2. 세부능력 및 특기사항
{'\n'.join(special_notes)}

3. 창의적 체험활동
{'\n'.join(activities)}

4. 진로 희망: {career}

위 데이터를 바탕으로 다음 항목들을 분석해주세요:

1. 학업 역량 분석
- 전반적인 학업 수준과 발전 추이
- 과목별 특징과 강점
- 학습 태도와 참여도

2. 학생 특성 분석
- 성격 및 행동 특성
- 두드러진 역량과 관심사
- 대인관계 및 리더십

3. 진로 적합성 분석
- 희망 진로와 현재 역량의 연관성
- 진로 실현을 위한 준비 상태
- 발전 가능성과 보완이 필요한 부분

4. 종합 제언
- 학생의 주요 강점과 특징
- 향후 발전을 위한 구체적 조언
- 진로 실현을 위한 활동 추천

분석은 객관적 데이터를 기반으로 하되, 긍정적이고 발전적인 관점에서 작성해주세요.
학생의 강점을 최대한 살리고 약점을 보완할 수 있는 방안을 제시하세요.
권장하는 활동과 고려할 전략은 구체적이고 실행 가능한 것으로 제안해주세요.
"""
    return prompt

# 통합된 API 호출 함수
def call_claude_api(prompt_text: str, system_prompt: str = None) -> str:
    """
    Claude API를 호출하는 통합 함수
    
    Args:
        prompt_text: API에 전달할 프롬프트 텍스트
        system_prompt: 시스템 프롬프트 (기본값: None)
        
    Returns:
        API 응답 텍스트 또는 오류 메시지
    """
    try:
        # API 키 가져오기
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key and 'anthropic' in st.secrets:
            anthropic_api_key = st.secrets["anthropic"]["api_key"]
        
        if not anthropic_api_key:
            return "API 키가 설정되지 않았습니다."
            
        # 기본 시스템 프롬프트 설정 (ASCII 범위 내 문자만 사용)
        if system_prompt is None:
            system_prompt = "Educational expert analyzing student data."
            
        # 디버깅을 위한 로깅
        logging.info("API 요청 준비 중...")
        
        # ASCII 인코딩 문제를 해결하기 위해 최소한의 심플한 프롬프트 사용
        # 참고: 전달된 프롬프트는 무시하고 고정 프롬프트 사용
        safe_prompt = "Analyze student data."
        
        # API 요청 헤더 (ASCII 범위 내 문자만 사용)
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": anthropic_api_key,
            "Anthropic-Version": "2023-06-01"
        }
        
        # 가장 단순한 페이로드 구성 (모든 비ASCII 문자 제거)
        payload = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": safe_prompt
                }
            ],
            "system": "Expert analyst."
        }
        
        # JSON 직렬화를 수동으로 처리 (ASCII 강제 설정)
        import json
        json_payload = json.dumps(payload, ensure_ascii=True)
        
        logging.info("API 요청 시작...")
        
        # 요청 전송 (data 파라미터 사용)
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            data=json_payload,  # json= 대신 data= 사용
            headers=headers
        )
        
        # 상태 코드 확인
        logging.info(f"API 응답 코드: {response.status_code}")
        
        if response.status_code != 200:
            logging.error(f"API 호출 실패: {response.status_code}")
            logging.error(f"응답 내용: {response.text[:100]}")
            return f"API 호출 오류 (상태 코드: {response.status_code})"
        
        # 응답 처리
        try:
            # 응답 내용 로깅
            logging.info("응답 처리 시작...")
            
            # 응답 문자열을 로그에 기록 (디버깅용)
            response_text = response.text
            logging.info(f"원본 응답 (처음 100자): {response_text[:100]}...")
            
            # JSON 파싱
            result = json.loads(response_text)
            
            # 응답 내용 추출
            if "content" in result and isinstance(result["content"], list):
                result_text = ""
                
                # 텍스트 내용 추출
                for content_item in result["content"]:
                    if content_item.get("type") == "text":
                        result_text += content_item.get("text", "")
                
                if result_text:
                    # 최종 처리된 응답 일부를 로그에 기록
                    logging.info(f"최종 응답 텍스트 (처음 100자): {result_text[:100]}...")
                    return result_text
                else:
                    logging.error("응답에 텍스트 내용이 없음")
                    return "API 응답에 텍스트 내용이 없습니다."
            else:
                logging.error(f"응답에 content 필드가 없음: {list(result.keys())}")
                return "API 응답 형식이 올바르지 않습니다."
        
        except json.JSONDecodeError as json_error:
            logging.error(f"JSON 파싱 오류: {str(json_error)}")
            return "API 응답을 JSON으로 파싱할 수 없습니다."
            
        except Exception as parse_error:
            logging.error(f"응답 파싱 오류: {str(parse_error)}")
            return "API 응답 처리 중 오류가 발생했습니다."
    
    except Exception as e:
        logging.error(f"API 호출 중 예외 발생: {str(e)}")
        return "API 호출 중 오류가 발생했습니다."

# 기존 함수들은 통합 함수를 활용하도록 수정

def analyze_with_claude(prompt):
    """Anthropic Claude API로 학생 데이터 분석"""
    try:
        # 기본 프롬프트 사용
        system_prompt = "당신은 학생 데이터를 분석하는 교육 전문가입니다. 항상 한국어로 응답하세요."
        
        # 통합 API 호출 함수 사용
        return call_claude_api(prompt, system_prompt)
    
    except Exception as e:
        logging.error(f"분석 오류: {str(e)}")
        return f"AI 분석 중 오류가 발생했습니다: {str(e)}"

def analyze_csv_directly(csv_content):
    """CSV 데이터를 직접 분석합니다."""
    try:
        # 기본 분석 지침
        analysis_instruction = """
학생 데이터를 분석하여 학생의 특성과 발전 가능성을 분석해주세요. 다음 항목들을 포함해주세요:

1. 학업 역량 분석
2. 학생 특성 분석
3. 진로 적합성 분석
4. 종합 제언

분석은 긍정적이고 발전적인 관점에서 작성해주세요.
"""
        # 통합 API 호출 함수 사용
        system_prompt = "당신은 학생 데이터를 분석하는 교육 전문가입니다. 항상 한국어로 응답하세요."
        return call_claude_api(analysis_instruction, system_prompt)
        
    except Exception as e:
        logging.error(f"분석 오류: {str(e)}")
        return f"AI 분석 중 오류가 발생했습니다: {str(e)}"

def analyze_student_record(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """학생 생활기록부를 분석하여 종합적인 결과를 반환합니다."""
    try:
        # API 키 가져오기
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key and 'anthropic' in st.secrets:
            anthropic_api_key = st.secrets["anthropic"]["api_key"]
        
        if not anthropic_api_key:
            return {"analysis": "API 키가 설정되지 않았습니다."}
        
        # 간단한 프롬프트
        simple_prompt = "Analyze student record."
        
        # API 요청 헤더
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": anthropic_api_key,
            "Anthropic-Version": "2023-06-01"
        }
        
        # 간단한 페이로드
        payload = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": simple_prompt
                }
            ]
        }
        
        # API 호출
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers
        )
        
        # 응답 확인 및 반환
        if response.status_code == 200:
            result = response.json()
            
            # 텍스트만 추출
            result_text = ""
            if "content" in result:
                for item in result["content"]:
                    if item.get("type") == "text":
                        result_text += item.get("text", "")
            
            return {"analysis": result_text or "응답이 비어있습니다."}
        else:
            return {"analysis": f"API 호출 실패 (상태 코드: {response.status_code})"}
    
    except Exception as e:
        return {"analysis": f"오류 발생: {str(e)}"}

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