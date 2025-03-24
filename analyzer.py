import os
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
import logging

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

def analyze_with_claude(prompt):
    """Anthropic Claude API로 학생 데이터 분석"""
    try:
        # API 키 가져오기 (환경변수 또는 Streamlit secrets)
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key and 'anthropic' in st.secrets:
            anthropic_api_key = st.secrets["anthropic"]["api_key"]
        
        if not anthropic_api_key:
            return "API 키가 설정되지 않았습니다. .env 파일에 ANTHROPIC_API_KEY를 설정하세요."
        
        # API 호출
        client = anthropic.Anthropic(
            api_key=anthropic_api_key,
            # 인코딩 문제 해결: 명시적으로 UTF-8 헤더 추가
            default_headers={
                "Content-Type": "application/json; charset=utf-8"
            }
        )
        
        try:
            # 로깅 추가
            logging.info("Claude API 호출 시작")
            logging.info(f"프롬프트 길이: {len(prompt)}")
            
            # UTF-8 인코딩 처리
            if isinstance(prompt, str):
                # URL 인코딩 대신 직접 UTF-8 인코딩 적용
                from urllib.parse import quote
                encoded_prompt = quote(prompt)
                logging.info(f"UTF-8 인코딩 후 프롬프트 길이: {len(encoded_prompt)}")
            
            # 프롬프트를 사용하여 메시지 생성
            message_data = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 4000,
                "system": "당신은 학생 데이터를 분석하는 교육 전문가입니다. 주어진 학생 데이터를 분석하여 학생의 강점, 약점, 진로 적합성 등을 종합적으로 평가해주세요. 항상 한국어로 응답하세요.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # JSON 직렬화 시 ensure_ascii=False 적용
            message_json = json.dumps(message_data, ensure_ascii=False).encode('utf-8')
            
            # 직접 API 호출 대신 JSON을 활용한 자체 처리
            try:
                # client.messages.create 대신 직접 API 호출 구현
                import requests
                
                # Anthropic API 엔드포인트
                api_endpoint = "https://api.anthropic.com/v1/messages"
                
                # 헤더 설정
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01"
                }
                
                # API 요청
                response = requests.post(
                    api_endpoint,
                    headers=headers,
                    data=message_json
                )
                
                # 응답 처리
                if response.status_code == 200:
                    result = response.json()
                    if "content" in result:
                        result_text = ""
                        for content_item in result["content"]:
                            if content_item["type"] == "text":
                                result_text += content_item["text"]
                        return result_text
                    else:
                        return "API 응답에서 내용을 찾을 수 없습니다."
                else:
                    error_msg = f"API 호출 실패 (상태 코드: {response.status_code}): {response.text}"
                    logging.error(error_msg)
                    return f"AI 분석 중 API 오류가 발생했습니다: {error_msg}"
                
            except Exception as api_error:
                # 표준 클라이언트로 대체 시도
                message = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=4000,
                    system="당신은 학생 데이터를 분석하는 교육 전문가입니다. 주어진 학생 데이터를 분석하여 학생의 강점, 약점, 진로 적합성 등을 종합적으로 평가해주세요. 항상 한국어로 응답하세요.",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """
학생 데이터를 분석하여 학생의 특성과 발전 가능성을 분석해주세요. 다음 항목들을 포함해주세요:

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
"""
                                }
                            ]
                        }
                    ]
                )
                
                logging.info("표준 클라이언트로 대체 호출 성공")
                
                # 응답 처리
                if hasattr(message, 'content') and message.content:
                    result_text = ""
                    for content_item in message.content:
                        if content_item.type == "text":
                            result_text += content_item.text
                    return result_text
                else:
                    return "API 응답에서 내용을 찾을 수 없습니다."
                
        except Exception as api_error:
            error_msg = str(api_error)
            logging.error(f"Claude API 호출 오류: {error_msg}")
            
            # ASCII 인코딩 에러가 발생한 경우 특별 처리
            if "ascii" in error_msg and "codec" in error_msg:
                return "인코딩 오류가 발생했습니다. 분석을 완료할 수 없습니다. 관리자에게 문의하세요."
            
            return f"AI 분석 중 API 오류가 발생했습니다: {error_msg}"
            
    except Exception as e:
        logging.error(f"분석 오류: {str(e)}")
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

def analyze_csv_directly(csv_content):
    """CSV 데이터를 직접 분석합니다."""
    try:
        # API 키 가져오기 (환경변수 또는 Streamlit secrets)
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key and 'anthropic' in st.secrets:
            anthropic_api_key = st.secrets["anthropic"]["api_key"]
        
        if not anthropic_api_key:
            return "API 키가 설정되지 않았습니다. .env 파일에 ANTHROPIC_API_KEY를 설정하세요."
        
        # API 호출 준비
        try:
            # 로깅 추가
            logging.info("CSV 파일 직접 분석 중...")
            
            # 인코딩 문제 해결: 직접 API 호출 구현
            import requests
            import json
            
            # 기본 분석 지침
            analysis_instruction = """
학생 데이터를 분석하여 학생의 특성과 발전 가능성을 분석해주세요. 다음 항목들을 포함해주세요:

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

분석은 긍정적이고 발전적인 관점에서 작성해주세요.
"""
            
            # 메시지 데이터 준비
            message_data = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 4000,
                "system": "당신은 학생 데이터를 분석하는 교육 전문가입니다. 주어진 학생 데이터를 분석하여 학생의 강점, 약점, 진로 적합성 등을 종합적으로 평가해주세요. 항상 한국어로 응답하세요.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_instruction
                            }
                        ]
                    }
                ]
            }
            
            # JSON 직렬화 (ensure_ascii=False로 한글 유지)
            message_json = json.dumps(message_data, ensure_ascii=False).encode('utf-8')
            
            # API 엔드포인트
            api_endpoint = "https://api.anthropic.com/v1/messages"
            
            # 헤더 설정
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # API 요청
            response = requests.post(
                api_endpoint,
                headers=headers,
                data=message_json
            )
            
            # 응답 처리
            if response.status_code == 200:
                result = response.json()
                if "content" in result:
                    result_text = ""
                    for content_item in result["content"]:
                        if content_item["type"] == "text":
                            result_text += content_item["text"]
                    return result_text
                else:
                    return "API 응답에서 내용을 찾을 수 없습니다."
            else:
                error_msg = f"API 호출 실패 (상태 코드: {response.status_code}): {response.text}"
                logging.error(error_msg)
                return f"AI 분석 중 API 오류가 발생했습니다: {error_msg}"
            
        except Exception as api_error:
            error_msg = str(api_error)
            logging.error(f"Claude API 호출 오류: {error_msg}")
            
            # 대체 방법: 표준 클라이언트 사용
            try:
                # API 호출
                client = anthropic.Anthropic(
                    api_key=anthropic_api_key,
                    default_headers={
                        "Content-Type": "application/json; charset=utf-8"
                    }
                )
                
                message = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=4000,
                    system="당신은 학생 데이터를 분석하는 교육 전문가입니다. 주어진 학생 데이터를 분석하여 학생의 강점, 약점, 진로 적합성 등을 종합적으로 평가해주세요. 항상 한국어로 응답하세요.",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": analysis_instruction
                                }
                            ]
                        }
                    ]
                )
                
                # 응답 처리
                if hasattr(message, 'content') and message.content:
                    result_text = ""
                    for content_item in message.content:
                        if content_item.type == "text":
                            result_text += content_item.text
                    return result_text
                else:
                    return "API 응답에서 내용을 찾을 수 없습니다."
                
            except Exception as client_error:
                return f"AI 분석 중 오류가 발생했습니다: {str(client_error)}"
            
    except Exception as e:
        logging.error(f"분석 오류: {str(e)}")
        return f"AI 분석 중 오류가 발생했습니다: {str(e)}"

def analyze_student_record(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """학생 생활기록부를 분석하여 종합적인 결과를 반환합니다."""
    try:
        # API 키 가져오기 (환경변수 또는 Streamlit secrets)
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key and 'anthropic' in st.secrets:
            anthropic_api_key = st.secrets["anthropic"]["api_key"]
        
        if not anthropic_api_key:
            return {"analysis": "API 키가 설정되지 않았습니다. .env 파일에 ANTHROPIC_API_KEY를 설정하세요."}
        
        try:
            import requests
            import logging
            
            # API 엔드포인트
            api_endpoint = "https://api.anthropic.com/v1/messages"
            
            # 헤더 설정
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # 메시지 페이로드 - 극도로 단순화
            payload = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 4000,
                "messages": [
                    {
                        "role": "user", 
                        "content": "Analyze student data and provide a detailed report."
                    }
                ],
                "system": "You are an educational expert. Respond in Korean."
            }
            
            # API 요청
            response = requests.post(
                api_endpoint,
                json=payload,  # json 파라미터 사용 (자동 직렬화)
                headers=headers
            )
            
            # 응답 처리
            logging.info(f"응답 상태 코드: {response.status_code}")
            logging.info(f"응답 헤더: {response.headers}")
            
            if response.status_code == 200:
                result = response.json()
                logging.info("API 호출 성공: 응답 받음")
                
                if "content" in result:
                    result_text = ""
                    for content_item in result["content"]:
                        if content_item["type"] == "text":
                            result_text += content_item["text"]
                    
                    # 응답이 너무 짧은지 확인
                    if len(result_text) < 100:
                        logging.warning(f"API 응답이 너무 짧습니다: {result_text}")
                        return {"analysis": "API 응답이 너무 짧습니다. 다시 시도해주세요."}
                    
                    return {"analysis": result_text}
                else:
                    logging.error(f"API 응답에 content 필드가 없습니다: {result}")
                    return {"analysis": "API 응답 형식이 올바르지 않습니다. 다시 시도해주세요."}
            else:
                error_msg = f"API 호출 실패 (상태 코드: {response.status_code}): {response.text}"
                logging.error(error_msg)
                return {"analysis": f"AI 분석 중 API 오류가 발생했습니다. 상태 코드: {response.status_code}"}
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            stacktrace = traceback.format_exc()
            logging.error(f"API 호출 오류: {error_msg}")
            logging.error(f"스택 트레이스: {stacktrace}")
            
            # 더미 응답 대신 명확한 오류 메시지 반환
            return {"analysis": f"AI 분석 중 오류가 발생했습니다: {error_msg[:100]}... (오류 로그를 확인하세요)"}
        
    except Exception as e:
        import logging
        logging.error(f"학생 기록 분석 중 오류 발생: {str(e)}")
        return {"analysis": f"분석 중 오류가 발생했습니다: {str(e)[:100]}..."} 