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
import requests

# .env 파일 로드
load_dotenv()

# OpenAI API 키 가져오기 함수
def get_openai_api_key():
    """OpenAI API 키를 환경 변수 또는 Streamlit secrets에서 가져옵니다."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and 'openai' in st.secrets:
        api_key = st.secrets["openai"]["api_key"]
    return api_key

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

def analyze_csv_directly(csv_content):
    """CSV 데이터를 GPT로 직접 분석합니다."""
    try:
        # OpenAI API 키 가져오기
        openai_api_key = get_openai_api_key()
        
        if not openai_api_key:
            return "OpenAI API 키가 설정되지 않았습니다. 환경 변수나 Streamlit secrets에 OPENAI_API_KEY를 설정하세요."
            
        # CSV 내용 처리 - 안전을 위해 크기 제한
        csv_sample = csv_content
        max_length = 10000  # 최대 토큰 수 고려
        
        if len(csv_content) > max_length:
            # 너무 크면 앞부분만 사용 - 헤더와 주요 데이터 포함
            lines = csv_content.split('\n')
            if len(lines) > 20:  # 충분한 행이 있는 경우
                header = lines[0]
                data_sample = lines[1:20]  # 19개 데이터 행 + 헤더
                csv_sample = header + '\n' + '\n'.join(data_sample)
            else:
                csv_sample = csv_content[:max_length]
            
            csv_sample += "\n(내용이 너무 길어 일부만 표시됨)"
        
        # 프롬프트 구성 - 명확한 지시와 함께 CSV 데이터 전체 전달
        prompt = f"""
이 CSV 파일은 한 학생의 생활기록부 데이터를 포함하고 있습니다. 
파일을 철저히 분석하여 다음 항목에 대한 상세한 분석 결과를 제공해주세요:

CSV 데이터:
```
{csv_sample}
```

다음 항목을 포함하여 철저히 분석해주세요:
1. 학업 역량 분석
   - 각 과목별 성취도와 특징 분석
   - 학기별 성적 변화 추이
   - 강점 과목과 보완이 필요한 과목

2. 학생 특성 분석
   - 세부능력특기사항에서 확인되는, 학생의 성격 및 행동 특성
   - 두드러진 역량과 관심사
   - 대인관계 및 리더십 특성

3. 진로 적합성 분석
   - 성적과 특기사항을 바탕으로 한 적합한 진로 방향
   - 진로 실현을 위한 준비 상태
   - 발전 가능성과 보완이 필요한 부분

4. 종합 제언
   - 학생의 주요 강점과 특징을 종합적으로 분석
   - 진로 목표 달성을 위한 구체적인 조언 5가지 이상
   - 학업 및 비교과 활동에서 집중해야 할 부분 제안

분석은 객관적 데이터를 기반으로 하되, 긍정적이고 발전적인 관점에서 작성해주세요.
진로희망을 가장 중요한 요소로 고려하여, 모든 분석과 제언이 학생의 진로희망을 중심으로 연결되도록 해주세요.
"""
        
        # API 요청 헤더
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        # API 요청 페이로드
        payload = {
            "model": "o3-mini",
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 학생 생활기록부를 분석하는 교육 전문가입니다. CSV 파일 내용을 철저히 분석하여 학생의 강점, 약점, 진로 적합성 등을 종합적으로 평가해주세요. 항상 한국어로 응답하며, 최대한 구체적이고 개인화된 분석을 제공합니다."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_completion_tokens": 4000
        }
        
        # OpenAI API 호출
        logging.info("CSV 분석 API 호출 시작")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers
        )
        
        # 응답 확인 및 반환
        if response.status_code == 200:
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "응답 내용을 찾을 수 없습니다."
        else:
            error_msg = f"API 호출 실패 (상태 코드: {response.status_code})"
            if hasattr(response, 'text'):
                error_msg += f": {response.text[:200]}"
            logging.error(error_msg)
            return error_msg
    
    except Exception as e:
        import logging
        import traceback
        error_msg = f"CSV 분석 중 오류 발생: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return error_msg

def analyze_student_record(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """학생 생활기록부를 GPT를 사용하여 분석합니다."""
    try:
        # OpenAI API 키 가져오기
        openai_api_key = get_openai_api_key()
        
        if not openai_api_key:
            return {"analysis": "OpenAI API 키가 설정되지 않았습니다. 환경 변수나 Streamlit secrets에 OPENAI_API_KEY를 설정하세요."}
        
        # 학생 데이터 전처리
        processed_data = {
            "학업 기록": student_data.get("academic_records", {}),
            "특별 활동": student_data.get("special_notes", {}),
            "진로 희망": student_data.get("career_aspiration", "미정")
        }
        
        # 프롬프트 구성
        prompt = f"""
다음 학생 데이터를 분석하여 종합적인 평가를 제공해주세요:

학생 데이터: {json.dumps(processed_data, ensure_ascii=False)[:1000]}

다음 항목을 포함한 분석을 제공해주세요:
1. 학업 역량 분석 (전반적인 학업 수준, 과목별 특징, 학습 태도)
2. 학생 특성 분석 (성격, 관심사, 대인관계 및 리더십)
3. 진로 적합성 분석 (희망 진로와의 연관성, 준비 상태, 발전 가능성)
4. 종합 제언 (구체적인 발전 방향 5가지 이상)

학생의 강점을 살리고 약점을 보완할 수 있는 구체적이고 실행 가능한 조언을 포함해주세요.
"""
        
        # API 요청 헤더
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        # API 요청 페이로드
        payload = {
            "model": "o3-mini",
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 학생 데이터를 분석하는 교육 전문가입니다. 객관적이면서도 긍정적인 관점으로 분석하고, 항상 한국어로 응답하세요."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_completion_tokens": 4000
        }
        
        # OpenAI API 호출
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers
        )
        
        # 응답 확인 및 반환
        if response.status_code == 200:
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return {"analysis": result["choices"][0]["message"]["content"]}
            else:
                return {"analysis": "응답 내용을 찾을 수 없습니다."}
        else:
            return {"analysis": f"API 호출 실패 (상태 코드: {response.status_code}): {response.text[:200]}"}
    
    except Exception as e:
        return {"analysis": f"분석 중 오류 발생: {str(e)}"}

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