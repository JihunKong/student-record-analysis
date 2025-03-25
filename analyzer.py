import os
import logging
import requests
import json
import traceback
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# OpenAI API 키 가져오는 함수
def get_openai_api_key() -> Optional[str]:
    """환경변수나 Streamlit secrets에서 OpenAI API 키를 가져옵니다."""
    # 환경 변수에서 확인 - 여러 가능한 환경 변수명 확인
    for key_name in ["OPENAI_API_KEY", "OPENAI_KEY", "openai_api_key"]:
        api_key = os.getenv(key_name)
        if api_key and api_key.strip():
            logger.info(f"환경 변수에서 API 키를 찾았습니다: {key_name}")
            return api_key
    
    # Streamlit secrets에서 확인 - 여러 가능한 키 경로 확인
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # 직접 접근 시도
            if "OPENAI_API_KEY" in st.secrets:
                logger.info("Streamlit secrets에서 API 키를 찾았습니다: OPENAI_API_KEY")
                return st.secrets["OPENAI_API_KEY"]
                
            # openai 섹션 내부 확인
            if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
                logger.info("Streamlit secrets에서 API 키를 찾았습니다: openai.api_key")
                return st.secrets["openai"]["api_key"]
    except Exception as e:
        logger.error(f"Streamlit secrets에서 API 키 로드 중 오류: {str(e)}")
    
    logger.warning("환경 변수나 Streamlit secrets에서 API 키를 찾을 수 없습니다.")
    return None

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

응답 형식에 대한 중요 지침:
1. 반드시 마크다운 형식으로 응답해주세요.
2. 각 섹션은 ## 수준의 헤더로 구분해주세요 (예: ## 1. 학업 역량 분석)
3. 소제목은 ### 수준의 헤더로 표시해주세요 (예: ### 과목별 성취도 및 특징 분석)
4. 리스트는 적절히 마크다운 형식 (* 또는 숫자)으로 표시해주세요
5. 구분선 대신 섹션 헤더를 사용하여 내용을 구분해주세요
6. 표 형식의 데이터는 마크다운 표로 작성해주세요
7. 강조가 필요한 부분은 **굵은 글씨**나 *기울임 글씨*로 표시해주세요
8. 문단 사이에는 빈 줄을 넣어 가독성을 높여주세요
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
                    "content": "당신은 학생 생활기록부를 분석하는 교육 전문가입니다. CSV 파일 내용을 철저히 분석하여 학생의 강점, 약점, 진로 적합성 등을 종합적으로 평가해주세요. 항상 한국어로 응답하며, 최대한 구체적이고 개인화된 분석을 제공합니다. 응답은 반드시 마크다운 형식으로 작성하여 가독성을 높이고, 헤더, 리스트, 강조 등을 적절히 활용하세요."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_completion_tokens": 4000
        }
        
        # OpenAI API 호출
        logger.info("CSV 분석 API 호출 시작")
        
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
            logger.error(error_msg)
            return error_msg
    
    except Exception as e:
        error_msg = f"CSV 분석 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg

def analyze_student_record(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """학생 생활기록부를 분석하여 종합적인 결과를 반환합니다."""
    try:
        # OpenAI API 키 가져오기
        openai_api_key = get_openai_api_key()
        
        if not openai_api_key:
            return {"analysis": "OpenAI API 키가 설정되지 않았습니다. 환경 변수나 Streamlit secrets에 OPENAI_API_KEY를 설정하세요.", "error": "API 키 없음"}
        
        # 분석 프롬프트 작성
        from app import create_analysis_prompt
        prompt = create_analysis_prompt(student_data)
        
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
                    "content": "당신은 학생 생활기록부를 분석하는 교육 전문가입니다. 제공된 학업 데이터를 종합적으로 분석하여 학생의 특성과 발전 가능성에 대해 객관적이고 발전적인 관점에서 분석해주세요. 항상 한국어로 응답하세요."
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
                analysis = result["choices"][0]["message"]["content"]
                return {"analysis": analysis}
            else:
                return {"analysis": "응답 내용을 찾을 수 없습니다.", "error": "응답 형식 오류"}
        else:
            error_msg = f"API 호출 실패 (상태 코드: {response.status_code}): {response.text[:200] if hasattr(response, 'text') else ''}"
            logger.error(error_msg)
            return {"analysis": "API 호출에 실패했습니다. 잠시 후 다시 시도해주세요.", "error": error_msg}
    
    except Exception as e:
        error_msg = f"학생 기록 분석 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"analysis": "분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", "error": str(e)} 
