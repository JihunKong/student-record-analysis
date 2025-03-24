import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai

# 로컬 모듈 임포트
from utils import preprocess_csv, extract_student_info, create_downloadable_report, plot_timeline, create_radar_chart
from analyzer import analyze_student_record

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="학생 생활기록부 분석 시스템",
    page_icon="📊",
    layout="wide"
)

# Gemini API 설정
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    st.error("GitHub 환경변수에 GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-002')

# 앱 타이틀
st.title("📚 생활기록부 분석 및 시각화 자동화 프로그램")
st.markdown("---")

# 사이드바
with st.sidebar:
    st.title("학생 생활기록부 분석기")
    st.write("""
    이 앱은 학생의 생활기록부 데이터를 분석하여 
    학생의 특성과 진로 적합성을 파악하는 도구입니다.
    """)
    st.markdown("---")
    st.markdown("© 2025 생활기록부 분석기 Made by 공지훈")

# 세션 상태 초기화
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'student_info' not in st.session_state:
    st.session_state.student_info = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# 파일 업로드 섹션
st.title("학생 생활기록부 분석 시스템")
st.write("CSV 형식의 생활기록부 데이터를 업로드하여 분석을 시작하세요.")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])

if uploaded_file is not None:
    try:
        # 파일 처리 시작을 알림
        with st.spinner('파일을 처리하는 중입니다...'):
            df = preprocess_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.uploaded_file = uploaded_file
            
            # 성공 메시지 표시
            st.success('파일이 성공적으로 업로드되었습니다!')
            
            # 데이터 미리보기
            st.subheader("데이터 미리보기")
            
            # 메인 데이터와 성적 데이터 분리
            main_data, grade_data = df
            
            # 메인 데이터 미리보기
            st.write("### 교과 및 활동 데이터")
            if not main_data.empty:
                st.dataframe(main_data.head())
            else:
                st.warning("교과 및 활동 데이터를 찾을 수 없습니다.")
            
            # 성적 데이터 미리보기
            st.write("### 성적 데이터")
            if not grade_data.empty:
                st.dataframe(grade_data.head())
            else:
                st.warning("성적 데이터를 찾을 수 없습니다.")
            
            # 학생 정보 추출
            try:
                student_info = extract_student_info(df)
                st.session_state.student_info = student_info
                
                # 학생 정보 표시
                st.subheader("학생 정보")
                
                # 교과별 세부능력 및 특기사항 표시
                st.write("### 교과별 세부능력 및 특기사항")
                academic_performance = student_info.get('academic_performance', {})
                if academic_performance:
                    for subject, content in academic_performance.items():
                        with st.expander(f"📚 {subject}"):
                            st.write(content)
                else:
                    st.info("교과별 세부능력 및 특기사항이 없습니다.")
                
                # 활동 내역 표시
                st.write("### 활동 내역")
                activities = student_info.get('activities', {})
                if activities:
                    for activity_type, content in activities.items():
                        with st.expander(f"🎯 {activity_type}"):
                            st.write(content)
                else:
                    st.info("활동 내역이 없습니다.")
                
                # 진로 희망 표시
                if student_info.get('career_aspiration'):
                    st.write("### 진로 희망")
                    st.info(f"🎯 {student_info['career_aspiration']}")
                
                # 분석 시작 버튼
                if st.button("분석 시작", key="start_analysis"):
                    with st.spinner("학생 데이터를 분석하는 중입니다..."):
                        try:
                            analysis_results = analyze_student_record(student_info)
                            st.session_state.analysis_results = analysis_results
                            st.success("분석이 완료되었습니다!")
                            
                            # 분석 결과 표시 섹션으로 자동 스크롤
                            st.experimental_set_query_params(section='analysis_results')
                            
                        except Exception as e:
                            st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                            st.info("다시 시도해주세요. 문제가 지속되면 관리자에게 문의하세요.")
                
            except Exception as e:
                st.error(f"학생 정보 추출 중 오류가 발생했습니다: {str(e)}")
                st.info("파일 형식이 올바른지 확인해주세요.")
                
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        st.info("""
        다음 사항을 확인해주세요:
        1. 파일이 올바른 CSV 형식인가요?
        2. 한글이 포함된 경우 인코딩이 올바른가요?
        3. 필수 컬럼이 모두 포함되어 있나요?
        """)

# 앱 실행
if __name__ == "__main__":
    # 추가 설정 등을 여기에 작성할 수 있습니다.
    pass 