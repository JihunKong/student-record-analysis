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
import numpy as np

# 로컬 모듈 임포트
from utils import process_csv_file, extract_student_info
from analyzer import analyze_student_record, create_subject_radar_chart, create_activity_timeline

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="학생부 분석 시스템",
    page_icon="📊",
    layout="wide"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1em;
        color: #1E3D59;
    }
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
        color: #2E5077;
    }
    .subsection-header {
        font-size: 1.4em;
        font-weight: bold;
        margin-top: 0.8em;
        margin-bottom: 0.4em;
        color: #3A6095;
    }
    .info-box {
        background-color: #F5F7FA;
        padding: 1em;
        border-radius: 5px;
        border-left: 5px solid #2E5077;
        margin: 1em 0;
    }
    .metric-container {
        background-color: white;
        padding: 1em;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5em 0;
    }
    .analysis-card {
        background-color: white;
        padding: 1.5em;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 1em 0;
    }
    .subject-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Gemini API 설정
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    st.error("GitHub 환경변수에 GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-002')

# 앱 타이틀
st.markdown('<h1 class="main-header">📚 학생부 분석 시스템</h1>', unsafe_allow_html=True)
st.markdown("---")

# 사이드바
with st.sidebar:
    st.title("학생부 분석기")
    st.write("""
    이 앱은 학생의 학생부 데이터를 분석하여 
    학생의 특성과 진로 적합성을 파악하는 도구입니다.
    """)
    st.markdown("---")
    st.markdown("© 2025 학생부 분석기 Made by 공지훈")

    # 사이드바에 파일 업로더 배치
    st.header("데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    
    if uploaded_file:
        st.success("파일이 성공적으로 업로드되었습니다!")

# 메인 컨텐츠 영역
if uploaded_file:
    try:
        # CSV 파일 처리
        df = process_csv_file(uploaded_file)
        student_info = extract_student_info(df)
        
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["원본 데이터", "성적 분석", "세특 열람", "AI 분석"])
        
        with tab1:
            st.markdown('<h2 class="section-header">📊 원본 데이터</h2>', unsafe_allow_html=True)
            st.dataframe(df)
        
        with tab2:
            st.markdown('<h2 class="section-header">📈 성적 분석</h2>', unsafe_allow_html=True)
            
            # 과목별 비교 차트
            subjects = ['국어', '수학', '영어', '한국사', '사회', '과학', '정보']
            semester1_grades = [student_info['academic_records']['semester1']['grades'][subject]['rank'] for subject in subjects]
            semester2_grades = [student_info['academic_records']['semester2']['grades'][subject]['rank'] for subject in subjects]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='1학기', x=subjects, y=semester1_grades))
            fig.add_trace(go.Bar(name='2학기', x=subjects, y=semester2_grades))
            fig.update_layout(
                title='과목별 등급 비교',
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 평균 지표 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 전체 과목 평균", unsafe_allow_html=True)
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write(f"1학기 평균: {student_info['academic_records']['semester1']['average']['total']:.2f}")
                st.write(f"2학기 평균: {student_info['academic_records']['semester2']['average']['total']:.2f}")
                st.write(f"전체 평균: {student_info['academic_records']['total']['average']['total']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("#### 평균 계산 과정", unsafe_allow_html=True)
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write("1. 각 과목의 원점수 합산")
                st.write("2. 과목 수로 나누어 평균 계산")
                st.write("3. 가중치 적용 (이수단위 고려)")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 주요 과목 평균", unsafe_allow_html=True)
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write(f"1학기 주요과목 평균: {student_info['academic_records']['semester1']['average']['main_subjects']:.2f}")
                st.write(f"2학기 주요과목 평균: {student_info['academic_records']['semester2']['average']['main_subjects']:.2f}")
                st.write(f"전체 주요과목 평균: {student_info['academic_records']['total']['average']['main_subjects']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("#### 주요 과목", unsafe_allow_html=True)
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write("- 국어")
                st.write("- 수학")
                st.write("- 영어")
                st.write("- 한국사")
                st.write("- 사회")
                st.write("- 과학")
                st.write("- 정보")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<h2 class="section-header">📝 세부능력 및 특기사항 열람</h2>', unsafe_allow_html=True)
            
            # 세특 데이터 표시
            if student_info['special_notes']['subjects']:
                st.markdown('<h3 class="subsection-header">🎓 교과별 세부능력 및 특기사항</h3>', unsafe_allow_html=True)
                for subject, content in student_info['special_notes']['subjects'].items():
                    with st.expander(f"{subject} 세부특기사항"):
                        st.write(content)
            
            # 활동 내역 표시
            if student_info['special_notes']['activities']:
                st.markdown('<h3 class="subsection-header">🎯 창의적 체험활동</h3>', unsafe_allow_html=True)
                for activity_type, content in student_info['special_notes']['activities'].items():
                    with st.expander(f"{activity_type} 활동"):
                        st.write(content)
            
            # 진로 희망 표시
            if student_info['career_aspiration']:
                st.markdown('<h3 class="subsection-header">🎯 진로 희망</h3>', unsafe_allow_html=True)
                st.markdown('<div class="subject-content">', unsafe_allow_html=True)
                st.write(student_info['career_aspiration'])
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<h2 class="section-header">🤖 AI 분석</h2>', unsafe_allow_html=True)
            
            if st.button("AI 분석 실행", use_container_width=True):
                with st.spinner("AI가 학생부를 분석하고 있습니다..."):
                    try:
                        # 데이터를 문자열로 변환
                        data_str = str(student_info)
                        
                        # AI 분석 수행
                        analysis_result = analyze_student_record(student_info)
                        
                        if "error" not in analysis_result:
                            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
                            st.markdown(analysis_result["analysis"])
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error(f"AI 분석 중 오류가 발생했습니다: {analysis_result['error']}")
                    
                    except Exception as e:
                        st.error(f"AI 분석 중 오류가 발생했습니다: {str(e)}")
            
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

# 앱 실행
if __name__ == "__main__":
    pass 