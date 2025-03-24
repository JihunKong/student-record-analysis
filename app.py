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
from utils import preprocess_csv, extract_student_info, create_downloadable_report, plot_timeline, create_radar_chart, process_csv_file, create_analysis_prompt, analyze_grades, create_grade_comparison_chart, create_average_comparison_chart, create_credit_weighted_chart
from analyzer import analyze_student_record, analyze_with_gemini

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
st.write("CSV 형식의 학생 생활기록부 파일을 업로드하면 AI가 분석 결과를 제공합니다.")

uploaded_file = st.file_uploader("생활기록부 CSV 파일을 업로드하세요", type=['csv'])

if uploaded_file is not None:
    try:
        # 파일 처리
        csv_content, original_data = process_csv_file(uploaded_file)
        
        # 분석 프롬프트 생성
        prompt = create_analysis_prompt(csv_content)
        
        # Gemini API를 통한 분석
        with st.spinner('AI가 생활기록부를 분석하고 있습니다...'):
            analysis_result = analyze_with_gemini(prompt)
        
        if "error" in analysis_result:
            st.error(f"분석 중 오류가 발생했습니다: {analysis_result['error']}")
            st.stop()
            
        # 분석 결과 표시
        st.header("📊 분석 결과")
        
        # 학생 프로필
        st.subheader("👤 학생 프로필")
        profile = analysis_result["학생_프로필"]
        st.write(f"**기본 정보:** {profile['기본_정보_요약']}")
        st.write(f"**진로 희망:** {profile['진로희망']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**강점:**")
            for strength in profile["강점"]:
                st.write(f"- {strength}")
        with col2:
            st.write("**개선이 필요한 부분:**")
            for weakness in profile["약점"]:
                st.write(f"- {weakness}")
        
        # 교과 성취도
        st.subheader("📚 교과 성취도")
        for subject, analysis in analysis_result["교과_성취도"]["과목별_분석"].items():
            st.write(f"**{subject}:** {analysis}")
            
        # 성적 데이터 시각화 코드
        if "시각화_코드" in analysis_result:
            st.subheader("📊 성적 데이터 시각화")
            
            # React 컴포넌트 코드
            with st.expander("React 컴포넌트 코드 보기"):
                st.code(analysis_result["시각화_코드"]["react_component"], language="jsx")
            
            # CSS 스타일 코드
            with st.expander("CSS 스타일 코드 보기"):
                st.code(analysis_result["시각화_코드"]["css_styles"], language="css")
            
            # 사용 방법 설명
            with st.expander("사용 방법 및 필요한 라이브러리"):
                st.markdown(analysis_result["시각화_코드"]["설명"])
            
            st.info("""
            위 코드를 React 프로젝트에서 사용하려면:
            1. 필요한 라이브러리를 설치하세요
            2. React 컴포넌트 파일을 생성하고 코드를 복사하세요
            3. CSS 파일을 생성하고 스타일 코드를 복사하세요
            4. 컴포넌트를 import하여 사용하세요
            """)
        
        # 활동 내역
        st.subheader("🎯 활동 내역")
        activities = analysis_result["활동_내역"]
        for activity_type, content in activities.items():
            st.write(f"**{activity_type}:** {content}")
        
        # 진로 적합성
        st.subheader("🎯 진로 적합성")
        career = analysis_result["진로_적합성"]
        st.write(f"**현재 진로희망과의 일치도:** {career['일치도']}")
        st.write("**추천 진로 옵션:**")
        for option in career["적합_진로_옵션"]:
            st.write(f"- {option}")
        
        # 학업 발전 전략
        st.subheader("📈 학업 발전 전략")
        strategy = analysis_result["학업_발전_전략"]
        st.write("**교과목별 학습 전략:**")
        for subject, strat in strategy["교과목_분석"].items():
            st.write(f"- **{subject}:** {strat}")
        
        st.write("**권장 학습 전략:**")
        for strat in strategy["권장_전략"]:
            st.write(f"- {strat}")
        
        # 진로 로드맵
        st.subheader("🗺 진로 로드맵")
        roadmap = analysis_result["진로_로드맵"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**단기 목표:**")
            for goal in roadmap["단기_목표"]:
                st.write(f"- {goal}")
        with col2:
            st.write("**중기 목표:**")
            for goal in roadmap["중기_목표"]:
                st.write(f"- {goal}")
        with col3:
            st.write("**장기 목표:**")
            for goal in roadmap["장기_목표"]:
                st.write(f"- {goal}")
        
        st.write("**추천 활동:**")
        activities = roadmap["추천_활동"]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**교과 활동:**")
            for activity in activities["교과_활동"]:
                st.write(f"- {activity}")
        with col2:
            st.write("**비교과 활동:**")
            for activity in activities["비교과_활동"]:
                st.write(f"- {activity}")
            
        # 원본 데이터 토글
        with st.expander("원본 데이터 보기"):
            st.dataframe(original_data)
        
        # 학생 정보 추출
        student_info = extract_student_info(original_data)
        
        # 성적 분석
        grade_data = pd.DataFrame([
            {
                '학 기': row['학 기'],
                '과 목': row['과 목'],
                '학점수': row['학점수'],
                '석차등급': row['석차등급']
            }
            for _, row in original_data.iterrows()
            if pd.notna(row.get('석차등급', None))
        ])
        
        if not grade_data.empty:
            st.header("성적 분석")
            
            # 성적 분석 수행
            grade_analysis = analyze_grades(grade_data)
            
            # 탭 생성
            tab1, tab2, tab3 = st.tabs(["학기별 과목 비교", "평균 비교", "가중치 비교"])
            
            with tab1:
                st.subheader("학기별 과목 등급 비교")
                fig1 = create_grade_comparison_chart(grade_analysis)
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                st.subheader("평균 등급 비교")
                fig2 = create_average_comparison_chart(grade_analysis)
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                st.subheader("과목별 등급과 가중등급 비교")
                fig3 = create_credit_weighted_chart(grade_analysis)
                st.plotly_chart(fig3, use_container_width=True)
            
            # 평균 정보 표시
            st.subheader("평균 등급 정보")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 1학기")
                st.write(f"가중평균: {grade_analysis['1학기']['가중_평균']:.2f}")
                st.write(f"단순평균: {grade_analysis['1학기']['단순_평균']:.2f}")
            
            with col2:
                st.markdown("#### 2학기")
                st.write(f"가중평균: {grade_analysis['2학기']['가중_평균']:.2f}")
                st.write(f"단순평균: {grade_analysis['2학기']['단순_평균']:.2f}")
            
            st.markdown("#### 전체 평균")
            st.write(f"주요과목(국영수사과) 평균: {grade_analysis['전체']['주요과목_평균']:.2f}")
            st.write(f"전체과목 평균: {grade_analysis['전체']['전체과목_평균']:.2f}")
        
        # Gemini 분석 수행
        analysis_results = analyze_student_record(student_info, original_data)
        
        if "error" not in analysis_results:
            st.header("분석 결과")
            
            # 학생 프로필
            st.subheader("학생 프로필")
            st.write(analysis_results["학생_프로필"]["기본_정보"])
            
            # 강점
            st.markdown("#### 강점")
            for strength in analysis_results["학생_프로필"]["강점"]:
                st.markdown(f"- {strength}")
            
            # 진로 적합성
            st.subheader("진로 적합성 분석")
            st.write(analysis_results["진로_적합성"]["분석_결과"])
            
            # 추천 진로
            st.markdown("#### 추천 진로")
            for option in analysis_results["진로_적합성"]["추천_진로"]:
                st.markdown(f"- {option}")
            
            # 학업 발전 전략
            st.subheader("학업 발전 전략")
            st.write(analysis_results["학업_발전_전략"]["분석_결과"])
            
            # 개선 전략
            st.markdown("#### 개선 전략")
            for strategy in analysis_results["학업_발전_전략"]["개선_전략"]:
                st.markdown(f"- {strategy}")
            
            # 진로 로드맵
            st.subheader("진로 로드맵")
            
            # 단기 목표
            st.markdown("#### 단기 목표")
            for goal in analysis_results["진로_로드맵"]["단기_목표"]:
                st.markdown(f"- {goal}")
            
            # 중기 목표
            st.markdown("#### 중기 목표")
            for goal in analysis_results["진로_로드맵"]["중기_목표"]:
                st.markdown(f"- {goal}")
            
            # 장기 목표
            st.markdown("#### 장기 목표")
            for goal in analysis_results["진로_로드맵"]["장기_목표"]:
                st.markdown(f"- {goal}")
        
        else:
            st.error(f"분석 중 오류가 발생했습니다: {analysis_results['error']}")
        
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

# 앱 실행
if __name__ == "__main__":
    # 추가 설정 등을 여기에 작성할 수 있습니다.
    pass 