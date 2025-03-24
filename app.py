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

# 사이드바 설정
with st.sidebar:
    st.title("학생 생활기록부 분석기")
    st.markdown("---")
    st.markdown("### 사용 방법")
    st.markdown("1. CSV 파일을 업로드합니다.")
    st.markdown("2. 분석할 학생을 선택합니다.")
    st.markdown("3. 분석 결과를 확인합니다.")
    st.markdown("---")
    st.markdown("© 2025 생활기록부 분석기 Made by 공지훈")
    
    # 메뉴 선택
    menu = st.radio(
        "메뉴 선택",
        ["파일 업로드", "분석 결과", "도움말"]
    )

# 세션 상태 초기화
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'student_info' not in st.session_state:
    st.session_state.student_info = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# 파일 업로드 페이지
if menu == "파일 업로드":
    st.title("학생 생활기록부 분석 시스템")
    st.write("CSV 형식의 생활기록부 데이터를 업로드하여 분석을 시작하세요.")
    
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = preprocess_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.uploaded_file = uploaded_file
            
            # 데이터 미리보기
            st.subheader("데이터 미리보기")
            st.dataframe(df.head())
            
            # 학생 정보 추출
            student_info = extract_student_info(df)
            st.session_state.student_info = student_info
            
            # 학생 정보 표시
            st.subheader("학생 정보")
            
            # 교과별 세부능력 및 특기사항 표시
            st.subheader("교과별 세부능력 및 특기사항")
            academic_performance = student_info.get('academic_performance', {})
            if academic_performance:
                for subject, content in academic_performance.items():
                    with st.expander(subject):
                        st.write(content)
            
            # 활동 내역 표시
            st.subheader("활동 내역")
            activities = student_info.get('activities', {})
            if activities:
                for activity_type, content in activities.items():
                    with st.expander(activity_type):
                        st.write(content)
            
            # 학기별 성적 표시
            st.subheader("학기별 성적")
            grades = student_info.get('grades', [])
            if grades:
                grades_df = pd.DataFrame(grades)
                st.dataframe(grades_df)
            
            # 분석 시작 버튼
            if st.button("분석 시작"):
                with st.spinner("분석 중..."):
                    try:
                        analysis_results = analyze_student_record(student_info)
                        st.session_state.analysis_results = analysis_results
                        st.success("분석이 완료되었습니다!")
                        
                        # 분석 결과 바로 표시
                        st.header("📊 생활기록부 분석 결과")
                        
                        # 학생 프로필
                        st.subheader("👤 학생 프로필 요약")
                        if "학생_프로필" in analysis_results:
                            profile = analysis_results["학생_프로필"]
                            
                            # 프로필 정보 표시
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                if "기본_정보_요약" in profile:
                                    st.markdown("### 기본 정보")
                                    st.write(profile["기본_정보_요약"])
                                
                                if "강점" in profile:
                                    st.markdown("### 강점")
                                    if isinstance(profile["강점"], list):
                                        for item in profile["강점"]:
                                            st.write(f"- {item}")
                                    else:
                                        st.write(profile["강점"])
                                
                                if "약점" in profile:
                                    st.markdown("### 개선 필요 사항")
                                    if isinstance(profile["약점"], list):
                                        for item in profile["약점"]:
                                            st.write(f"- {item}")
                                    else:
                                        st.write(profile["약점"])
                            
                            with col2:
                                if "능력_점수" in profile:
                                    st.markdown("### 학생 능력 프로필")
                                    radar_fig = create_radar_chart(profile["능력_점수"])
                                    st.pyplot(radar_fig)
                        
                        # 진로 적합성
                        st.subheader("🎯 진로 적합성 분석")
                        if "진로_적합성" in analysis_results:
                            career = analysis_results["진로_적합성"]
                            
                            if "일치도" in career:
                                st.markdown("### 희망 진로와 현재 역량 일치도")
                                st.write(career["일치도"])
                            
                            if "적합_진로_옵션" in career:
                                st.markdown("### 추천 진로 옵션")
                                options = career["적합_진로_옵션"]
                                
                                if isinstance(options, list):
                                    for idx, option in enumerate(options):
                                        with st.expander(f"추천 진로 {idx+1}: {option.get('진로명', '')}"):
                                            st.write(f"**적합 이유:** {option.get('적합_이유', '')}")
                                            st.write(f"**보완 필요 사항:** {option.get('보완_필요사항', '')}")
                                else:
                                    st.write(options)
                        
                        # 학업 발전 전략
                        st.subheader("📚 학업 발전 전략")
                        if "학업_발전_전략" in analysis_results:
                            academic = analysis_results["학업_발전_전략"]
                            
                            if "교과목_분석" in academic:
                                st.markdown("### 교과목별 분석")
                                subjects = academic["교과목_분석"]
                                
                                if isinstance(subjects, dict):
                                    for subject, analysis in subjects.items():
                                        with st.expander(f"교과목: {subject}"):
                                            st.write(f"**현재 성취도:** {analysis.get('현재_성취도', '')}")
                                            st.write(f"**발전 가능성:** {analysis.get('발전_가능성', '')}")
                                            st.write(f"**권장 전략:** {analysis.get('권장_전략', '')}")
                                else:
                                    st.write(subjects)
                        
                        # 진로 로드맵
                        st.subheader("🗺️ 진로 로드맵")
                        if "진로_로드맵" in analysis_results:
                            roadmap = analysis_results["진로_로드맵"]
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                if "단기_목표" in roadmap:
                                    st.markdown("### 단기 목표 (고등학교)")
                                    goals = roadmap["단기_목표"]
                                    
                                    if isinstance(goals, list):
                                        for goal in goals:
                                            st.write(f"- {goal}")
                                    else:
                                        st.write(goals)
                                
                                if "중기_목표" in roadmap:
                                    st.markdown("### 중기 목표 (대학)")
                                    goals = roadmap["중기_목표"]
                                    
                                    if isinstance(goals, list):
                                        for goal in goals:
                                            st.write(f"- {goal}")
                                    else:
                                        st.write(goals)
                                
                                if "장기_목표" in roadmap:
                                    st.markdown("### 장기 목표 (졸업 후)")
                                    goals = roadmap["장기_목표"]
                                    
                                    if isinstance(goals, list):
                                        for goal in goals:
                                            st.write(f"- {goal}")
                                    else:
                                        st.write(goals)
                            
                            with col2:
                                if "추천_활동" in roadmap:
                                    st.markdown("### 추천 활동")
                                    activities = roadmap["추천_활동"]
                                    
                                    if isinstance(activities, dict):
                                        for stage, acts in activities.items():
                                            st.markdown(f"**{stage}**")
                                            if isinstance(acts, list):
                                                for act in acts:
                                                    st.write(f"- {act}")
                                            else:
                                                st.write(acts)
                                    else:
                                        st.write(activities)
                        
                        # 전체 보고서 다운로드 버튼
                        st.markdown("---")
                        st.subheader("📥 분석 보고서 다운로드")
                        
                        if st.button("전체 보고서 다운로드"):
                            report_content = create_downloadable_report(analysis_results, "생활기록부_분석보고서.md")
                            st.markdown(report_content, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                        st.stop()
        
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

# 도움말 페이지
elif menu == "도움말":
    st.header("❓ 도움말")
    
    st.subheader("프로그램 사용 방법")
    st.write(
        """
        1. **파일 업로드**: 생활기록부 CSV 파일을 업로드합니다.
        2. **분석 시작**: 업로드한 파일을 Google Gemini API를 통해 분석합니다.
        3. **분석 결과**: 분석 결과를 확인하고 필요한 정보를 확인합니다.
        4. **보고서 다운로드**: 분석 결과를 마크다운 형식으로 다운로드할 수 있습니다.
        """
    )
    
    st.subheader("CSV 파일 형식")
    st.write(
        """
        분석을 위한 CSV 파일은 다음과 같은 정보를 포함해야 합니다:
        
        - 학생의 기본 정보 (학년, 반, 번호, 이름 등)
        - 학업 성취도 데이터
        - 활동 내역 및 진로 희망사항
        - 담임 선생님 의견 및 평가
        
        필요한 경우 아래 샘플 CSV 파일을 참조하여 데이터를 준비해주세요.
        """
    )
    
    st.subheader("자주 묻는 질문")
    with st.expander("Q: 어떤 형식의 파일을 업로드해야 하나요?"):
        st.write("A: CSV 형식의 파일만 업로드할 수 있습니다. 엑셀 파일의 경우 CSV 형식으로 저장 후 업로드해주세요.")
    
    with st.expander("Q: 분석에 얼마나 시간이 소요되나요?"):
        st.write("A: 데이터의 양과 현재 서버 상황에 따라 다르지만, 일반적으로 1-3분 정도 소요됩니다.")
    
    with st.expander("Q: 개인정보는 안전하게 처리되나요?"):
        st.write("A: 네, 모든 데이터는 사용자 환경 내에서만 처리되며, 별도의 외부 저장소에 저장되지 않습니다. Google Gemini API 호출 시에는 HTTPS를 통해 보안이 유지됩니다.")

# 앱 실행
if __name__ == "__main__":
    # 추가 설정 등을 여기에 작성할 수 있습니다.
    pass 