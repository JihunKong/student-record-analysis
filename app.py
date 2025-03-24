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

    # 사이드바에 파일 업로더 배치
    st.header("데이터 업로드")
    uploaded_file = st.file_uploader("생활기록부 CSV 파일을 업로드하세요", type=['csv'])
    
    if uploaded_file:
        st.success("파일이 성공적으로 업로드되었습니다!")

# 세션 상태 초기화
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'student_info' not in st.session_state:
    st.session_state.student_info = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# 메인 컨텐츠 영역
if uploaded_file:
    try:
        # CSV 파일 처리
        df = process_csv_file(uploaded_file)
        student_info = extract_student_info(df)
        
        # 탭 생성
        tab1, tab2 = st.tabs(["원본 데이터", "성적 분석"])
        
        with tab1:
            st.header("📊 원본 데이터")
            st.dataframe(df)
        
        with tab2:
            st.header("📈 성적 분석")
            
            # 1학기와 2학기 과목 비교
            semester_subjects = go.Figure()
            
            # 1학기 과목 및 등급
            first_semester = []
            first_grades = []
            for grade in student_info['grades']:
                if grade['semester'] == '1' and grade['grade'] != '0':
                    first_semester.append(grade['subject'])
                    first_grades.append(float(grade['grade']))
            
            # 2학기 과목 및 등급
            second_semester = []
            second_grades = []
            for grade in student_info['grades']:
                if grade['semester'] == '2' and grade['grade'] != '0':
                    second_semester.append(grade['subject'])
                    second_grades.append(float(grade['grade']))
            
            # 1학기 데이터 추가
            semester_subjects.add_trace(go.Bar(
                name='1학기',
                x=first_semester,
                y=first_grades,
                text=first_grades,
                textposition='auto',
            ))
            
            # 2학기 데이터 추가
            semester_subjects.add_trace(go.Bar(
                name='2학기',
                x=second_semester,
                y=second_grades,
                text=second_grades,
                textposition='auto',
            ))
            
            # 레이아웃 업데이트
            semester_subjects.update_layout(
                title='학기별 과목 등급 비교',
                xaxis_title='과목',
                yaxis_title='등급',
                yaxis=dict(
                    range=[9.5, 0.5],  # 1등급이 위로 가도록 y축 반전
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                ),
                barmode='group'
            )
            
            st.plotly_chart(semester_subjects)
            
            # 평균 비교 차트
            averages = go.Figure()
            
            # 평균 데이터
            avg_labels = ['1학기 평균', '2학기 평균', '전체 평균']
            avg_values = [
                student_info['first_semester_average'],
                student_info['second_semester_average'],
                student_info['total_average']
            ]
            
            averages.add_trace(go.Bar(
                x=avg_labels,
                y=avg_values,
                text=[f'{avg:.2f}' for avg in avg_values],
                textposition='auto',
            ))
            
            averages.update_layout(
                title='평균 등급 비교',
                xaxis_title='구분',
                yaxis_title='등급',
                yaxis=dict(
                    range=[9.5, 0.5],  # 1등급이 위로 가도록 y축 반전
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                )
            )
            
            st.plotly_chart(averages)
            
            # 평균 정보 표시
            st.subheader("📊 평균 등급 정보")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="1학기 평균", value=f"{student_info['first_semester_average']:.2f}")
            
            with col2:
                st.metric(label="2학기 평균", value=f"{student_info['second_semester_average']:.2f}")
            
            with col3:
                st.metric(label="전체 평균", value=f"{student_info['total_average']:.2f}")
            
            # 하드코딩된 분석 결과 예시
            st.subheader("🤖 AI 분석 결과")
            
            with st.expander("학생 프로필", expanded=True):
                st.write("""
                **기본 정보**: 이 학생은 전반적으로 안정적인 학업 성취도를 보이고 있습니다.
                
                **강점**:
                - 수학과 과학 과목에서 우수한 성적을 보임
                - 꾸준한 성적 향상 추세를 보임
                - 자기주도적 학습 능력이 돋보임
                
                **약점**:
                - 언어 영역에서 상대적으로 낮은 성취도
                - 과목간 성적 편차가 다소 큼
                
                **학업 패턴**: 이과 과목에서 강점을 보이며, 꾸준한 성장세를 유지하고 있습니다.
                """)
            
            with st.expander("진로 적합성", expanded=True):
                st.write("""
                **분석 결과**: 이과 계열 적성이 뚜렷하며, 특히 공학 계열에 적합한 성향을 보입니다.
                
                **추천 진로**:
                1. 컴퓨터공학
                2. 전자공학
                3. 기계공학
                
                **진로 로드맵**: 수학, 과학 과목의 심화학습을 통해 공학 계열 진학을 준비하는 것이 좋겠습니다.
                """)
            
            with st.expander("학업 발전 전략", expanded=True):
                st.write("""
                **분석 결과**: 현재의 강점을 살리면서 약점을 보완하는 전략이 필요합니다.
                
                **개선 전략**:
                1. 언어 영역 학습 시간 확대
                2. 과목간 균형있는 학습 계획 수립
                3. 자기주도학습 습관 강화
                """)
            
            with st.expander("학부모 상담 가이드", expanded=True):
                st.write("""
                **분석 결과**: 학생의 강점을 살리는 방향으로 진로를 설정하되, 균형잡힌 발전이 필요합니다.
                
                **상담 포인트**:
                1. 이과 계열 적성 강화 방안
                2. 언어 영역 보완 전략
                
                **지원 방안**:
                1. 과학/수학 심화 프로그램 참여 지원
                2. 독서 활동 장려
                """)
            
            with st.expander("진로 로드맵", expanded=True):
                st.write("""
                **단기 목표**:
                1. 수학, 과학 성적 현 수준 유지
                2. 언어 영역 성적 향상
                
                **중기 목표**:
                1. 이과 계열 진학 준비
                2. 관련 분야 활동 참여
                
                **장기 목표**:
                1. 공학 계열 대학 진학
                2. 관련 자격증 취득
                """)
            
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        st.info("다른 인코딩을 선택하여 다시 시도해보세요.")

# 앱 실행
if __name__ == "__main__":
    # 추가 설정 등을 여기에 작성할 수 있습니다.
    pass 