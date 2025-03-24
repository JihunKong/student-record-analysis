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
from utils import process_csv_file, extract_student_info
from analyzer import analyze_with_claude

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="학생 생활기록부 분석 시스템",
    page_icon="📊",
    layout="wide"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #1E88E5;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1.5rem 0;
        color: #333;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #555;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .analysis-card {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
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
st.markdown('<h1 class="main-header">📚 생활기록부 분석 및 시각화 자동화 프로그램</h1>', unsafe_allow_html=True)
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
            
            # 과목별 1,2학기 비교 차트
            all_subjects = set()
            semester_data = {'1': {}, '2': {}}
            
            for grade in student_info['grades']:
                if grade['grade'] != '0':
                    semester = grade['semester']
                    subject = grade['subject']
                    all_subjects.add(subject)
                    semester_data[semester][subject] = {
                        'grade': float(grade['grade']),
                        'credit': float(grade['credit'])
                    }
            
            # 과목별 비교 차트
            subject_comparison = go.Figure()
            
            # 정렬된 과목 리스트
            sorted_subjects = sorted(list(all_subjects))
            
            # 1학기 데이터
            first_semester_grades = [semester_data['1'].get(subject, {}).get('grade', None) for subject in sorted_subjects]
            
            # 2학기 데이터
            second_semester_grades = [semester_data['2'].get(subject, {}).get('grade', None) for subject in sorted_subjects]
            
            # 1학기 막대 그래프
            subject_comparison.add_trace(go.Bar(
                name='1학기',
                x=sorted_subjects,
                y=first_semester_grades,
                text=first_semester_grades,
                textposition='auto',
            ))
            
            # 2학기 막대 그래프
            subject_comparison.add_trace(go.Bar(
                name='2학기',
                x=sorted_subjects,
                y=second_semester_grades,
                text=second_semester_grades,
                textposition='auto',
            ))
            
            # 레이아웃 업데이트
            subject_comparison.update_layout(
                title='과목별 1,2학기 등급 비교',
                xaxis_title='과목',
                yaxis_title='등급',
                yaxis=dict(
                    range=[9.5, 0.5],  # 1등급이 위로 가도록 y축 반전
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                ),
                barmode='group',
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            st.plotly_chart(subject_comparison, use_container_width=True)
            
            # 평균 정보 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 class="subsection-header">📊 전체 과목 평균</h3>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(label="1학기 단순평균", value=f"{student_info['first_semester_average']:.2f}")
                st.metric(label="2학기 단순평균", value=f"{student_info['second_semester_average']:.2f}")
                st.metric(label="전체 단순평균", value=f"{student_info['total_average']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(label="1학기 가중평균", value=f"{student_info['first_semester_weighted_average']:.2f}")
                st.metric(label="2학기 가중평균", value=f"{student_info['second_semester_weighted_average']:.2f}")
                st.metric(label="전체 가중평균", value=f"{student_info['total_weighted_average']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<h3 class="subsection-header">📚 주요 과목 평균 (국영수사과/한국사/정보)</h3>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(label="1학기 단순평균", value=f"{student_info['first_semester_main_average']:.2f}")
                st.metric(label="2학기 단순평균", value=f"{student_info['second_semester_main_average']:.2f}")
                st.metric(label="전체 단순평균", value=f"{student_info['total_main_average']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(label="1학기 가중평균", value=f"{student_info['first_semester_main_weighted_average']:.2f}")
                st.metric(label="2학기 가중평균", value=f"{student_info['second_semester_main_weighted_average']:.2f}")
                st.metric(label="전체 가중평균", value=f"{student_info['total_main_weighted_average']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 평균 계산 과정 표시
            st.markdown('<h3 class="subsection-header">📝 평균 계산 과정</h3>', unsafe_allow_html=True)
            
            # 1학기 계산 과정
            st.markdown("### 1학기")
            first_semester_grades = [g for g in student_info['grades'] if g['semester'] == '1' and g['grade'] != '0']
            if first_semester_grades:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write("**전체 과목 단순평균 계산:**")
                grade_sum = sum(float(g['grade']) for g in first_semester_grades)
                grade_count = len(first_semester_grades)
                st.write(f"- 등급 합계: {grade_sum}")
                st.write(f"- 과목 수: {grade_count}")
                st.write(f"- 계산: {grade_sum} ÷ {grade_count} = {grade_sum/grade_count:.2f}")
                
                st.write("\n**전체 과목 가중평균 계산:**")
                weighted_sum = sum(float(g['grade']) * float(g['credit']) for g in first_semester_grades)
                total_credits = sum(float(g['credit']) for g in first_semester_grades)
                st.write(f"- 가중합계: {weighted_sum}")
                st.write(f"- 총학점: {total_credits}")
                st.write(f"- 계산: {weighted_sum} ÷ {total_credits} = {weighted_sum/total_credits:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 주요 과목 계산
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                main_grades = [g for g in first_semester_grades if g['is_main']]
                if main_grades:
                    st.write("**주요 과목 단순평균 계산:**")
                    main_grade_sum = sum(float(g['grade']) for g in main_grades)
                    main_grade_count = len(main_grades)
                    st.write(f"- 등급 합계: {main_grade_sum}")
                    st.write(f"- 과목 수: {main_grade_count}")
                    st.write(f"- 계산: {main_grade_sum} ÷ {main_grade_count} = {main_grade_sum/main_grade_count:.2f}")
                    
                    st.write("\n**주요 과목 가중평균 계산:**")
                    main_weighted_sum = sum(float(g['grade']) * float(g['credit']) for g in main_grades)
                    main_total_credits = sum(float(g['credit']) for g in main_grades)
                    st.write(f"- 가중합계: {main_weighted_sum}")
                    st.write(f"- 총학점: {main_total_credits}")
                    st.write(f"- 계산: {main_weighted_sum} ÷ {main_total_credits} = {main_weighted_sum/main_total_credits:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 2학기 계산 과정
            st.markdown("### 2학기")
            second_semester_grades = [g for g in student_info['grades'] if g['semester'] == '2' and g['grade'] != '0']
            if second_semester_grades:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write("**전체 과목 단순평균 계산:**")
                grade_sum = sum(float(g['grade']) for g in second_semester_grades)
                grade_count = len(second_semester_grades)
                st.write(f"- 등급 합계: {grade_sum}")
                st.write(f"- 과목 수: {grade_count}")
                st.write(f"- 계산: {grade_sum} ÷ {grade_count} = {grade_sum/grade_count:.2f}")
                
                st.write("\n**전체 과목 가중평균 계산:**")
                weighted_sum = sum(float(g['grade']) * float(g['credit']) for g in second_semester_grades)
                total_credits = sum(float(g['credit']) for g in second_semester_grades)
                st.write(f"- 가중합계: {weighted_sum}")
                st.write(f"- 총학점: {total_credits}")
                st.write(f"- 계산: {weighted_sum} ÷ {total_credits} = {weighted_sum/total_credits:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 주요 과목 계산
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                main_grades = [g for g in second_semester_grades if g['is_main']]
                if main_grades:
                    st.write("**주요 과목 단순평균 계산:**")
                    main_grade_sum = sum(float(g['grade']) for g in main_grades)
                    main_grade_count = len(main_grades)
                    st.write(f"- 등급 합계: {main_grade_sum}")
                    st.write(f"- 과목 수: {main_grade_count}")
                    st.write(f"- 계산: {main_grade_sum} ÷ {main_grade_count} = {main_grade_sum/main_grade_count:.2f}")
                    
                    st.write("\n**주요 과목 가중평균 계산:**")
                    main_weighted_sum = sum(float(g['grade']) * float(g['credit']) for g in main_grades)
                    main_total_credits = sum(float(g['credit']) for g in main_grades)
                    st.write(f"- 가중합계: {main_weighted_sum}")
                    st.write(f"- 총학점: {main_total_credits}")
                    st.write(f"- 계산: {main_weighted_sum} ÷ {main_total_credits} = {main_weighted_sum/main_total_credits:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<h2 class="section-header">📝 세부능력 및 특기사항 열람</h2>', unsafe_allow_html=True)
            
            # 세특 데이터 표시
            if student_info['academic_performance']:
                st.markdown('<h3 class="subsection-header">🎓 교과별 세부능력 및 특기사항</h3>', unsafe_allow_html=True)
                for subject, content in student_info['academic_performance'].items():
                    st.markdown(f'<div class="subject-content">', unsafe_allow_html=True)
                    st.markdown(f"**{subject}**")
                    st.write(content)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # 활동 내역 표시
            if student_info['activities']:
                st.markdown('<h3 class="subsection-header">🎯 창의적 체험활동</h3>', unsafe_allow_html=True)
                for activity_type, content in student_info['activities'].items():
                    st.markdown(f'<div class="subject-content">', unsafe_allow_html=True)
                    st.markdown(f"**{activity_type} 활동**")
                    st.write(content)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # 진로 희망 표시
            if student_info['career_aspiration']:
                st.markdown('<h3 class="subsection-header">🎯 진로 희망</h3>', unsafe_allow_html=True)
                st.markdown('<div class="subject-content">', unsafe_allow_html=True)
                st.write(student_info['career_aspiration'])
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<h2 class="section-header">🤖 AI 분석</h2>', unsafe_allow_html=True)
            
            if st.button("AI 분석 실행", use_container_width=True):
                with st.spinner("AI가 생활기록부를 분석중입니다..."):
                    try:
                        # 전체 데이터를 문자열로 변환
                        data_str = df.to_string()
                        
                        # AI 분석 실행
                        analysis_result = analyze_with_claude(data_str)
                        
                        # 분석 결과 표시
                        if isinstance(analysis_result, dict):
                            if '학생_프로필' in analysis_result:
                                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                                st.markdown('<h3 class="subsection-header">👤 학생 프로필</h3>', unsafe_allow_html=True)
                                profile = analysis_result['학생_프로필']
                                st.write("**기본 정보**")
                                st.write(profile['기본_정보'])
                                
                                st.write("**강점**")
                                for strength in profile['강점']:
                                    st.write(f"- {strength}")
                                
                                st.write("**약점**")
                                for weakness in profile['약점']:
                                    st.write(f"- {weakness}")
                                
                                st.write("**학업 패턴**")
                                st.write(profile['학업_패턴'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if '강점_분석' in analysis_result:
                                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                                st.markdown('<h3 class="subsection-header">💪 강점 분석</h3>', unsafe_allow_html=True)
                                strengths = analysis_result['강점_분석']
                                
                                st.write("**교과 영역**")
                                for strength in strengths['교과_영역']:
                                    st.write(f"- {strength}")
                                
                                st.write("**비교과 영역**")
                                for strength in strengths['비교과_영역']:
                                    st.write(f"- {strength}")
                                
                                st.write("**종합 평가**")
                                st.write(strengths['종합_평가'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if '진로_적합성' in analysis_result:
                                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                                st.markdown('<h3 class="subsection-header">🎯 진로 적합성</h3>', unsafe_allow_html=True)
                                career = analysis_result['진로_적합성']
                                
                                st.write("**분석 결과**")
                                st.write(career['분석_결과'])
                                
                                st.write("**추천 진로**")
                                for path in career['추천_진로']:
                                    st.write(f"- {path}")
                                
                                st.write("**진로 로드맵**")
                                st.write(career['진로_로드맵'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if '개선_방향' in analysis_result:
                                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                                st.markdown('<h3 class="subsection-header">📈 개선 방향</h3>', unsafe_allow_html=True)
                                improvements = analysis_result['개선_방향']
                                
                                st.write("**학업 영역**")
                                for improvement in improvements['학업_영역']:
                                    st.write(f"- {improvement}")
                                
                                st.write("**활동 영역**")
                                for activity in improvements['활동_영역']:
                                    st.write(f"- {activity}")
                                
                                st.write("**종합 제언**")
                                st.write(improvements['종합_제언'])
                                st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("AI 분석 결과가 올바른 형식이 아닙니다.")
                    
                    except Exception as e:
                        st.error(f"AI 분석 중 오류가 발생했습니다: {str(e)}")
            
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

# 앱 실행
if __name__ == "__main__":
    pass 