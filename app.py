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
from analyzer import analyze_with_gemini

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

# 메인 컨텐츠 영역
if uploaded_file:
    try:
        # CSV 파일 처리
        df = process_csv_file(uploaded_file)
        student_info = extract_student_info(df)
        
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["원본 데이터", "성적 분석", "세특 열람", "AI 분석"])
        
        with tab1:
            st.header("📊 원본 데이터")
            st.dataframe(df)
        
        with tab2:
            st.header("📈 성적 분석")
            
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
                barmode='group'
            )
            
            st.plotly_chart(subject_comparison)
            
            # 평균 정보 표시
            st.subheader("📊 평균 등급 정보")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="1학기 단순평균", value=f"{student_info['first_semester_average']:.2f}")
                st.metric(label="1학기 가중평균", value=f"{student_info['first_semester_weighted_average']:.2f}")
            
            with col2:
                st.metric(label="2학기 단순평균", value=f"{student_info['second_semester_average']:.2f}")
                st.metric(label="2학기 가중평균", value=f"{student_info['second_semester_weighted_average']:.2f}")
            
            with col3:
                st.metric(label="전체 단순평균", value=f"{student_info['total_average']:.2f}")
                st.metric(label="전체 가중평균", value=f"{student_info['total_weighted_average']:.2f}")
            
            # 평균 계산 과정 표시
            st.subheader("📝 평균 계산 과정")
            
            # 1학기 계산 과정
            st.write("### 1학기")
            first_semester_grades = [g for g in student_info['grades'] if g['semester'] == '1' and g['grade'] != '0']
            if first_semester_grades:
                st.write("단순평균 계산:")
                grade_sum = sum(float(g['grade']) for g in first_semester_grades)
                grade_count = len(first_semester_grades)
                st.write(f"- 등급 합계: {grade_sum}")
                st.write(f"- 과목 수: {grade_count}")
                st.write(f"- 계산: {grade_sum} ÷ {grade_count} = {grade_sum/grade_count:.2f}")
                
                st.write("\n가중평균 계산:")
                weighted_sum = sum(float(g['grade']) * float(g['credit']) for g in first_semester_grades)
                total_credits = sum(float(g['credit']) for g in first_semester_grades)
                st.write(f"- 가중합계: {weighted_sum}")
                st.write(f"- 총학점: {total_credits}")
                st.write(f"- 계산: {weighted_sum} ÷ {total_credits} = {weighted_sum/total_credits:.2f}")
            
            # 2학기 계산 과정
            st.write("### 2학기")
            second_semester_grades = [g for g in student_info['grades'] if g['semester'] == '2' and g['grade'] != '0']
            if second_semester_grades:
                st.write("단순평균 계산:")
                grade_sum = sum(float(g['grade']) for g in second_semester_grades)
                grade_count = len(second_semester_grades)
                st.write(f"- 등급 합계: {grade_sum}")
                st.write(f"- 과목 수: {grade_count}")
                st.write(f"- 계산: {grade_sum} ÷ {grade_count} = {grade_sum/grade_count:.2f}")
                
                st.write("\n가중평균 계산:")
                weighted_sum = sum(float(g['grade']) * float(g['credit']) for g in second_semester_grades)
                total_credits = sum(float(g['credit']) for g in second_semester_grades)
                st.write(f"- 가중합계: {weighted_sum}")
                st.write(f"- 총학점: {total_credits}")
                st.write(f"- 계산: {weighted_sum} ÷ {total_credits} = {weighted_sum/total_credits:.2f}")
        
        with tab3:
            st.header("📝 세부능력 및 특기사항 열람")
            
            # 세특 데이터 표시
            if student_info['academic_performance']:
                st.subheader("🎓 교과별 세부능력 및 특기사항")
                for subject, content in student_info['academic_performance'].items():
                    with st.expander(f"{subject} 세특", expanded=False):
                        st.write(content)
            
            # 활동 내역 표시
            if student_info['activities']:
                st.subheader("🎯 창의적 체험활동")
                for activity_type, content in student_info['activities'].items():
                    with st.expander(f"{activity_type} 활동", expanded=False):
                        st.write(content)
            
            # 진로 희망 표시
            if student_info['career_aspiration']:
                st.subheader("🎯 진로 희망")
                st.write(student_info['career_aspiration'])
        
        with tab4:
            st.header("🤖 AI 분석")
            
            if st.button("AI 분석 실행"):
                with st.spinner("AI가 세특을 분석중입니다..."):
                    try:
                        # 전체 데이터를 문자열로 변환
                        data_str = df.to_string()
                        
                        # AI 분석 실행
                        analysis_result = analyze_with_gemini(data_str)
                        
                        # 분석 결과 표시
                        if isinstance(analysis_result, dict):
                            if '학생_프로필' in analysis_result:
                                with st.expander("학생 프로필", expanded=True):
                                    st.write(analysis_result['학생_프로필'])
                            
                            if '강점_분석' in analysis_result:
                                with st.expander("강점 분석", expanded=True):
                                    st.write(analysis_result['강점_분석'])
                            
                            if '진로_적합성' in analysis_result:
                                with st.expander("진로 적합성", expanded=True):
                                    st.write(analysis_result['진로_적합성'])
                            
                            if '개선_방향' in analysis_result:
                                with st.expander("개선 방향", expanded=True):
                                    st.write(analysis_result['개선_방향'])
                        else:
                            st.error("AI 분석 결과가 올바른 형식이 아닙니다.")
                    
                    except Exception as e:
                        st.error(f"AI 분석 중 오류가 발생했습니다: {str(e)}")
            
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

# 앱 실행
if __name__ == "__main__":
    pass 