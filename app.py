import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 로컬 모듈 임포트
from utils import process_csv_file, extract_student_info

# .env 파일 로드
load_dotenv()

def analyze_student_record(student_info: dict) -> dict:
    """학생 생활기록부를 분석하여 종합적인 결과를 반환합니다."""
    try:
        # 영문 프롬프트만 사용하여 인코딩 문제 해결
        # analyzer.py의 analyze_student_record 직접 호출
        from analyzer import analyze_student_record as analyzer_analyze
        
        # 직접 analyzer.py의 함수 호출
        return analyzer_analyze(student_info)
        
    except Exception as e:
        import logging
        logging.error(f"분석 중 오류 발생: {str(e)}")
        return {"analysis": f"AI 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", "error": str(e)}

def create_analysis_prompt(student_info: dict) -> str:
    """학생 정보를 바탕으로 Claude에게 보낼 분석 프롬프트를 생성합니다."""
    
    # 성적 데이터 요약
    grades_summary = []
    for semester in ['semester1', 'semester2']:
        if semester in student_info['academic_records']:
            semester_data = student_info['academic_records'][semester]
            grades = semester_data.get('grades', {})
            averages = semester_data.get('average', {})
            
            semester_summary = f"{semester.replace('semester', '')}학기:\n"
            semester_summary += f"- 전체 평균 등급: {averages.get('total', 0):.1f}\n"
            semester_summary += f"- 주요과목 평균 등급: {averages.get('main_subjects', 0):.1f}\n"
            semester_summary += "- 과목별 등급:\n"
            
            for subject, grade in grades.items():
                if 'rank' in grade:
                    semester_summary += f"  * {subject}: {grade['rank']}등급\n"
            
            grades_summary.append(semester_summary)
    
    # 세특 데이터 요약
    special_notes = []
    for subject, content in student_info['special_notes']['subjects'].items():
        if content and len(content) > 10:  # 의미 있는 내용만 포함
            special_notes.append(f"[{subject}]\n{content}\n")
    
    # 활동 데이터 요약
    activities = []
    for activity_type, content in student_info['special_notes']['activities'].items():
        if content and len(content) > 10:  # 의미 있는 내용만 포함
            activities.append(f"[{activity_type}]\n{content}\n")
    
    # 진로 희망
    career = student_info.get('career_aspiration', '미정')
    
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

중요: 학생의 '진로희망'을 가장 중요한 요소로 고려하여 분석해주세요. 모든 분석과 제언은 학생의 진로희망을 중심으로 연결하고, 진로 실현을 위한 구체적인 방향성을 제시해주세요. 만약 진로희망이 '미정'인 경우, 학생의 강점과 관심사를 바탕으로 적합한 진로 방향을 제안해주세요.
"""
    return prompt

def display_grade_data(student_data):
    """성적 데이터 표시"""
    if student_data and 'academic_records' in student_data:
        academic_records = student_data['academic_records']
        col1, col2 = st.columns(2)
        
        for i, (semester, semester_data) in enumerate(academic_records.items()):
            col = col1 if i == 0 else col2
            
            with col:
                st.subheader(f"{semester.replace('semester', '')}학기 성적")
                
                if 'grades' in semester_data and semester_data['grades']:
                    # 간소화된 성적 표시 - 학점수와 등급만 표시
                    grades_data = []
                    for subject, grade_info in semester_data['grades'].items():
                        if 'rank' in grade_info:  # 등급이 있는 경우만 표시
                            grades_data.append({
                                "과목": subject,
                                "등급": grade_info['rank']
                            })
                    
                    if grades_data:
                        df = pd.DataFrame(grades_data)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.info("등급 정보가 없습니다.")
                else:
                    st.info("성적 데이터가 없습니다.")

def display_special_notes(student_data):
    """세부능력 및 특기사항 표시"""
    if student_data and 'special_notes' in student_data:
        st.subheader("세부능력 및 특기사항")
        
        # 탭 생성
        tabs = st.tabs(["과목별 특기사항", "활동별 특기사항"])
        
        # 과목별 특기사항
        with tabs[0]:
            if 'subjects' in student_data['special_notes']:
                for subject, content in student_data['special_notes']['subjects'].items():
                    if content:  # 내용이 있는 경우만 표시
                        with st.expander(f"{subject}"):
                            st.write(content)
            else:
                st.info("과목별 특기사항이 없습니다.")
        
        # 활동별 특기사항
        with tabs[1]:
            if 'activities' in student_data['special_notes']:
                for activity, content in student_data['special_notes']['activities'].items():
                    if content:  # 내용이 있는 경우만 표시
                        with st.expander(f"{activity}"):
                            st.write(content)
            else:
                st.info("활동별 특기사항이 없습니다.")
        
        # 진로 희망 표시
        if 'career_aspiration' in student_data and student_data['career_aspiration'] != "미정":
            st.subheader("진로 희망")
            st.info(student_data['career_aspiration'])

# CSV 파일 처리 함수
def process_uploaded_file(uploaded_file):
    try:
        # 파일 내용 읽기
        file_content = uploaded_file.getvalue().decode('utf-8')
        
        # 파일 처리 및 학생 정보 추출
        student_info = process_csv_file(uploaded_file)
        
        # AI 분석은 필요한 경우에만 진행 
        if "ai_analysis" not in student_info or not student_info["ai_analysis"]:
            import logging
            logging.info("CSV 파일 분석 준비 중...")
            
            try:
                # 원본 CSV 내용을 직접 AI에 전달하여 분석
                from analyzer import analyze_csv_directly
                
                # CSV 파일 원본 내용으로 AI 분석 실행 (한 번만)
                analysis_result = analyze_csv_directly(file_content)
                
                # 분석 결과 저장
                student_info["ai_analysis"] = analysis_result
                    
            except Exception as e:
                logging.error(f"AI 분석 중 오류 발생: {str(e)}")
                student_info["ai_analysis"] = f"AI 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. (오류: {str(e)})"
        
        return student_info
        
    except Exception as e:
        import logging
        logging.error(f"파일 처리 중 오류 발생: {str(e)}")
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        return None

# 메인 애플리케이션
def main():
    # 페이지 설정
    st.set_page_config(
        page_title="학생 생활기록부 분석",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS 간소화 - 필수 스타일만 유지
    st.markdown("""
    <style>
        .block-container {padding: 1rem;}
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E3A8A;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        div[data-testid="stSidebarContent"] {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 상단 여백 추가
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.title("📚 학생부 분석 시스템")
        st.write("### 파일 업로드")
        uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
        
        if uploaded_file is not None:
            st.success("파일이 성공적으로 업로드되었습니다!")
        
        st.markdown("---")
        st.markdown("© 2025 학생부 분석기 Made by 공지훈")
    
    # 메인 영역에 제목 추가
    st.markdown("<div class='main-title'>📚 학생부 분석 시스템</div>", unsafe_allow_html=True)
    
    # 탭 생성 부분
    tabs = st.tabs(["원본 데이터", "성적 분석", "세특 열람", "AI 분석"])
    tab1, tab2, tab3, tab4 = tabs
    
    # 파일이 업로드되지 않은 경우
    if not uploaded_file:
        for tab in tabs:
            with tab:
                st.info("분석을 시작하려면, 좌측 사이드바에서 CSV 파일을 업로드해주세요.")
        return
    
    # 파일 처리 시작
    try:
        # 파일이 이미 처리되었는지 확인 (파일명 기반 캐싱)
        current_file_name = uploaded_file.name
        
        if ('current_file' not in st.session_state or 
            st.session_state.current_file != current_file_name or 
            'student_info' not in st.session_state or 
            not st.session_state.student_info):
            
            # 새 파일이 업로드되었거나 처리된 적이 없는 경우에만 처리
            with st.spinner("파일을 처리 중입니다..."):
                student_info = process_uploaded_file(uploaded_file)
                # 세션에 저장
                st.session_state.student_info = student_info
                st.session_state.current_file = current_file_name
        else:
            # 이미 처리된 정보가 있으면 재사용
            student_info = st.session_state.student_info
        
        # 학생 정보가 비어있는 경우
        if not student_info:
            for tab in tabs:
                with tab:
                    st.warning("학생 정보를 충분히 추출할 수 없습니다.")
            return
        
        # 각 탭에 데이터 표시
        with tab1:
            st.header("📊 원본 데이터")
            
            # 세특 데이터 표시
            st.subheader("세부능력 및 특기사항")
            
            # 과목별 세특 표시
            if 'special_notes' in student_info and 'subjects' in student_info['special_notes'] and student_info['special_notes']['subjects']:
                subjects_df = pd.DataFrame({
                    '과목': list(student_info['special_notes']['subjects'].keys()),
                    '내용': list(student_info['special_notes']['subjects'].values())
                })
                st.dataframe(subjects_df, use_container_width=True)
            else:
                st.info("과목별 세특 데이터가 없습니다.")
            
            # 활동별 세특 표시
            if 'special_notes' in student_info and 'activities' in student_info['special_notes'] and student_info['special_notes']['activities']:
                st.subheader("활동별 특기사항")
                activities_df = pd.DataFrame({
                    '활동': list(student_info['special_notes']['activities'].keys()),
                    '내용': list(student_info['special_notes']['activities'].values())
                })
                st.dataframe(activities_df, use_container_width=True)
            else:
                st.info("활동별 세특 데이터가 없습니다.")
            
            # 성적 데이터 표시
            st.subheader("성적 데이터")
            display_grade_data(student_info)
        
        with tab2:
            st.header("📈 성적 분석")
            
            # 모든 과목 데이터 수집
            all_subjects = set()
            semester1_grades = {}
            semester2_grades = {}
            semester1_credits = {}
            semester2_credits = {}
            
            # 1학기 데이터 수집
            if 'semester1' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester1']:
                for subject, grade_info in student_info['academic_records']['semester1']['grades'].items():
                    if 'rank' in grade_info:
                        all_subjects.add(subject)
                        semester1_grades[subject] = grade_info['rank']
                        semester1_credits[subject] = grade_info.get('credit', 1)
            
            # 2학기 데이터 수집
            if 'semester2' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester2']:
                for subject, grade_info in student_info['academic_records']['semester2']['grades'].items():
                    if 'rank' in grade_info:
                        all_subjects.add(subject)
                        semester2_grades[subject] = grade_info['rank']
                        semester2_credits[subject] = grade_info.get('credit', 1)
            
            # 과목별 등급 비교 차트
            if all_subjects:
                st.subheader("과목별 등급 비교")
                fig = go.Figure()
                
                subjects = sorted(list(all_subjects))
                sem1_grades_list = [semester1_grades.get(subject, 0) for subject in subjects]
                sem2_grades_list = [semester2_grades.get(subject, 0) for subject in subjects]
                
                # 등급을 높이로 변환 (1등급=9, 9등급=1)
                sem1_heights = [10 - g if g > 0 else 0 for g in sem1_grades_list]
                sem2_heights = [10 - g if g > 0 else 0 for g in sem2_grades_list]
                
                # 1학기 데이터
                if any(grade > 0 for grade in sem1_grades_list):
                    fig.add_trace(go.Bar(
                        name='1학기', 
                        x=subjects, 
                        y=sem1_heights,
                        text=[f"{g}등급" if g > 0 else "N/A" for g in sem1_grades_list],
                        textposition='auto'
                    ))
                
                # 2학기 데이터
                if any(grade > 0 for grade in sem2_grades_list):
                    fig.add_trace(go.Bar(
                        name='2학기', 
                        x=subjects, 
                        y=sem2_heights,
                        text=[f"{g}등급" if g > 0 else "N/A" for g in sem2_grades_list],
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="과목별 등급 비교 (막대가 높을수록 좋은 등급)",
                    barmode='group',
                    yaxis=dict(
                        title="성취도",
                        tickmode='array',
                        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        ticktext=['9등급', '8등급', '7등급', '6등급', '5등급', '4등급', '3등급', '2등급', '1등급'],
                        range=[0, 9.5]
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 평균 등급 계산 (정보 제외)
                total_credit_grade = 0
                total_credits = 0
                
                for subject in subjects:
                    if subject != '정보':
                        # 1학기
                        if subject in semester1_grades:
                            total_credit_grade += semester1_grades[subject] * semester1_credits[subject]
                            total_credits += semester1_credits[subject]
                        # 2학기
                        if subject in semester2_grades:
                            total_credit_grade += semester2_grades[subject] * semester2_credits[subject]
                            total_credits += semester2_credits[subject]
                
                if total_credits > 0:
                    average_grade = total_credit_grade / total_credits
                    st.subheader("평균 등급 계산 (정보 제외)")
                    st.write(f"등급 × 이수단위의 합: {total_credit_grade}")
                    st.write(f"이수단위의 합: {total_credits}")
                    st.write(f"평균 등급 = {total_credit_grade} ÷ {total_credits} = {round(average_grade, 2)}")
            else:
                st.info("과목별 등급 데이터가 없습니다.")
        
        with tab3:
            st.header("📋 세부능력 및 특기사항")
            
            # 교과별 세특
            st.subheader("🎓 교과별 세특")
            
            display_special_notes(student_info)
        
        with tab4:
            st.header("🤖 AI 분석")
            
            if "ai_analysis" in student_info and student_info["ai_analysis"]:
                # 마크다운이 제대로 표시되도록 st.write() 사용
                st.write(student_info["ai_analysis"])
            else:
                st.info("AI 분석 결과가 없습니다. 파일을 다시 업로드하거나 나중에 다시 시도해주세요.")
    
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

# 앱 실행
if __name__ == "__main__":
    main() 