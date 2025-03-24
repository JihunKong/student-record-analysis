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
        # 1학기 성적 분석
        semester1 = student_info['academic_records']['semester1']
        semester1_avg = semester1['average']['total']
        semester1_main_avg = semester1['average']['main_subjects']
        
        # 2학기 성적 분석
        semester2 = student_info['academic_records']['semester2']
        semester2_avg = semester2['average']['total']
        semester2_main_avg = semester2['average']['main_subjects']
        
        # 전체 성적 분석
        total = student_info['academic_records']['total']
        total_avg = total['average']['total']
        total_main_avg = total['average']['main_subjects']
        
        # 성적 향상도 분석
        grade_improvement = semester2_avg - semester1_avg
        main_subjects_improvement = semester2_main_avg - semester1_main_avg
        
        # 교과별 세부특기사항 분석
        subject_strengths = []
        for subject, content in student_info['special_notes']['subjects'].items():
            if "우수" in content or "탁월" in content or "뛰어난" in content:
                subject_strengths.append(subject)
        
        # 활동 분석
        activity_summary = []
        for activity_type, content in student_info['special_notes']['activities'].items():
            if content.strip():
                activity_summary.append(f"- {activity_type}: {content[:100]}...")
        
        # 진로 희망
        career = student_info.get('career_aspiration', '미정')
        
        # 분석 결과 생성
        analysis = f"""
### 1. 학업 역량 분석

#### 전반적인 학업 수준과 발전 추이
- 전체 평균: {total_avg:.2f}
- 1학기 평균: {semester1_avg:.2f} → 2학기 평균: {semester2_avg:.2f}
- 성적 향상도: {grade_improvement:+.2f}점

#### 주요 과목 분석
- 주요 과목 전체 평균: {total_main_avg:.2f}
- 1학기 주요과목 평균: {semester1_main_avg:.2f} → 2학기 주요과목 평균: {semester2_main_avg:.2f}
- 주요 과목 향상도: {main_subjects_improvement:+.2f}점

#### 학업 강점 과목
{', '.join(subject_strengths) if subject_strengths else '분석 중...'}

### 2. 비교과 활동 분석

#### 주요 활동 내역
{''.join(f"\\n{activity}" for activity in activity_summary) if activity_summary else '활동 내역이 충분하지 않습니다.'}

### 3. 진로 분석

#### 진로 희망
{career}

### 4. 종합 제언
1. {'성적이 전반적으로 향상되는 추세를 보입니다.' if grade_improvement > 0 else '성적 향상을 위한 노력이 필요합니다.'}
2. {'주요 과목에서 긍정적인 발전을 보이고 있습니다.' if main_subjects_improvement > 0 else '주요 과목 보완이 필요합니다.'}
3. {'다양한 비교과 활동에 적극적으로 참여하고 있습니다.' if len(activity_summary) >= 3 else '비교과 활동 참여를 늘리는 것이 좋겠습니다.'}
"""
        
        return {"analysis": analysis}
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        return {"error": str(e)}

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

def main():
    st.set_page_config(
        page_title="학생 생활기록부 분석",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 커스텀 CSS - 여백 줄이기
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .st-emotion-cache-16idsys {
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }
        .st-emotion-cache-13ln4jf {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

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
            st.info("파일을 처리 중입니다. 잠시만 기다려주세요...")
            
            # 파일 처리 및 학생 정보 추출
            special_notes, grades = process_csv_file(uploaded_file)
            
            if special_notes.empty and grades.empty:
                st.error("파일 처리에 실패했습니다. 파일 형식을 확인해주세요.")
                st.stop()
            
            # 디버깅 정보
            st.write("파일 처리 결과")
            st.write(f"- 세특 데이터: {len(special_notes)} 행, {len(special_notes.columns)} 열")
            st.write(f"- 성적 데이터: {len(grades)} 행, {len(grades.columns)} 열")
            
            student_info = extract_student_info(special_notes, grades)
            
            # 학생 정보가 비어있으면 예외 발생
            if not student_info or not student_info.get('special_notes', {}).get('subjects'):
                st.warning("학생 정보를 충분히 추출할 수 없습니다. 일부 기능이 제한될 수 있습니다.")
            else:
                # 탭 생성
                tab1, tab2, tab3, tab4 = st.tabs(["원본 데이터", "성적 분석", "세특 열람", "AI 분석"])
                
                with tab1:
                    st.markdown('<h2 class="section-header">📊 원본 데이터</h2>', unsafe_allow_html=True)
                    
                    # 세특 데이터 표시
                    st.markdown("### 세부능력 및 특기사항")
                    st.dataframe(special_notes)
                    
                    # 성적 데이터 표시
                    st.markdown("### 성적 데이터")
                    display_grade_data(student_info)
                
                with tab2:
                    st.markdown('<h2 class="section-header">📈 성적 분석</h2>', unsafe_allow_html=True)
                    
                    # 과목별 비교 차트 - 학점수와 등급만 사용
                    subjects = ['국어', '수학', '영어', '한국사', '사회', '과학', '정보']
                    
                    # 안전하게 데이터 접근
                    semester1_grades = []
                    semester2_grades = []
                    
                    for subject in subjects:
                        # 1학기 데이터
                        if 'semester1' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester1']:
                            if subject in student_info['academic_records']['semester1']['grades']:
                                # 등급만 사용
                                semester1_grades.append(student_info['academic_records']['semester1']['grades'][subject]['rank'])
                            else:
                                semester1_grades.append(0)
                        else:
                            semester1_grades.append(0)
                        
                        # 2학기 데이터
                        if 'semester2' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester2']:
                            if subject in student_info['academic_records']['semester2']['grades']:
                                # 등급만 사용
                                semester2_grades.append(student_info['academic_records']['semester2']['grades'][subject]['rank'])
                            else:
                                semester2_grades.append(0)
                        else:
                            semester2_grades.append(0)
                    
                    # 0인 값은 제외하고 표시할 과목과 데이터 준비
                    valid_subjects = []
                    valid_sem1_grades = []
                    valid_sem2_grades = []
                    
                    for i, subject in enumerate(subjects):
                        if semester1_grades[i] > 0 or semester2_grades[i] > 0:
                            valid_subjects.append(subject)
                            valid_sem1_grades.append(semester1_grades[i])
                            valid_sem2_grades.append(semester2_grades[i])
                    
                    # 차트 생성
                    if valid_subjects:
                        fig = go.Figure()
                        
                        # 1학기 데이터가 있는 경우만 추가
                        if any(grade > 0 for grade in valid_sem1_grades):
                            fig.add_trace(go.Bar(
                                name='1학기', 
                                x=valid_subjects, 
                                y=valid_sem1_grades,
                                text=[f"{g:.1f}" if g > 0 else "N/A" for g in valid_sem1_grades],
                                textposition='auto'
                            ))
                        
                        # 2학기 데이터가 있는 경우만 추가
                        if any(grade > 0 for grade in valid_sem2_grades):
                            fig.add_trace(go.Bar(
                                name='2학기', 
                                x=valid_subjects, 
                                y=valid_sem2_grades,
                                text=[f"{g:.1f}" if g > 0 else "N/A" for g in valid_sem2_grades],
                                textposition='auto'
                            ))
                        
                        # 레이아웃 설정 - 등급은 낮을수록 좋으므로 Y축 반전
                        fig.update_layout(
                            title="과목별 등급 비교",
                            barmode='group',
                            yaxis=dict(
                                title="등급",
                                autorange="reversed",  # 등급 축 반전 (1등급이 위쪽)
                                tickmode='linear',
                                tick0=1,
                                dtick=1,
                                range=[9.5, 0.5]  # 9등급부터 1등급까지 표시
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
                    else:
                        st.info("과목별 등급 데이터가 없습니다.")
                    
                    # 학기별 평균 등급 계산
                    semester1_avg = 0
                    semester2_avg = 0
                    
                    sem1_grades = [g for g in valid_sem1_grades if g > 0]
                    sem2_grades = [g for g in valid_sem2_grades if g > 0]
                    
                    if sem1_grades:
                        semester1_avg = sum(sem1_grades) / len(sem1_grades)
                    
                    if sem2_grades:
                        semester2_avg = sum(sem2_grades) / len(sem2_grades)
                    
                    # 학기별 평균 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if semester1_avg > 0:
                            st.metric("1학기 평균 등급", f"{semester1_avg:.2f}")
                        else:
                            st.metric("1학기 평균 등급", "N/A")
                    
                    with col2:
                        if semester2_avg > 0:
                            delta = semester1_avg - semester2_avg if semester1_avg > 0 else None
                            delta_color = "inverse" if delta and delta > 0 else "normal"  # 등급은 낮을수록 좋으므로 색상 반전
                            st.metric("2학기 평균 등급", f"{semester2_avg:.2f}", delta=f"{delta:.2f}" if delta else None, delta_color=delta_color)
                        else:
                            st.metric("2학기 평균 등급", "N/A")
                
                with tab3:
                    st.markdown('<h2 class="section-header">📋 세부능력 및 특기사항</h2>', unsafe_allow_html=True)
                    
                    # 교과별 세특
                    st.markdown('<h3 class="subsection-header">🎓 교과별 세특</h3>', unsafe_allow_html=True)
                    
                    display_special_notes(student_info)
                
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
    main() 