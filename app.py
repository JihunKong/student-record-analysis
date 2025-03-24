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
        # Claude API로 분석 요청을 위한 프롬프트 생성
        from analyzer import analyze_with_claude
        
        # 프롬프트 생성
        prompt = create_analysis_prompt(student_info)
        
        # Claude API를 통한 분석 요청
        try:
            analysis_result = analyze_with_claude(student_info)
            return {"analysis": analysis_result}
        except Exception as e:
            print(f"Claude API 호출 중 오류: {str(e)}")
            # 오류 발생 시 프롬프트만 반환
            return {"analysis": prompt}
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        return {"error": str(e)}

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
            
            # 파일을 임시 파일로 저장하고 경로 얻기
            temp_file_path = "temp_uploaded_file.csv"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 파일 처리 및 학생 정보 추출
            student_data = process_csv_file(temp_file_path)
            
            if not student_data:
                st.error("파일 처리에 실패했습니다. 파일 형식을 확인해주세요.")
                st.stop()
            
            # 디버깅 정보
            st.write("파일 처리 결과")
            
            if 'special_notes' in student_data and 'subjects' in student_data['special_notes']:
                st.write(f"- 세특 데이터: {len(student_data['special_notes']['subjects'])} 과목")
            else:
                st.write("- 세특 데이터: 없음")
                
            if 'academic_records' in student_data:
                semester1_count = len(student_data['academic_records'].get('semester1', {}).get('grades', {}))
                semester2_count = len(student_data['academic_records'].get('semester2', {}).get('grades', {}))
                st.write(f"- 성적 데이터: 1학기 {semester1_count}과목, 2학기 {semester2_count}과목")
            else:
                st.write("- 성적 데이터: 없음")
            
            student_info = student_data  # 더 이상 별도 추출이 필요 없음
            
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
                    
                    # 과목별 세특 표시
                    if 'subjects' in student_info['special_notes'] and student_info['special_notes']['subjects']:
                        subjects_df = pd.DataFrame({
                            '과목': list(student_info['special_notes']['subjects'].keys()),
                            '내용': list(student_info['special_notes']['subjects'].values())
                        })
                        st.dataframe(subjects_df, use_container_width=True)
                    else:
                        st.info("과목별 세특 데이터가 없습니다.")
                    
                    # 활동별 세특 표시
                    if 'activities' in student_info['special_notes'] and student_info['special_notes']['activities']:
                        st.markdown("### 활동별 특기사항")
                        activities_df = pd.DataFrame({
                            '활동': list(student_info['special_notes']['activities'].keys()),
                            '내용': list(student_info['special_notes']['activities'].values())
                        })
                        st.dataframe(activities_df, use_container_width=True)
                    else:
                        st.info("활동별 세특 데이터가 없습니다.")
                    
                    # 성적 데이터 표시
                    st.markdown("### 성적 데이터")
                    display_grade_data(student_info)
                
                with tab2:
                    st.markdown('<h2 class="section-header">📈 성적 분석</h2>', unsafe_allow_html=True)
                    
                    # 과목별 비교 차트 - 학점수와 등급만 사용
                    main_subjects = ['국어', '영어', '수학', '사회', '과학']
                    all_subjects = ['국어', '영어', '수학', '사회', '과학', '한국사', '정보']
                    
                    # 안전하게 데이터 접근
                    semester1_grades = []
                    semester2_grades = []
                    semester1_credits = []
                    semester2_credits = []
                    
                    # 메인 과목 데이터 (5과목)
                    for subject in main_subjects:
                        # 1학기 데이터
                        if 'semester1' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester1']:
                            if subject in student_info['academic_records']['semester1']['grades']:
                                semester1_grades.append(student_info['academic_records']['semester1']['grades'][subject]['rank'])
                                if 'credit' in student_info['academic_records']['semester1']['grades'][subject]:
                                    semester1_credits.append(student_info['academic_records']['semester1']['grades'][subject]['credit'])
                                else:
                                    semester1_credits.append(1)  # 기본 학점수 1
                            else:
                                semester1_grades.append(0)
                                semester1_credits.append(0)
                        else:
                            semester1_grades.append(0)
                            semester1_credits.append(0)
                        
                        # 2학기 데이터
                        if 'semester2' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester2']:
                            if subject in student_info['academic_records']['semester2']['grades']:
                                semester2_grades.append(student_info['academic_records']['semester2']['grades'][subject]['rank'])
                                if 'credit' in student_info['academic_records']['semester2']['grades'][subject]:
                                    semester2_credits.append(student_info['academic_records']['semester2']['grades'][subject]['credit'])
                                else:
                                    semester2_credits.append(1)  # 기본 학점수 1
                            else:
                                semester2_grades.append(0)
                                semester2_credits.append(0)
                        else:
                            semester2_grades.append(0)
                            semester2_credits.append(0)
                    
                    # 0인 값은 제외하고 표시할 과목과 데이터 준비
                    valid_subjects = []
                    valid_sem1_grades = []
                    valid_sem2_grades = []
                    valid_sem1_credits = []
                    valid_sem2_credits = []
                    
                    for i, subject in enumerate(main_subjects):
                        if semester1_grades[i] > 0 or semester2_grades[i] > 0:
                            valid_subjects.append(subject)
                            valid_sem1_grades.append(semester1_grades[i])
                            valid_sem2_grades.append(semester2_grades[i])
                            valid_sem1_credits.append(semester1_credits[i])
                            valid_sem2_credits.append(semester2_credits[i])
                    
                    # 차트 생성 - 등급을 그래프 높이로 변환 (1등급=9칸, 9등급=1칸)
                    if valid_subjects:
                        # 1. 등급 차트
                        st.subheader("과목별 등급 비교")
                        fig = go.Figure()
                        
                        # 등급을 높이로 변환 (1등급=9, 9등급=1)
                        sem1_heights = [10 - g if g > 0 else 0 for g in valid_sem1_grades]
                        sem2_heights = [10 - g if g > 0 else 0 for g in valid_sem2_grades]
                        
                        # 1학기 데이터가 있는 경우만 추가
                        if any(grade > 0 for grade in valid_sem1_grades):
                            fig.add_trace(go.Bar(
                                name='1학기', 
                                x=valid_subjects, 
                                y=sem1_heights,
                                text=[f"{g}등급" if g > 0 else "N/A" for g in valid_sem1_grades],
                                textposition='auto'
                            ))
                        
                        # 2학기 데이터가 있는 경우만 추가
                        if any(grade > 0 for grade in valid_sem2_grades):
                            fig.add_trace(go.Bar(
                                name='2학기', 
                                x=valid_subjects, 
                                y=sem2_heights,
                                text=[f"{g}등급" if g > 0 else "N/A" for g in valid_sem2_grades],
                                textposition='auto'
                            ))
                        
                        # 레이아웃 설정 - 등급을 높이로 표현 (높을수록 좋은 등급)
                        fig.update_layout(
                            title="과목별 등급 비교 (막대가 높을수록 좋은 등급)",
                            barmode='group',
                            yaxis=dict(
                                title="성취도",
                                tickmode='array',
                                tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                ticktext=['9등급', '8등급', '7등급', '6등급', '5등급', '4등급', '3등급', '2등급', '1등급'],
                                range=[0, 9.5]  # 0부터 9.5까지 표시
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
                        
                        # 2. 학점 가중치를 반영한 성적 차트
                        st.subheader("학점 가중치를 반영한 성적")
                        
                        # 등급과 학점을 곱한 가중 점수 계산 (학점이 높고 등급이 좋을수록(낮을수록) 점수가 높음)
                        sem1_weighted = [(10 - g) * c if g > 0 and c > 0 else 0 for g, c in zip(valid_sem1_grades, valid_sem1_credits)]
                        sem2_weighted = [(10 - g) * c if g > 0 and c > 0 else 0 for g, c in zip(valid_sem2_grades, valid_sem2_credits)]
                        
                        fig_weighted = go.Figure()
                        
                        # 1학기 데이터
                        if any(score > 0 for score in sem1_weighted):
                            fig_weighted.add_trace(go.Bar(
                                name='1학기', 
                                x=valid_subjects, 
                                y=sem1_weighted,
                                text=[f"{s:.1f}" if s > 0 else "N/A" for s in sem1_weighted],
                                textposition='auto'
                            ))
                        
                        # 2학기 데이터
                        if any(score > 0 for score in sem2_weighted):
                            fig_weighted.add_trace(go.Bar(
                                name='2학기', 
                                x=valid_subjects, 
                                y=sem2_weighted,
                                text=[f"{s:.1f}" if s > 0 else "N/A" for s in sem2_weighted],
                                textposition='auto'
                            ))
                        
                        fig_weighted.update_layout(
                            title="과목별 학점 가중치 반영 성적 (막대가 높을수록 좋은 점수)",
                            barmode='group',
                            yaxis=dict(
                                title="가중 점수",
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
                        
                        st.plotly_chart(fig_weighted, use_container_width=True)
                    else:
                        st.info("과목별 등급 데이터가 없습니다.")
                    
                    # 학기별 평균 등급 계산 - 5과목, 7과목 모두 계산
                    # 5과목 평균 (국어, 영어, 수학, 사회, 과학)
                    semester1_avg_5 = 0
                    semester2_avg_5 = 0
                    
                    sem1_grades_5 = [g for g in valid_sem1_grades if g > 0]
                    sem2_grades_5 = [g for g in valid_sem2_grades if g > 0]
                    
                    if sem1_grades_5:
                        semester1_avg_5 = sum(sem1_grades_5) / len(sem1_grades_5)
                    
                    if sem2_grades_5:
                        semester2_avg_5 = sum(sem2_grades_5) / len(sem2_grades_5)
                    
                    # 7과목 평균 (국어, 영어, 수학, 사회, 과학, 한국사, 정보)
                    semester1_avg_7 = 0
                    semester2_avg_7 = 0
                    
                    # 7과목 데이터 수집
                    all_sem1_grades = []
                    all_sem2_grades = []
                    
                    for subject in all_subjects:
                        # 1학기 등급
                        if 'semester1' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester1']:
                            if subject in student_info['academic_records']['semester1']['grades']:
                                grade = student_info['academic_records']['semester1']['grades'][subject]['rank']
                                if grade > 0:
                                    all_sem1_grades.append(grade)
                        
                        # 2학기 등급
                        if 'semester2' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester2']:
                            if subject in student_info['academic_records']['semester2']['grades']:
                                grade = student_info['academic_records']['semester2']['grades'][subject]['rank']
                                if grade > 0:
                                    all_sem2_grades.append(grade)
                    
                    if all_sem1_grades:
                        semester1_avg_7 = sum(all_sem1_grades) / len(all_sem1_grades)
                    
                    if all_sem2_grades:
                        semester2_avg_7 = sum(all_sem2_grades) / len(all_sem2_grades)
                    
                    # 학기별 평균 표시
                    st.subheader("학기별 평균 등급")
                    
                    # 5과목 평균 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if semester1_avg_5 > 0:
                            st.metric("1학기 평균 등급 (5과목)", f"{semester1_avg_5:.2f}")
                        else:
                            st.metric("1학기 평균 등급 (5과목)", "N/A")
                    
                    with col2:
                        if semester2_avg_5 > 0:
                            delta = semester1_avg_5 - semester2_avg_5 if semester1_avg_5 > 0 else None
                            delta_color = "inverse" if delta and delta > 0 else "normal"  # 등급은 낮을수록 좋으므로 색상 반전
                            st.metric("2학기 평균 등급 (5과목)", f"{semester2_avg_5:.2f}", delta=f"{delta:.2f}" if delta else None, delta_color=delta_color)
                        else:
                            st.metric("2학기 평균 등급 (5과목)", "N/A")
                    
                    # 7과목 평균 표시
                    st.markdown("##### 참고: 7과목 평균 (국어, 영어, 수학, 사회, 과학, 한국사, 정보)")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if semester1_avg_7 > 0:
                            st.metric("1학기 평균 등급 (7과목)", f"{semester1_avg_7:.2f}")
                        else:
                            st.metric("1학기 평균 등급 (7과목)", "N/A")
                    
                    with col2:
                        if semester2_avg_7 > 0:
                            delta = semester1_avg_7 - semester2_avg_7 if semester1_avg_7 > 0 else None
                            delta_color = "inverse" if delta and delta > 0 else "normal"
                            st.metric("2학기 평균 등급 (7과목)", f"{semester2_avg_7:.2f}", delta=f"{delta:.2f}" if delta else None, delta_color=delta_color)
                        else:
                            st.metric("2학기 평균 등급 (7과목)", "N/A")
                
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