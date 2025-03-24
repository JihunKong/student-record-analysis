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
                st.dataframe(grades)
            
            with tab2:
                st.markdown('<h2 class="section-header">📈 성적 분석</h2>', unsafe_allow_html=True)
                
                # 과목별 비교 차트
                subjects = ['국어', '수학', '영어', '한국사', '사회', '과학', '정보']
                
                # 안전하게 데이터 접근
                semester1_grades = []
                semester2_grades = []
                
                for subject in subjects:
                    # 1학기 데이터
                    if 'semester1' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester1']:
                        if subject in student_info['academic_records']['semester1']['grades']:
                            semester1_grades.append(student_info['academic_records']['semester1']['grades'][subject]['rank'])
                        else:
                            semester1_grades.append(0)
                    else:
                        semester1_grades.append(0)
                    
                    # 2학기 데이터
                    if 'semester2' in student_info['academic_records'] and 'grades' in student_info['academic_records']['semester2']:
                        if subject in student_info['academic_records']['semester2']['grades']:
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
                        ),
                        yaxis=dict(
                            title='등급',
                            range=[9.5, 0.5],  # 1등급이 위로 가도록 y축 반전
                            tickmode='linear',
                            tick0=1,
                            dtick=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("과목별 등급 데이터가 충분하지 않습니다.")
                
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
                    st.markdown('<h3 class="subsection-header">교과별 세부능력 및 특기사항</h3>', unsafe_allow_html=True)
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