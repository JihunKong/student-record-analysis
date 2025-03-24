import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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

# GitHub 환경변수 확인
if not os.getenv("GEMINI_API_KEY"):
    st.error("GitHub 환경변수에 GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# 앱 타이틀
st.title("📚 생활기록부 분석 및 시각화 자동화 프로그램")
st.markdown("---")

# 사이드바 설정
with st.sidebar:
    st.header("💼 메뉴")
    menu = st.radio(
        "메뉴를 선택하세요:",
        ["파일 업로드", "분석 결과", "도움말"]
    )
    
    st.markdown("---")
    st.subheader("🔍 프로그램 정보")
    st.info(
        """
        이 프로그램은 고등학교 생활기록부 CSV 데이터를 분석하여 
        학생 진로지도, 학업 전략, 학부모 상담 자료를 제공합니다.
        """
    )
    
    st.markdown("---")
    st.caption("© 2023 생활기록부 분석기")

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
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**기본 정보**")
                st.write(f"- 학년: {student_info['grade']}")
                st.write(f"- 반: {student_info['class']}")
                st.write(f"- 번호: {student_info['student_number']}")
                st.write(f"- 이름: {student_info['name']}")
                st.write(f"- 생년월일: {student_info['birth_date']}")
                st.write(f"- 성별: {student_info['gender']}")
            
            with col2:
                st.write("**진로 정보**")
                st.write(f"- 진로희망: {student_info['career_aspiration']}")
                st.write(f"- 희망직업: {student_info['desired_profession']}")
                st.write(f"- 희망대학: {student_info['desired_university']}")
                st.write(f"- 희망학과: {student_info['desired_major']}")
            
            # 교과별 성취도 표시
            st.subheader("교과별 성취도")
            academic_performance = student_info['academic_performance']
            if academic_performance:
                performance_df = pd.DataFrame(list(academic_performance.items()), columns=['과목', '성취도'])
                st.dataframe(performance_df)
            
            # 활동 내역 표시
            st.subheader("활동 내역")
            activities = student_info['activities']
            if activities:
                for activity, content in activities.items():
                    st.write(f"**{activity}**")
                    st.write(content)
            
            # 분석 시작 버튼
            if st.button("분석 시작"):
                with st.spinner("데이터를 분석하고 있습니다..."):
                    analysis_results = analyze_student_record(student_info)
                    st.session_state.analysis_results = analysis_results
                    st.success("분석이 완료되었습니다!")
                    st.experimental_rerun()
        
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

# 분석 결과 페이지
elif menu == "분석 결과":
    st.header("📊 생활기록부 분석 결과")
    
    if st.session_state.show_analysis and st.session_state.analysis_results:
        analysis_results = st.session_state.analysis_results
        
        # 탭 생성
        tabs = st.tabs(["학생 프로필", "진로 적합성", "학업 발전 전략", "학부모 상담 가이드", "진로 로드맵"])
        
        # 학생 프로필 탭
        with tabs[0]:
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
                    # 레이더 차트 표시 (예시)
                    if "능력_점수" in profile:
                        st.markdown("### 학생 능력 프로필")
                        radar_fig = create_radar_chart(profile["능력_점수"])
                        st.pyplot(radar_fig)
            else:
                st.warning("학생 프로필 분석 결과가 없습니다.")
        
        # 진로 적합성 탭
        with tabs[1]:
            st.subheader("🎯 진로 적합성 분석")
            
            if "진로_적합성" in analysis_results:
                career = analysis_results["진로_적합성"]
                
                # 진로 적합성 정보 표시
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
                
                if "권장_계획" in career:
                    st.markdown("### 진로 성취를 위한 권장 계획")
                    plans = career["권장_계획"]
                    
                    if isinstance(plans, dict):
                        for stage, plan in plans.items():
                            st.markdown(f"**{stage}**")
                            if isinstance(plan, list):
                                for item in plan:
                                    st.write(f"- {item}")
                            else:
                                st.write(plan)
                    else:
                        st.write(plans)
            else:
                st.warning("진로 적합성 분석 결과가 없습니다.")
        
        # 학업 발전 전략 탭
        with tabs[2]:
            st.subheader("📚 학업 발전 전략")
            
            if "학업_발전_전략" in analysis_results:
                academic = analysis_results["학업_발전_전략"]
                
                # 학업 발전 전략 정보 표시
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
                
                if "학습_스타일" in academic:
                    st.markdown("### 학습 스타일 및 효과적인 학습 방법")
                    st.write(academic["학습_스타일"])
                
                if "취약_과목_전략" in academic:
                    st.markdown("### 취약 과목 개선 전략")
                    strategies = academic["취약_과목_전략"]
                    
                    if isinstance(strategies, dict):
                        for subject, strategy in strategies.items():
                            st.markdown(f"**{subject}**")
                            st.write(strategy)
                    elif isinstance(strategies, list):
                        for strategy in strategies:
                            st.write(f"- {strategy}")
                    else:
                        st.write(strategies)
                
                if "학업_로드맵" in academic:
                    st.markdown("### 대학 입시를 고려한 학업 로드맵")
                    st.write(academic["학업_로드맵"])
            else:
                st.warning("학업 발전 전략 분석 결과가 없습니다.")
        
        # 학부모 상담 가이드 탭
        with tabs[3]:
            st.subheader("👨‍👩‍👧‍👦 학부모 상담 가이드")
            
            if "학부모_상담_가이드" in analysis_results:
                parent = analysis_results["학부모_상담_가이드"]
                
                # 학부모 상담 가이드 정보 표시
                if "현재_상황_평가" in parent:
                    st.markdown("### 학생의 현재 상황 평가")
                    st.write(parent["현재_상황_평가"])
                
                if "가정_지원_방법" in parent:
                    st.markdown("### 가정에서의 지원 방법")
                    methods = parent["가정_지원_방법"]
                    
                    if isinstance(methods, list):
                        for method in methods:
                            st.write(f"- {method}")
                    else:
                        st.write(methods)
                
                if "주의사항" in parent:
                    st.markdown("### 학부모 주의사항")
                    cautions = parent["주의사항"]
                    
                    if isinstance(cautions, list):
                        for caution in cautions:
                            st.write(f"- {caution}")
                    else:
                        st.write(cautions)
                
                if "소통_방법" in parent:
                    st.markdown("### 효과적인 소통 방법")
                    st.write(parent["소통_방법"])
                
                if "교육적_접근법" in parent:
                    st.markdown("### 교육적 접근법")
                    st.write(parent["교육적_접근법"])
            else:
                st.warning("학부모 상담 가이드 분석 결과가 없습니다.")
        
        # 진로 로드맵 탭
        with tabs[4]:
            st.subheader("🗺️ 진로 로드맵")
            
            if "진로_로드맵" in analysis_results:
                roadmap = analysis_results["진로_로드맵"]
                
                # 진로 로드맵 정보 표시
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
                
                # 타임라인 표시
                if "이정표" in roadmap:
                    st.markdown("### 진로 발전 타임라인")
                    milestones = roadmap["이정표"]
                    
                    if isinstance(milestones, list) and len(milestones) > 0:
                        # 데이터 형식에 따라 처리
                        timeline_events = []
                        for idx, milestone in enumerate(milestones):
                            if isinstance(milestone, dict) and "제목" in milestone and "날짜" in milestone:
                                timeline_events.append({
                                    "title": milestone["제목"],
                                    "date": milestone["날짜"]
                                })
                            elif isinstance(milestone, str):
                                # 문자열인 경우 가상의 날짜 설정
                                timeline_events.append({
                                    "title": milestone,
                                    "date": f"{datetime.now().year + idx}-01-01"
                                })
                        
                        if timeline_events:
                            timeline_fig = plot_timeline(timeline_events)
                            st.pyplot(timeline_fig)
                    else:
                        st.write(milestones)
            else:
                st.warning("진로 로드맵 분석 결과가 없습니다.")
        
        # 전체 보고서 다운로드 버튼
        st.markdown("---")
        st.subheader("📥 분석 보고서 다운로드")
        
        if st.button("전체 보고서 다운로드"):
            # 보고서 내용 생성
            report_content = "# 생활기록부 분석 보고서\n\n"
            report_content += f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # 학생 기본 정보
            if st.session_state.student_info:
                report_content += "## 학생 기본 정보\n\n"
                for key, value in st.session_state.student_info.items():
                    if key != '진로정보':
                        report_content += f"- {key}: {value}\n"
                
                if '진로정보' in st.session_state.student_info:
                    report_content += "\n### 진로 희망사항\n\n"
                    for key, value in st.session_state.student_info['진로정보'].items():
                        report_content += f"- {key}: {value}\n"
            
            # 분석 결과
            if "학생_프로필" in analysis_results:
                report_content += "\n## 학생 프로필 요약\n\n"
                profile = analysis_results["학생_프로필"]
                
                if "기본_정보_요약" in profile:
                    report_content += f"### 기본 정보\n\n{profile['기본_정보_요약']}\n\n"
                
                if "강점" in profile:
                    report_content += "### 강점\n\n"
                    if isinstance(profile["강점"], list):
                        for item in profile["강점"]:
                            report_content += f"- {item}\n"
                    else:
                        report_content += f"{profile['강점']}\n\n"
                
                if "약점" in profile:
                    report_content += "### 개선 필요 사항\n\n"
                    if isinstance(profile["약점"], list):
                        for item in profile["약점"]:
                            report_content += f"- {item}\n"
                    else:
                        report_content += f"{profile['약점']}\n\n"
            
            # 진로 적합성
            if "진로_적합성" in analysis_results:
                report_content += "\n## 진로 적합성 분석\n\n"
                career = analysis_results["진로_적합성"]
                
                if "일치도" in career:
                    report_content += f"### 희망 진로와 현재 역량 일치도\n\n{career['일치도']}\n\n"
                
                if "적합_진로_옵션" in career:
                    report_content += "### 추천 진로 옵션\n\n"
                    options = career["적합_진로_옵션"]
                    
                    if isinstance(options, list):
                        for idx, option in enumerate(options):
                            report_content += f"#### 추천 진로 {idx+1}: {option.get('진로명', '')}\n\n"
                            report_content += f"- 적합 이유: {option.get('적합_이유', '')}\n"
                            report_content += f"- 보완 필요 사항: {option.get('보완_필요사항', '')}\n\n"
                    else:
                        report_content += f"{options}\n\n"
            
            # 나머지 섹션도 유사하게 추가...
            
            # 다운로드 링크 생성
            download_link = create_downloadable_report(report_content, "생활기록부_분석보고서.md")
            st.markdown(download_link, unsafe_allow_html=True)
    
    else:
        st.info("분석 결과가 없습니다. '파일 업로드' 메뉴에서 파일을 업로드하고 분석을 진행해주세요.")

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