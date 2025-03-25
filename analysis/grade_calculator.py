import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def print_detailed_grades(grades_data):
    print("\n=== 상세 성적 내역 ===")
    for semester in ['1학기', '2학기']:
        print(f"\n{semester}:")
        semester_data = grades_data[semester]
        for subject, data in semester_data.items():
            if subject != '정보':  # 정보 과목 제외
                grade = data['등급']
                credits = data['이수단위']
                print(f"{subject}: {grade}등급 × {credits}학점 = {grade * credits}")

def calculate_average_grade(grades_data):
    total_credit_grade = 0
    total_credits = 0
    
    for semester in ['1학기', '2학기']:
        semester_data = grades_data[semester]
        for subject, data in semester_data.items():
            if subject != '정보':  # 정보 과목 제외
                grade = data['등급']
                credits = data['이수단위']
                total_credit_grade += (grade * credits)
                total_credits += credits
    
    average_grade = total_credit_grade / total_credits
    return round(average_grade, 2)

def create_grade_graph(grades_data):
    subjects = ['국어', '수학', '영어', '한국사', '통합사회', '통합과학']
    semester1_grades = []
    semester2_grades = []
    
    for subject in subjects:
        semester1_grades.append(grades_data['1학기'].get(subject, {}).get('등급', 0))
        semester2_grades.append(grades_data['2학기'].get(subject, {}).get('등급', 0))
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(subjects))
    width = 0.35
    
    plt.bar(x - width/2, semester1_grades, width, label='1학기')
    plt.bar(x + width/2, semester2_grades, width, label='2학기')
    
    plt.ylabel('등급')
    plt.title('학기별 과목 등급 비교')
    plt.xticks(x, subjects, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def analyze_grades(grades_data):
    # 상세 성적 내역 출력
    print_detailed_grades(grades_data)
    
    # 평균 등급 계산
    total_credit_grade = 0
    total_credits = 0
    
    for semester in ['1학기', '2학기']:
        semester_data = grades_data[semester]
        for subject, data in semester_data.items():
            if subject != '정보':  # 정보 과목 제외
                grade = data['등급']
                credits = data['이수단위']
                total_credit_grade += (grade * credits)
                total_credits += credits
    
    average_grade = total_credit_grade / total_credits
    print(f"\n=== 평균 등급 계산 ===")
    print(f"등급 × 이수단위의 합: {total_credit_grade}")
    print(f"이수단위의 합: {total_credits}")
    print(f"평균 등급 = {total_credit_grade} ÷ {total_credits} = {round(average_grade, 2)}")
    
    # 그래프 생성
    plt = create_grade_graph(grades_data)
    plt.show() 