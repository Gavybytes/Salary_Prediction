import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'salary_model.joblib')

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model file '{MODEL_PATH}' not found.")
        print("[!] Please run 'python \"Salary Prediction.py\"' first to train and save the model.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)

def predict_salary(age, gender, education, experience, job_title, model_data=None):
    if model_data is None:
        model_data = load_model()
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    gender_classes = model_data['gender_classes']
    education_mapping = model_data['education_mapping']
    
    # 1. Gender encoding
    gender_clean = gender.strip().capitalize()
    if gender_clean in gender_classes:
        gender_code = gender_classes.index(gender_clean)
    else:
        gender_code = 1  # Default to Male if unknown
    
    # 2. Education encoding
    edu_code = education_mapping.get(education, 1)  # Default to Bachelor's (1)
    
    # 3. Create single row DataFrame with all feature columns initialized to 0
    row = pd.DataFrame(0.0, index=[0], columns=feature_names)
    row['Age'] = float(age)
    row['Gender'] = float(gender_code)
    row['Education Level'] = float(edu_code)
    row['Years of Experience'] = float(experience)
    
    # 4. Set job title dummy
    job_clean = job_title.strip()
    if job_clean in row.columns:
        row[job_clean] = 1.0
    
    # 5. Predict
    prediction = model.predict(row)[0]
    return float(prediction)

def interactive_mode(model_data):
    print("=" * 55)
    print("        SALARY PREDICTION - INTERACTIVE CLI")
    print("=" * 55)
    
    gender_options = model_data['gender_classes']
    job_columns = [c for c in model_data['feature_names'] if c not in ['Age', 'Gender', 'Education Level', 'Years of Experience']]
    
    try:
        age_in = input("Enter Age (e.g., 30): ").strip()
        age = float(age_in) if age_in else 30.0
        
        print(f"Available Genders: {', '.join(gender_options)}")
        gender = input("Enter Gender [Male]: ").strip() or "Male"
        
        print("Education Levels: High School, Bachelor's, Master's, PhD")
        education = input("Enter Education Level [Bachelor's]: ").strip() or "Bachelor's"
        
        exp_in = input("Enter Years of Experience (e.g., 5): ").strip()
        experience = float(exp_in) if exp_in else 5.0
        
        print("\nCommon Job Titles: Software Engineer, Data Scientist, Data Analyst, Product Manager, etc.")
        job_title = input("Enter Job Title [Software Engineer]: ").strip() or "Software Engineer"
        
        salary = predict_salary(age, gender, education, experience, job_title, model_data)
        
        print("\n" + "=" * 55)
        print(f" Predicted Annual Salary: ${salary:,.2f} USD")
        print("=" * 55)
        
    except (ValueError, KeyboardInterrupt) as e:
        print("\nOperation cancelled or invalid input.")

def main():
    parser = argparse.ArgumentParser(description="Predict Salary based on Age, Gender, Education, Experience, and Job Title.")
    parser.add_argument("--age", type=float, help="Age in years (e.g., 28)")
    parser.add_argument("--gender", type=str, choices=["Male", "Female", "Other", "male", "female", "other"], help="Gender")
    parser.add_argument("--education", type=str, choices=["High School", "Bachelor's", "Master's", "PhD"], help="Education Level")
    parser.add_argument("--experience", type=float, help="Years of Experience (e.g., 4.5)")
    parser.add_argument("--job", type=str, help="Job Title (e.g. 'Software Engineer', 'Data Scientist')")
    
    args = parser.parse_args()
    model_data = load_model()
    
    if args.age is not None and args.gender and args.education and args.experience is not None and args.job:
        pred = predict_salary(args.age, args.gender, args.education, args.experience, args.job, model_data)
        print(f"\n[Result] Predicted Salary for a {int(args.age)}-year-old {args.gender} {args.job} with {args.experience} yrs exp ({args.education}):")
        print(f" >>> ${pred:,.2f} USD <<<\n")
    else:
        interactive_mode(model_data)

if __name__ == "__main__":
    main()
