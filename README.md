# Employee Attrition Prediction App

## Overview
This application helps predict **employee attrition risk** and provides **actionable insights**. Employee turnover (also known as "employee churn") is a costly problem for companies. The true cost of replacing an employee can often be quite large due to factors like:
- Time spent interviewing and finding a replacement
- Sign-on bonuses
- Loss of productivity during onboarding

By leveraging machine learning, this app helps **identify employees at risk** and **provides recommendations** to improve retention.

## Features
- **Upload Data**: Provide your employee dataset in CSV format
- **Model Evaluation**: Review model performance metrics such as accuracy, F1-score, and AUC-ROC
- **Feature Analysis**: Understand key attrition drivers through feature importance visualization
- **Risk Categorization**: Employees are classified into **Low**, **Medium**, or **High** attrition risk groups
- **Actionable Recommendations**: Based on key factors like job satisfaction, monthly income, and work-life balance

## Getting Started Guide
1. **Upload Data**: Provide your employee dataset in CSV format
2. **Model Evaluation**: Review model performance metrics
3. **Feature Analysis**: Understand key attrition drivers
4. **Recommendations**: Get personalized action plans

## Installation
To run this project locally, follow these steps:

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/employee-attrition-app.git
cd employee-attrition-app
```

### **2️⃣ Install Dependencies**
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Application**
```bash
streamlit run app.py
```

## **Generating Random Employee Data**
This project includes a **random employee data generator** that creates a synthetic dataset of **1,000 employees** with relevant features for attrition prediction.

### **Generate the CSV File**
Run the following script to generate `random_employee_data_1000.csv`:

```python
import csv
import random

# Define the headers
headers = ["Age", "Attrition", "DailyRate", "DistanceFromHome", "EmployeeNumber", "EnvironmentSatisfaction",
    "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "OverTime", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsWithCurrManager", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely",
    "MaritalStatus_Married", "MaritalStatus_Single", "EducationField_Life Sciences", "EducationField_Marketing",
    "EducationField_Medical", "EducationField_Other", "EducationField_Technical Degree", "JobRole_Human Resources",
    "JobRole_Laboratory Technician", "JobRole_Manager", "JobRole_Manufacturing Director", "JobRole_Research Director",
    "JobRole_Research Scientist", "JobRole_Sales Executive", "JobRole_Sales Representative"]

# Function to generate random data
def generate_random_row(employee_number):
    return [
        random.randint(20, 65), random.choice([True, False]), random.randint(200, 2000), random.randint(1, 30), employee_number,
        random.randint(1, 4), random.randint(1, 4), random.randint(1, 5), random.randint(1, 4), random.randint(1000, 20000),
        random.choice([True, False]), random.randint(0, 3), random.randint(0, 40), random.randint(0, 6), random.randint(1, 4),
        random.randint(0, 40), random.randint(0, 20), random.randint(0, 20), random.choice([True, False]), random.choice([True, False]),
        random.choice([True, False]), random.choice([True, False]), random.choice([True, False]), random.choice([True, False]),
        random.choice([True, False]), random.choice([True, False]), random.choice([True, False]), random.choice([True, False]),
        random.choice([True, False]), random.choice([True, False]), random.choice([True, False]), random.choice([True, False]),
        random.choice([True, False])
    ]

data = [headers]
for i in range(1, 1001):
    data.append(generate_random_row(i))

with open("random_employee_data_1000.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV file 'random_employee_data_1000.csv' created successfully!")
```

This will create a **synthetic dataset** to test the app without needing real employee data.

---

## License
This project is licensed under the **MIT License**. You are free to modify and use it for commercial and non-commercial purposes.

## Contact
For questions or contributions, feel free to reach out!

📧 [Reach out](https://www.linkedin.com/in/joaorussofigueiredo/)

🔗 [JohnRusso](https://github.com/j0hnrusso)

