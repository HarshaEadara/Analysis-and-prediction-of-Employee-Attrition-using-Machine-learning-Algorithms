# Analysis and Prediction of Employee Attrition using Machine learning Algorithms
This repository provides a comprehensive analysis of employee attrition using IBM Employee Attrition Data. The project leverages Exploratory Data Analysis (EDA), machine learning algorithms, and interactive dashboards to identify trends, predict attrition, and assist HR teams in developing strategies to improve employee retention.

## Table of Contents
- [Overview](#overview)
- [Key Objectives](#key-objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Repository Contents](#repository-contents)
- [Results and Insights](#results-and-insights)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview
Employee attrition can significantly impact organizational performance, leading to increased costs and decreased morale. This project focuses on:

- **Analyzing attrition patterns** using EDA and visualization tools.
- **Building predictive models** with machine learning algorithms.
- **Creating interactive dashboards** in Power BI and Tableau to present actionable insights.

The dataset used in this analysis is the publicly available **IBM HR Analytics Employee Attrition & Performance dataset**, downloaded from Kaggle.

## Key Objectives

1. **Understand Attrition Trends:**
   - Perform Exploratory Data Analysis (EDA) to identify key factors influencing employee attrition.
   - Visualize patterns and relationships between employee demographics, job satisfaction, and attrition rates.
2. **Predict Employee Attrition:**
   - Build predictive models using supervised machine learning algorithms like Logistic Regression, Random Forest, and Gradient Boosting.
   - Use hyperparameter tuning (GridSearchCV) to improve model performance.
3. **Provide Actionable Insights:**
   - Identify top contributing features to employee attrition.
   - Develop interactive dashboards in Power BI and Tableau for clear and actionable visualization of key metrics.
4. **Assist Decision-Making:**
   - Deliver data-driven recommendations to HR teams to improve employee retention and optimize organizational strategies.
  
## Dataset
The dataset used in this project comes from publicly available information about **IBM HR Analytics Employee Attrition & Performance dataset** in Kaggle. It contains several columns, including:
- Employee demographics (Age, Gender, Marital Status, Education, etc.)
- Job-related attributes (Job Role, Job Satisfaction, Department, Monthly Income, etc.)
- Attrition status (Yes/No)
- Performance and satisfaction metrics (Work-Life Balance, Job Involvement, etc.)

You can find the **IBM HR Analytics Employee Attrition & Performance dataset** dataset in the `data` folder of this repository. If you prefer to download it yourself, ensure you get the dataset from Kaggle and place it in the `data` directory.

### Preprocessing Steps
- Removal of duplicate entries
- Filtering of sparse user-product interactions
- Normalization and preparation for models

## Technologies Used
The project is implemented using:
- **Programming Language:** Python
- **Libraries and Frameworks:**
   - **Pandas, NumPy:** Data manipulation and analysis.
   - **Matplotlib, Seaborn:** Data visualization.
   - **Scikit-learn:** Machine learning models and GridSearchCV for hyperparameter tuning.
- **Dashboard Tools:**
   - **Power BI:** Interactive and detailed visualizations for HR insights.
   - **Tableau:** Intuitive dashboards for demographic and attrition analysis.
- **Other Tools:**
   - **GridSearchCV:** For hyperparameter tuning.
   - **Jupyter Notebook:** For conducting analysis and building the recommendation system.

## Repository Contents
### Dashboards
1. **Power BI Dashboard:**
   - A three-page Interactive visualization of HR Analytics Employee Attrition & Performance data
   - Files:
     - `IBM_HR_Attrition_Dashboard.pbix`: PowerBI workbook containing the dashboard.
     - `IBM HR Attrition Dashboard - Overview (Power BI ).png`: Contains a preview of the Overview page.
     - `IBM HR Attrition Dashboard - Demographics (Power BI ).png`: Contains a preview of the Demographics page.
     - `IBM HR Attrition Dashboard - Attritions (Power BI ).png`: Contains a preview of the Attritions page.
     - `IBM HR Attrition Dashboard - Power BI.pdf`: Contains a preview of all three pages in one PDF file.
2. . **Tableau Dashboard:**
   - Visual analysis of HR Analytics Employee Attrition & Performance data
   - Files:
     - `IBM_HR_Attrition_Dashboard.twbx`: Tableau workbook containing the dashboard.
     - `IBM HR Attrition Dashboard - Overview.png`: Contains a preview of the Overview page.
     - `IBM HR Attrition Dashboard - Demographics.png`: Contains a preview of the Demographics page.
     - `IBM HR Attrition Dashboard - Attritions.png`: Contains a preview of the Attritions page.
     - `IBM HR Attrition Dashboard.pdf`: Contains a preview of all three pages in one PDF file.

### Prediction with Machine Learning
#### Overview:
In this section, we analyze Employee Attrition using IBM Employee Attrition Data. We perform Exploratory Data Analysis (EDA) to uncover key trends and utilize supervised machine learning algorithms, including Linear Regression, Random Forest, and Gradient Boosting. Additionally, we enhance model performance through hyperparameter tuning with GridSearchCV.
#### Workflow:
1. **Data Preparation:**
    - Importing data and libraries.
    - Performing EDA with visualizations.
2. **Data Preprocessing:**
    - Cleaning and transforming the data for modeling.
3. **Modelling:**
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Hyperparameter tuning using GridSearchCV.
4. **Evaluation:**
    - Comparing models using metrics such as AUC, Precision, Recall, and F1 Score.
5. **Feature Importance:**
    - Identifying the top 10 features contributing to attrition.
     
## Results and Insights
### Dashboards:
#### Power BI and Tableau Dashboards:
- Provided interactive visualizations of attrition trends, key performance indicators, and demographic breakdowns.
- Enabled HR teams to pinpoint high-risk areas and formulate actionable retention strategies.

#### Highlights:
- **Attrition by Department:** Departments such as Sales showed higher attrition rates compared to others.
- **Demographics:** Patterns based on age, gender, and marital status revealed key factors influencing attrition.
- **Salary Trends:** Insights into the salary distribution of employees who left helped identify potential financial triggers for attrition.

### Prediction with Machine Learning
#### Performance of Models
The performance of the models was assessed using various metrics, including AUC, Precision, Recall, and F1 Score. Here are the detailed results:

| Model                | AUC   | Precision | Recall | F1 Score |
|----------------------|-------|-----------|--------|----------|
| Logistic Regression  | 0.812 | 0.600     | 0.462  | 0.522    |
| Random Forest        | 0.766 | 0.625     | 0.256  | 0.364    |
| Gradient Boosting    | 0.776 | 0.519     | 0.359  | 0.424    |

- The Logistic Regression model showed strong performance with an AUC of 0.812. Despite a moderate recall, the precision and F1 score indicate that this model balances false positives and false negatives effectively.
- The Random Forest model, even with hyperparameter tuning, achieved a lower AUC of 0.766. Its recall is particularly low, leading to a reduced F1 score of 0.364, which suggests that the model struggles to correctly identify all positive cases.
- The Gradient Boosting model performed moderately with an AUC of 0.776. Its precision and recall indicate a slightly better balance than Random Forest, but it still falls short of Logistic Regression's performance.

#### Best Model
Considering all metrics, the **Logistic Regression** model outperformed the others with an AUC of 0.812 and an F1 score of 0.522. This result implies that Logistic Regression captures the linear relationships between features and attrition effectively, making it the most suitable model for this dataset. The other models, while useful, exhibited issues likely related to the dataset size and complexity.

#### Key Insights
- **Logistic Regression:** Emerged as the best-performing model with an AUC of 0.812 and an F1 score of 0.522, indicating strong overall performance and reliable predictions.
- **Random Forest and Gradient Boosting:** These models did not perform as well, likely due to the small dataset size and potential overfitting issues. Their complex model construction processes were less effective with the given data.
- **Feature Importance:** The top features influencing attrition were OverTime_Yes, PerformanceRating, and BusinessTravel_Travel_Frequently. These features play a significant role in predicting attrition and can be targeted for interventions.
  - **OverTime_Yes:** Most significant feature indicating a strong relationship between overtime and attrition.
  - **PerformanceRating:** Indicates that performance ratings significantly impact the likelihood of attrition.
  - **BusinessTravel_Travel_Frequently:** Frequent business travel is another important factor influencing attrition.

This analysis underscores the importance of linear relationships in predicting attrition and highlights the need for larger datasets to fully leverage ensemble methods like Random Forest and Gradient Boosting.

## Usage
To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaEadara/Analysis-and-prediction-of-Employee-Attrition-using-Machine-learning-Algorithms.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Analysis-and-prediction-of-Employee-Attrition-using-Machine-learning-Algorithms
   ```
3. Install Dependencies:
Make sure you have Python installed. Then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Notebook:
Open the Jupyter Notebook and execute the cells
   ```bash
   jupyter notebook nalysis_and_prediction_of_Employee_Attrition_using_Machine_learning_Algorithms.ipynb
   ```
5. Ensure the dataset `IBM-HR-Employee-Attrition.csv` is available in the `data` folder in the project directory.
6. Run the cells sequentially to execute the analysis.
7. Explore the Power BI and Tableau dashboards by opening the respective `.pbix` and `.twbx` files in Power BI and Tableau.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork this repository, make changes, and submit a pull request. Please ensure your code adheres to the project structure and is well-documented.
