# US-Visa-Approval-Prediction

## step 1:
Created git repository and Cloned to local. Created file structure using template.txt and also setup.txt for custom package built. Created virtual env and installed packages for the prj using requirement.txt

## step 2:
Mongodb setup and via connection string pushed data. 
Data we took from kaggle website

## step 3:
Code inside logger for custom logs and file will be creadited according to timestamp.
Code inside exception for error handling, and also we can log that error using logger and exception
Utils has function which are frequently used throughout the project, simply we have to import and use it.

## step 4: 📊 Exploratory Data Analysis
The focus is on understanding the data rather than building a model including load the raw dataset, inspect its structure, visualise distributions and relationships, and document the most important patterns. By the end of the analysis you know which columns carry useful information, which columns are redundant, and what kinds of transformations will be needed in the modelling stage.

### 🔍 Data Overview
- Inspected top rows and summarized columns.
- Checked data types to distinguish numeric vs. categorical features.
- Found wide range in numeric features:
  - `no_of_employees`: single digits to hundreds of thousands.
  - `yr_of_estab`: spans 1950s to 2010s.
- Target variable (`case_status`) is imbalanced: majority are **Denied**, minority are **Certified**.

### 📈 Univariate Analysis

#### Numeric Features
- **`no_of_employees`**: Right-skewed with many outliers → needs transformation and outlier clipping.
- **`yr_of_estab`**: Left-skewed; many companies founded after 2000 → will be transformed into company age.
- **`prevailing_wage`**: Right-skewed with high-value outliers → median varies widely across categories.

#### Categorical Features
- **`continent`**: Dominated by Asia; rare categories will be grouped into “Other”.
- **`unit_of_wage`** and **`full_time_position`**: Highly unbalanced, dominated by “Hourly” and “Yes”.

### 🔗 Multivariate Analysis & Feature–Target Relationships

- **Chi-squared tests** show most categorical features are dependent on `case_status`.
- **`requires_job_training`** shows no significant relationship → will be dropped.
- **Target imbalance**: Only ~⅓ of applications are Certified.
- **Continent vs. case_status**: Asia dominates volume; Europe has highest certification rate.
- **Education vs. case_status**: Master’s/Doctorate holders have higher approval rates.
- **Job experience vs. case_status**: Experienced applicants ~75% Certified vs. ~56% for inexperienced.
- **Employer size & wages**: Similar distributions across outcomes; many outliers.
- **Wage unit**: Hourly wages correlate with higher denial rates; yearly salaries with higher approval.
- **Region of employment**: Slight edge for Midwest and South.
- **Prevailing wage by education/experience/continent**:
  - Master’s → highest median wage.
  - No experience → slightly higher wages (unexpected).
  - Asian applicants → highest average wage.
- **Year of establishment**: Binned into 5-year intervals; surge post-2000 → motivates age transformation.

### ✅ Key Insights & Recommendations

- **Drop irrelevant columns**:
  - `case_id`: identifier, no predictive power.
  - `requires_job_training`: not correlated with outcome.
- **Handle skewness & outliers**:
  - Apply log/Yeo-Johnson transforms.
  - Cap extreme values.
- **Address target imbalance**:
  - Use SMOTE or class-weighted loss.
- **Transform `yr_of_estab`** into company age.
- **Combine rare categories**:
  - Group small `continent` values into “Other”.

### 🧠 Conclusion

This EDA provides a strong foundation for modeling:
- Highlights influential features.
- Identifies preprocessing needs (skewness, outliers, imbalance).
- Guides feature engineering and resampling strategies.
- Sets the stage for building a robust predictive model focused on meaningful patterns.
