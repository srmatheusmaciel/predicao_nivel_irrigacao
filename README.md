# Irrigation Area Prediction Model

A machine learning model designed to predict irrigation area coverage based on irrigation duration, helping optimize water usage for agricultural applications.

## 📋 Project Overview

This project implements a linear regression model to predict the irrigation area coverage per angle based on the number of hours of irrigation. The model establishes a clear relationship between irrigation time and coverage area, enabling farmers and agricultural engineers to plan irrigation schedules more efficiently.

## 🌱 Features

- Data preprocessing and cleaning
- Exploratory data analysis with comprehensive visualizations
- Linear regression model for irrigation area prediction
- Model performance evaluation using multiple metrics
- Prediction capabilities for new irrigation scenarios
- In-depth residual analysis

## 📁 Project Structure

```
predicao_nivel_irrigacao/
│
├── datasets/
│   └── dados_de_irrigacao.csv  # Dataset with irrigation information
│
├── modelo_pred_irrigacao.ipynb # Jupyter notebook with model development
├── Pipfile                     # Dependency management
├── Pipfile.lock               # Locked dependencies
└── README.md                  # This documentation file
```

## 📊 Data Description

The dataset (`dados_de_irrigacao.csv`) contains the following columns:
- `horas_irrigacao`: Number of hours of irrigation (independent variable)
- `area_irrigada`: Total irrigated area
- `area_irrigada_angulo`: Irrigated area per angle (target variable)

## 🔧 Requirements

The project requires the following Python libraries:
- pandas: Data manipulation and analysis
- matplotlib & seaborn: Data visualization
- numpy: Numerical operations
- scikit-learn: Machine learning algorithms
- scipy: Statistical functions
- statsmodels: Statistical models and tests
- fastapi & uvicorn: API development (for deployment)
- pydantic: Data validation
- pingouin: Statistical analysis

## 🛠️ Installation

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/username/predicao_nivel_irrigacao.git
cd predicao_nivel_irrigacao

# Install dependencies using pipenv
pipenv install
```

Or install the required packages manually:

```bash
pip install scikit-learn scipy pandas matplotlib statsmodel fastapi uvicorn pydantic pingouin seaborn ipykernel
```

## 📈 Model Development Process

### 1. Data Loading and Preprocessing

The model starts by loading the irrigation dataset and performing basic preprocessing:
- Cleaning column names (removing spaces and converting to lowercase)
- Checking for missing values
- Analyzing the data structure

### 2. Exploratory Data Analysis (EDA)

Before model training, we explore the data through:
- Correlation analysis using Pearson and Spearman methods
- Scatter plots to visualize relationships between variables
- Box plots to detect potential outliers
- Pair plots to understand variable distributions and relationships
- Statistical summaries of all variables

### 3. Model Training

We use scikit-learn's LinearRegression to train the model:

```python
# Split dataset between training and testing sets
X = df_irrigacao[['horas_irrigacao']].values.reshape(-1,1)
y = df_irrigacao['area_irrigada_angulo'].values.reshape(-1,1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)

# Train the model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
```

### 4. Model Evaluation

The model's performance is evaluated using multiple metrics:
- R-squared (R²) score: 1.0 (perfect fit)
- Mean Absolute Error (MAE): ~1.69e-12 (negligible)
- Mean Squared Error (MSE): ~5.15e-24 (negligible)
- Root Mean Squared Error (RMSE): ~2.27e-12 (negligible)

### 5. Residual Analysis

We perform a thorough analysis of the model's residuals:
- Calculating standardized residuals
- Testing for normality using the Kolmogorov-Smirnov test
- Visualizing residual distribution

## 📊 Results

The linear regression model demonstrates a perfect fit for this dataset, with an R² score of 1.0. The model determined the following linear equation:

```
area_irrigada_angulo = 66.666667 × horas_irrigacao + 0
```

This indicates that for each hour of irrigation, the irrigated area per angle increases by approximately 66.67 units.

## 🔍 Example Usage

```python
import numpy as np

# Predict irrigated area for 15 hours of irrigation
horas_exemplo = np.array([[15]])
area_predita = reg_model.predict(horas_exemplo)
# Result: For 15 hours of irrigation, the predicted irrigated area per angle is 1000

# Predict irrigated area for 30 hours of irrigation
reg_model.predict([[30]])
# Result: Array([[2000.]])
```

## 🚀 Future Improvements

- Incorporate additional features such as:
  - Soil type
  - Weather conditions
  - Crop type
  - Water pressure
- Develop a web interface for easy prediction access
- Implement time series forecasting for irrigation scheduling
- Add a monitoring system for real-time irrigation optimization

## 👨‍💻 Contributors

- Matheus Maciel - Model development and analysis

## 📜 License

MIT License
