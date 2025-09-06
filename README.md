# Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease based on patient health metrics. This educational demo includes data analysis, model training, hyperparameter tuning, and an interactive web application built with Streamlit.

## 📊 Project Overview

This project uses a heart disease dataset to train and compare multiple machine learning models for binary classification. The goal is to predict whether a patient has heart disease based on various health indicators.

**⚠️ Educational Use Only**: This is a demonstration project and should not be used for actual medical diagnosis.

## 🎯 Features

- **Comprehensive Data Analysis**: Exploratory data analysis (EDA) with visualizations
- **Multiple ML Models**: Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest
- **Hyperparameter Tuning**: Optimized models using RandomizedSearchCV
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Model Comparison**: Performance metrics including accuracy, precision, recall, F1-score, and ROC-AUC


## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Navneeth
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv Heart-VENV
   source Heart-VENV/bin/activate  # On Windows: Heart-VENV\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit web app**
   ```bash
   streamlit run website.py
   ```

2. **Access the application**
   - Open your browser and go to `http://localhost:8501`
   - Input patient health metrics using the sidebar controls
   - Select a trained model from the dropdown
   - Click "Predict" to get the heart disease probability

## 📈 Dataset Information

The dataset contains 918 records with the following features:

### Input Features
- **Age**: Patient age (28-77 years)
- **Sex**: M (Male) or F (Female)
- **ChestPainType**: Type of chest pain (TA, ATA, NAP, ASY)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Cholesterol level (mg/dl)
- **FastingBS**: Fasting blood sugar > 120 mg/dl (0/1)
- **RestingECG**: Resting electrocardiogram (Normal, ST, LVH)
- **MaxHR**: Maximum heart rate achieved (60-202)
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise
- **ST_Slope**: Slope of peak exercise ST segment (Up, Flat, Down)

### Target Variable
- **HeartDisease**: 0 = Normal, 1 = Heart Disease Present

### Data Distribution
- **Total Records**: 918
- **Heart Disease Cases**: 508 (55.3%)
- **Normal Cases**: 410 (44.7%)

## 🔬 Model Performance

The project trains and compares three machine learning algorithms with hyperparameter tuning:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|---------|----------|---------|
| **Logistic Regression** | 89.1% | 88.7% | 92.2% | 90.4% | 93.1% |
| **K-Nearest Neighbors** | 85.3% | 87.9% | 85.3% | 86.6% | 93.0% |
| **Random Forest** | 85.9% | 88.0% | 86.3% | 87.1% | 92.9% |

## 🛠️ Technical Details

### Data Preprocessing
- **Missing Value Imputation**: Zero values in RestingBP and Cholesterol are imputed using hierarchical filling
- **Feature Engineering**: Categorical variables are one-hot encoded, numerical features are standardized
- **Train/Test Split**: 80/20 stratified split

### Model Training
- **Cross-Validation**: 5-fold stratified cross-validation
- **Hyperparameter Tuning**: RandomizedSearchCV with ROC-AUC optimization
- **Model Persistence**: Trained models saved using joblib

### Web Application Features
- **Interactive Input**: User-friendly form for entering patient data
- **Model Selection**: Choose between different trained models
- **Real-time Prediction**: Instant probability calculation and classification
- **Results Display**: Clear visualization of prediction results

## 📚 Usage Examples

### Running the Jupyter Notebooks

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook EDA.ipynb
   ```

2. **Model Training and Evaluation**
   ```bash
   jupyter notebook model.ipynb
   ```

### Using the Web Application

1. Launch the Streamlit app: `streamlit run website.py`
2. Enter patient information in the left sidebar
3. Select a model from the dropdown menu
4. Click "Predict" to see the results

### Example Prediction Input
```
Age: 45
Sex: M
Chest Pain Type: ASY
Resting BP: 120 mm Hg
Cholesterol: 240 mg/dl
Fasting Blood Sugar: No (0)
Resting ECG: Normal
MaxHR: 150
Exercise Angina: N
Oldpeak: 1.0
ST Slope: Up
```

## 🔧 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and tools
- **streamlit**: Web application framework
- **joblib**: Model persistence
- **tabulate**: Pretty-print tabular data

## 📊 Visualizations

The project includes various visualizations:
- ROC curves comparing model performance
- Confusion matrices for each model
- Data distribution plots
- Feature correlation analysis

## 🤝 Contributing

This is an educational project. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure you have appropriate permissions for the dataset and comply with relevant data protection regulations.

## 🚨 Disclaimer

This application is intended for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.
