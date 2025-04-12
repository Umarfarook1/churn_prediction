# Customer Churn Prediction with Neural Network

Customer churn prediction is a crucial aspect of customer retention strategy for businesses. This project leverages a neural network to classify whether a customer is likely to churn based on historical data. The implementation uses Python, TensorFlow/Keras, and Scikit-learn, providing end-to-end functionality including preprocessing, training, evaluation, and saving the model.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Project Achievements](#project-achievements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Output Files](#output-files)
- [Contributing](#contributing)
- [License](#license)

## Overview
Customer churn refers to when a customer stops doing business or ends a subscription with a company. Identifying potential churners early can help organizations take preventive actions. This project trains a neural network on customer behavioral data to predict churn.

## Problem Statement
The objective is to build a predictive model that determines whether a customer will churn based on features such as tenure, contract type, monthly charges, payment method, etc. We aim to:
- Clean and preprocess the dataset
- Encode categorical features
- Scale numerical features
- Train and evaluate a deep learning model
- Save the model and preprocessing objects
- Visualize training history

## Dataset
We use the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn). It includes 7,043 rows and 21 columns, each representing customer-related features such as demographics, account information, and services used.

### Key Columns:
- **customerID**: Unique identifier for each customer
- **gender, SeniorCitizen, Partner, Dependents**: Demographic info
- **tenure, MonthlyCharges, TotalCharges**: Account and usage data
- **Churn**: Target variable (Yes/No)

## Project Architecture
```
customer-churn-prediction/
├── churn_prediction.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── label_encoders.pkl
├── scaler.pkl
├── customer_churn_model.h5
├── training_history.png
└── README.md
```

## Project Achievements

### ✅ What We Achieve in This Project:

#### 🧠 1. Machine Learning in Business Context
- Solves a real-world business problem: **predicting customer churn**
- Applicable in telecom, SaaS, banking, and other domains

#### 🧹 2. Robust Data Preprocessing Pipeline
- Handles missing values
- Encodes categorical data
- Scales numerical features
- Builds reusable encoders and scalers

#### 🧠 3. Neural Network Model Development
- Implements a deep learning model using **Keras**
- Uses modern best practices (ReLU, Adam optimizer, Binary Crossentropy loss)
- Demonstrates hands-on experience with TensorFlow/Keras

#### 📈 4. Evaluation & Performance Analysis
- Uses multiple metrics: **Accuracy, Precision, Recall, F1-score**
- Provides deeper insights into model performance

#### 💾 5. Reusability & Deployment Readiness
- Saves the model and preprocessing objects
- Enables future reuse or real-world deployment

#### 📊 6. Training Visualization
- Plots and saves training & validation **accuracy/loss** curves
- Helps diagnose overfitting and training trends

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Install the dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib joblib
```

## Usage
1. Ensure the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project directory.
2. Run the script:
```bash
python churn_prediction.py
```

## Model Architecture
- **Input Layer**: Number of neurons equal to feature size
- **Hidden Layer 1**: 64 neurons, ReLU activation
- **Hidden Layer 2**: 32 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation for binary classification
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Epochs**: 50
- **Batch Size**: 10

## Hyperparameter Tuning
Currently, the project uses fixed hyperparameters. To improve performance:
- Integrate Keras Tuner or Optuna for automated search
- Tune: number of layers, neurons per layer, activation functions, learning rate, batch size

## Evaluation Metrics
The model is evaluated on the test set using:
- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall

## Results
Example Output (varies per run):
```
Accuracy: 0.8027
Precision: 0.6518
Recall: 0.5469
F1-score: 0.5948
```

### Training History Plots
Training and validation accuracy/loss are visualized and saved as:
- `training_history.png`

## Output Files
- **customer_churn_model.h5**: Trained model
- **label_encoders.pkl**: Serialized label encoders
- **scaler.pkl**: Serialized feature scaler
- **training_history.png**: Model accuracy/loss plots

## Contributing
Contributions are welcome! Please fork the repo, create a new branch, and open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
