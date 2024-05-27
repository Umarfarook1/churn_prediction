# Customer Churn Prediction with Neural Network

Customer churn prediction is a crucial aspect of customer retention strategy for businesses. This project aims to predict customer churn using a neural network on the Telco customer churn dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Customer churn is when a customer decides to stop using services or products from a company. Retaining existing customers is less expensive than acquiring new ones. This project uses a neural network to predict which customers are likely to churn, helping businesses to take proactive measures.

## Dataset

The dataset used is the Telco customer churn dataset from Kaggle. It contains information about a fictional telco company that provided home phone and Internet services to customers in California. It indicates which customers have left, stayed, or signed up for their service.

- [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn tensorflow keras-tuner matplotlib joblib
    ```

## Usage

1. Run the main script to train the model and generate evaluation metrics:
    ```bash
    python churn_prediction_tuned.py
    ```

2. The script will output the model's performance metrics and save the trained model and preprocessing objects for future use.

## Hyperparameter Tuning

The project includes hyperparameter tuning using Keras Tuner. The tuner searches for the best hyperparameters and trains the model with the optimal configuration.

## Results

After running the script, you will see the training and validation accuracy and loss plotted over the epochs. The evaluation metrics, including accuracy, precision, recall, and F1-score, will be printed.

### Example Output

Accuracy: 0.82
Precision: 0.80
Recall: 0.75
F1-score: 0.77


### Training and Validation Plots

The plots showing training and validation accuracy and loss will be saved as `training_history_tuned.png`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
