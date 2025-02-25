# Predict Customer Churn

- Project **Predict Customer Churn** for ML DevOps Engineer Nanodegree at Udacity.

## Project Description
This repository contains the churn prediction project developed for the Udacity ML DevOps Engineer Nanodegree. The project involves building and testing a machine learning model to predict customer churn.

## Files and Data Description
- `churn_library.py`: Library containing all the functions for the churn prediction process.
- `churn_script_logging_and_tests.py`: Script for running all tests and logging the results to ensure the functionality of the churn library.
- `data/bank_data.csv`: Dataset used for model training and testing.
- `models/`: Directory where trained models are saved.
- `images/`: Directory where plots from the EDA are saved.
- `logs/`: Directory where logs from the tests are saved.
- `requirements.txt`: Required libraries for the project setup.

## Installation
To run this project, follow these steps:

## Installation and Setup

Follow these instructions to get the project set up and running on your local machine.

### Prerequisites

You need to have Python 3.7 or higher installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).

### Cloning the Repository

Clone the repository to your local machine by running the following command in your terminal or command prompt:

```bash
git clone https://github.com/brunogolfette/udacity-churn-prediction.git
cd udacity-churn-prediction

Installing Dependencies
Install the necessary Python dependencies by running:
pip install -r requirements.txt

Running the Tests
To execute the tests and verify the functionality, run the testing script from the project directory:
python churn_script_logging_and_tests.py


Usage
To use the churn prediction functionalities in your projects or scripts, ensure you import the necessary functions from the churn library:
from churn_library import perform_eda, encoder_helper, perform_feature_engineering, train_models

Contributing
Contributions are welcome, and any improvements or suggestions are appreciated. Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
