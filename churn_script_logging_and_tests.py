"""
This module performs unit tests for churn prediction library functions. It tests
data importing, exploratory data analysis (EDA), feature engineering, and model training
functions to ensure they work correctly and as expected.

Author: Bruno
Date: 2025-02-22
"""
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import churn_library as cls


# Creating a timestamp
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Setting up logging
log_filename = f'./logs/churn_library_{current_time}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_import(import_data_func):
    """
    Test the data import function for loading data correctly.

    Args:
        import_data_func (function): The function to test data importing.

    Raises:
        FileNotFoundError: If the specified file is not found.
        AssertionError: If the loaded dataframe does not have the expected dimensions.
    """
    try:
        data_frame = import_data_func("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as error:
        logging.error("Testing import_data: The file wasn't found - %s", error)
        raise error

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as error:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns - %s",
            error)
        raise error


def test_eda(perform_eda_func, data_frame):
    """
    Test the perform EDA function to ensure it completes without errors.

    Args:
        perform_eda_func (function): The function to test EDA operations.
        data_frame (DataFrame): The DataFrame on which EDA will be performed.

    Raises:
        Exception: If an error occurs during the EDA process.
    """
    try:
        perform_eda_func(data_frame)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as error:
        logging.error("Testing perform_eda: Failed to perform EDA - %s", error)
        raise error


def test_encoder_helper(encoder_helper_func, data_frame):
    """
    Test the encoder helper function by ensuring it adds encoded columns correctly.

    Args:
        encoder_helper_func (function): The function to test for encoding categorical features.
        data_frame (DataFrame): The DataFrame to encode.

    Raises:
        ValueError: If the 'Churn' column is missing in the DataFrame.
        AssertionError: If the encoded columns are missing after encoding.
    """
    required_column = 'Churn'
    if required_column not in data_frame.columns:
        error_message = f"{required_column} column is not present in the DataFrame for testing."
        logging.error(error_message)
        raise ValueError(error_message)
    category_list = ['Gender', 'Education_Level',
                     'Marital_Status', 'Income_Category', 'Card_Category']
    expected_columns = [
        f"{category}_{required_column}" for category in category_list]

    try:
        df_encoded = encoder_helper_func(data_frame, category_list)
        missing_columns = [
            col for col in expected_columns if col not in df_encoded.columns]
        assert not missing_columns, f"Missing encoded columns: {missing_columns}"
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as error:
        logging.error("Testing encoder_helper: Failed - %s", error)
        raise


def test_perform_feature_engineering(
        perform_feature_engineering_func,
        data_frame):
    """
    Test the perform feature engineering function to ensure it correctly splits the data.

    Args:
        perform_feature_engineering_func (function): The function to test feature engineering.
        data_frame (DataFrame): The DataFrame to process.

    Raises:
        AssertionError: If the splits do not contain data.
    """
    try:
        X_train_test, X_test_test, y_train_test, y_test_test = perform_feature_engineering_func(
            data_frame, 'Churn')
        assert X_train_test.shape[0] > 0
        assert X_test_test.shape[0] > 0
        assert y_train_test.shape[0] > 0
        assert y_test_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as error:
        logging.error(
            "Testing perform_feature_engineering: Failed to split data - %s",
            error)
        raise error


def test_train_models(train_models_func, X_train, X_test, y_train, y_test):
    """
    Test the train models function to ensure it trains models and generates predictions.

    Args:
        train_models_func (function): The function to test model training.
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.

    Raises:
        AssertionError: If the predictions are not of expected length or are None.
    """
    try:
        results = train_models_func(X_train, X_test, y_train, y_test)
        assert len(
            results) == 6, "Expected six output values from train_models function"
        # Unpack results into respective variables
        y_train, y_test = results[:2]
        y_train_preds_lr, y_train_preds_rf = results[2:4]
        y_test_preds_lr, y_test_preds_rf = results[4:6]

        # Log lengths to help diagnose issues
        logging.info("Length of y_train: %d", len(y_train))
        logging.info("Length of y_test: %d", len(y_test))
        logging.info("Length of y_train_preds_lr: %d", len(y_train_preds_lr))
        logging.info("Length of y_train_preds_rf: %d", len(y_train_preds_rf))
        logging.info("Length of y_test_preds_lr: %d", len(y_test_preds_lr))
        logging.info("Length of y_test_preds_rf: %d", len(y_test_preds_rf))
        # Check if predictions are not None
        assert y_train_preds_lr is not None
        assert y_train_preds_rf is not None
        assert y_test_preds_lr is not None
        assert y_test_preds_rf is not None
        # Check if predictions are the correct form
        assert len(y_train_preds_lr) == len(y_train)
        assert len(y_train_preds_rf) == len(y_train)
        assert len(y_test_preds_lr) == len(y_test)
        assert len(y_test_preds_rf) == len(y_test)
        # Optionally check the type of predictions
        assert isinstance(y_train_preds_lr, (np.ndarray, pd.Series))
        assert isinstance(y_train_preds_rf, (np.ndarray, pd.Series))
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as error:
        logging.error(
            "Testing train_models: The function's output is not as expected - %s",
            error)
        raise error


def run_tests():
    """
    Execute all unit tests for the churn prediction functions to ensure their
    integrity and correctness.
    """
    try:
        data_frame = cls.import_data("./data/bank_data.csv")
        test_import(cls.import_data)
        cls.perform_eda(data_frame)
        test_eda(cls.perform_eda, data_frame)
        category_list = ['Gender', 'Education_Level',
                         'Marital_Status', 'Income_Category', 'Card_Category']
        df_encoded = cls.encoder_helper(data_frame, category_list)
        test_encoder_helper(cls.encoder_helper, df_encoded)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df_encoded, 'Churn')
        test_perform_feature_engineering(
            cls.perform_feature_engineering, df_encoded)
        test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
        logging.info("All tests passed successfully!")
    except Exception as exc:
        logging.error("Error during testing: %s", str(exc))
        raise exc


if __name__ == "__main__":
    run_tests()
