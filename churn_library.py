"""
This module provides functions to perform data preprocessing, exploratory data analysis,
feature engineering, model training, and evaluation for a churn prediction model.
"""
# import libraries
import logging
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = "./data/bank_data.csv"
IMAGE_OUTPUT_PATH = './images/results/'
MODEL_PATH = './models/rfc_model.pkl'
RESPONSE = 'Churn'


def import_data(pth):
    ''' Returns dataframe for the csv found at path. '''
    try:
        return pd.read_csv(pth)
    except Exception as expt_import:
        logging.error("An error occurred during Import: %s", expt_import)
        raise RuntimeError("Failed to perform Import Data.") from expt_import


def perform_eda(df_input):
    ''' Perform EDA and save figures to the images/eda directory. '''

    try:
        # Convert 'Attrition_Flag' into a binary 'Churn' column if needed
        if 'Churn' not in df_input.columns:
            df_input['Churn'] = df_input['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1)

        # Path to the directory where the graphs will be saved
        save_path = './images/eda/'

        # Check if the directory exists; if not, create it.
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Function to save histogram of 'Churn'
        def save_churn_hist():
            plt.figure(figsize=(20, 10))
            df_input['Churn'].hist()
            file_path = os.path.join(save_path, 'histogram_churn.png')
            plt.savefig(file_path)
            plt.close()

        # Function to save histogram of 'Customer_Age'
        def save_customer_age_hist():
            plt.figure(figsize=(20, 10))
            df_input['Customer_Age'].hist()
            file_path = os.path.join(save_path, 'customer_age_hist.png')
            plt.savefig(file_path)
            plt.close()

        # Function to save bar plot of 'Marital_Status'
        def save_marital_status_bar():
            plt.figure(figsize=(20, 10))
            df_input['Marital_Status'].value_counts(
                normalize=True).plot(kind='bar')
            file_path = os.path.join(save_path, 'marital_status_bar.png')
            plt.savefig(file_path)
            plt.close()

        # Function to save histogram of 'Total_Trans_Ct' with KDE
        def save_transactions_hist():
            plt.figure(figsize=(20, 10))
            sns.histplot(df_input['Total_Trans_Ct'], stat='density', kde=True)
            file_path = os.path.join(save_path, 'total_trans_hist.png')
            plt.savefig(file_path)
            plt.close()

        # Function to save heatmap of correlations
        def save_correlation_heatmap():
            plt.figure(figsize=(20, 10))
            sns.heatmap(
                df_input.corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2)
            file_path = os.path.join(save_path, 'correlation_heatmap.png')
            plt.savefig(file_path)
            plt.close()

        # Call functions to generate and save plots
        save_churn_hist()
        save_customer_age_hist()
        save_marital_status_bar()
        save_transactions_hist()
        save_correlation_heatmap()

    except Exception as except_eda:
        logging.error("An error occurred during EDA: %s", except_eda)
        raise RuntimeError("Failed to perform EDA.") from except_eda


def encoder_helper(df_input, category_lst, response='Churn'):
    '''
    Helper function to turn each categorical column into a new column with
    the proportion of the response (e.g., churn) for each category.

    Input:
    df_input: pandas DataFrame - The dataframe to process.
    category_lst: list - A list of column names that contain categorical features.
    response: str - The name of the response column (default is 'Churn').

    Output:
    df_input: pandas DataFrame - The dataframe with new columns
    for each categorical variable encoded.
    '''
    if response not in df_input.columns:
        error_msg = f"The required column '{response}' is missing from the dataframe."
        logging.error(error_msg)
        raise KeyError(error_msg)
    try:
        def encode_column(column):
            '''
            Encodes a single categorical column using mean of the response variable.

            Input:
            column: str - The name of the column to encode.

            Output:
            None - The function directly modifies the dataframe in place.
            '''
            # Group by the category and calculate the mean of the response
            group_means = df_input.groupby(column).mean()[response]
            # Map the mean response to each entry in the dataframe
            encoded_column_name = f"{column}_{response}"
            df_input[encoded_column_name] = df_input[column].map(group_means)

        # Apply encoding to each category in the list
        for category in category_lst:
            encode_column(category)

        return df_input

    except Exception as excep_helper:
        logging.error(
            "An error occurred during encoder helper: %s",
            excep_helper)
        raise RuntimeError(
            "Failed to perform encoder helper.") from excep_helper


def perform_feature_engineering(df_input, response):
    '''
    Perform feature engineering to prepare the dataset for machine learning models
    by selecting specified columns and splitting the data into training and testing sets.

    Input:
        df_input: pandas DataFrame - The dataframe to process.
        response: str - The name of the response column to be used as the target variable.

    Output:
        X_train: DataFrame - Training feature data.
        X_test: DataFrame - Testing feature data.
        y_train: Series - Training target data.
        y_test: Series - Testing target data.

    Raises:
        Exception: Captures and logs general exceptions if the feature engineering process fails.
    '''
    try:
        # Extracting the response variable
        target = df_input[response]

        # Define the columns to keep in the feature matrix X
        keep_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn']

        # Ensure all specified columns are in the DataFrame, to avoid KeyError
        filtered_cols = [col for col in keep_cols if col in df_input.columns]

        # Creating the feature matrix X with the specified columns
        features = df_input[filtered_cols]

        return train_test_split(
            features,
            target,
            test_size=0.3,
            random_state=42)

    except Exception as excep_perf:
        logging.error(
            "An error occurred during feature engineer: %s",
            excep_perf)
        raise RuntimeError(
            "Failed to perform feature engineer.") from excep_perf


def train_models(X_train, X_test, y_train, y_test):
    '''
    Trains logistic regression and random forest models,
    evaluates them with classification reports,
    saves the models to disk, and returns training and testing predictions.

    Input:
        X_train: DataFrame - Training feature data.
        X_test: DataFrame - Testing feature data.
        y_train: Series - Training target data.
        y_test: Series - Testing target data.

    Output:
        Tuple - Contains y_train, y_test, y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf
    '''
    try:
        # Ensure the models directory exists
        models_dir = './models/'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        # Initialize the models
        lrc_model = LogisticRegression(solver='lbfgs',
                                       max_iter=3000
                                       )
        rfc_model = RandomForestClassifier(random_state=42)
        # Set up parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5],
            'criterion': ['gini', 'entropy']
        }
        # GridSearch for optimal parameters in Random Forest
        cv_rfc = GridSearchCV(estimator=rfc_model,
                              param_grid=param_grid,
                              cv=5
                              )
        cv_rfc.fit(X_train, y_train)

        # Fit Logistic Regression
        lrc_model.fit(X_train, y_train)

        # Predictions
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        y_train_preds_lr = lrc_model.predict(X_train)
        y_test_preds_lr = lrc_model.predict(X_test)

        # Save models
        joblib.dump(
            cv_rfc.best_estimator_,
            os.path.join(
                models_dir,
                'rfc_model.pkl'))
        joblib.dump(
            lrc_model,
            os.path.join(models_dir, 'logistic_model.pkl')
        )

        return y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


    except Exception as excep_train:
        logging.error("An error occurred during train models: %s", excep_train)
        raise RuntimeError("Failed to perform Train Models.") from excep_train


def classification_report_image(
        y_train_func,
        y_test_func,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf):
    '''
    Produces class report for training and testing results and save as image in images folder.

    Input:
        y_train: training response values
        y_test: test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    Output:
        None
    '''
    try:
        def save_classification_report(
                y_true, y_pred, model_name, dataset_type):
            '''
            Saves the classification report as an image.

            Input:
            y_true: actual target values
            y_pred: predictions from model
            model_name: name of the model (e.g., 'logistic_regression', 'random_forest')
            dataset_type: type of data (e.g., 'train', 'test')

            Output:
            None - saves an image file of the classification report
            '''
        #     report = classification_report(y_true, y_pred, output_dict=True)
            ax_class_graph = plt.subplots(figsize=(10, 10))[1]
            ax_class_graph.axis('off')
            ax_class_graph.text(
                0,
                0,
                f"{model_name.upper()} {dataset_type.upper()} REPORT:\n" +
                classification_report(
                    y_true,
                    y_pred),
                fontdict={
                    'fontsize': 14},
                va='top')
            filepath = f'./images/results/{model_name}_{dataset_type}_report.png'
            plt.savefig(filepath)
            plt.close()

        def plot_roc_curves():
            '''
            Plots ROC curves for both models and saves as an image.
            '''

            plt.figure(figsize=(15, 8))
            ax_roc_graph = plt.gca()
            lrc = joblib.load('./models/logistic_model.pkl')
            rfc_best = joblib.load('./models/rfc_model.pkl')

            models = [
                ('Random Forest', rfc_best, y_test),
                ('Logistic Regression', lrc, y_test)
            ]

            for name, model, preds in models:
                preds = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, preds)
                roc_auc = auc(fpr, tpr)
                RocCurveDisplay(
                    fpr=fpr,
                    tpr=tpr,
                    roc_auc=roc_auc,
                    estimator_name=name).plot(
                    ax=ax_roc_graph)

            plt.savefig('./images/results/roc_curves.png')
            plt.close()

        # Call the function to save classification reports
        save_classification_report(
            y_train_func,
            y_train_preds_lr,
            'logistic_regression',
            'train')
        save_classification_report(
            y_test_func,
            y_test_preds_lr,
            'logistic_regression',
            'test')
        save_classification_report(
            y_train_func,
            y_train_preds_rf,
            'random_forest',
            'train')
        save_classification_report(
            y_test_func,
            y_test_preds_rf,
            'random_forest',
            'test')

        # Plot and save the ROC curves
        plot_roc_curves()

    except Exception as excep_class_report:
        logging.error(
            "An error occurred during classification class report: %s",
            excep_class_report)
        raise RuntimeError(
            "Failed to perform classification class report.") from excep_class_report


def load_model(model_path):
    '''
    Safely loads a machine learning model from the specified path.

    Input:
        model_path: str - Path to the machine learning model file.

    Output:
        model - Loaded model if successful, None otherwise.
    '''
    try:
        model = joblib.load(model_path)
        return model
    except Exception as excep_load:
        logging.error(
            "An error occurred during classification load model: %s",
            excep_load)
        raise RuntimeError("Failed to perform load model:.") from excep_load


def feature_importance_plot(model, df_features, output_pth):
    '''
    Creates and stores both traditional feature importances and
    SHAP summary plots for tree-based models.

    Input:
    model: The trained tree-based model object (e.g., RandomForest, XGBoost).
    df_features: pandas DataFrame - The feature data used for the analysis (X_data)
    output_pth: str - The directory path where the feature importance plots will be saved.

    Output:
    None - The function saves feature importance plots in the specified output path.
    '''
    try:
        def ensure_dir(directory):
            ''' Ensure the directory exists. If not, create it. '''
            if not os.path.exists(directory):
                os.makedirs(directory)

        def plot_feature_importances():
            ''' Plot and save traditional feature importances. '''
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            names = [df_features.columns[i] for i in indices]

            plt.figure(figsize=(20, 5))
            plt.title("Feature Importance")
            plt.ylabel('Importance')
            plt.bar(range(df_features.shape[1]), importances[indices])
            plt.xticks(
                range(
                    df_features.shape[1]),
                names,
                rotation=90,
                ha='right')
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_pth, 'feature_importances.png')
            )
            plt.close()

        def plot_shap_summary():
            ''' Plot and save SHAP summary plot. '''
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_features)
            shap.summary_plot(
                shap_values,
                df_features,
                plot_type="bar",
                show=False)
            plt.savefig(os.path.join(output_pth, 'shap_summary_plot.png'))
            plt.close()

        # Ensure the output directory exists
        ensure_dir(output_pth)
        # Generate and save the plots
        plot_feature_importances()
        plot_shap_summary()
    except Exception as excep_feat:
        logging.error(
            "An error occurred during classification feature importance: %s",
            excep_feat)
        raise RuntimeError(
            "Failed to perform feature importance.") from excep_feat


if __name__ == '__main__':

    df_data = import_data(DATA_PATH)
    perform_eda(df_data)
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    DF_ENCODED = encoder_helper(df_data, category_list, RESPONSE)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        DF_ENCODED, RESPONSE)
    results = train_models(X_train, X_test, y_train, y_test)
    classification_report_image(*results)
    rfc = load_model(MODEL_PATH)
    if rfc:
        feature_importance_plot(rfc, X_test, IMAGE_OUTPUT_PATH)
