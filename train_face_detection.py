import pandas as pd
import os
import cv2
from feat import Detector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
from copy import copy
import pickle


# Function to train and evaluate a model with classification metrics
def train_and_eval(model, test_in, test_out):
    predicted_val = model.predict(test_in)

    # Evaluate model using F1 score and classification report
    f1 = f1_score(test_out, predicted_val, average='weighted')
    print(f'F1 Score (weighted) on test set: {f1:.2f}')
    print("\nClassification Report on test set:")
    print(classification_report(test_out, predicted_val))
    return accuracy_score(test_out, predicted_val)


# Function to train and evaluate a regressor with regression metrics
def train_and_eval_regressor(model, test_in, test_out):
    predicted_val = model.predict(test_in)

    # Evaluate model using Mean Squared Error and R^2 Score
    mse = mean_squared_error(test_out, predicted_val)
    r2 = r2_score(test_out, predicted_val)
    return mse, r2


# Preprocessing function to load data, clean, and merge datasets
def preprocessing(au_file_path, label_path):
    dataset = pd.read_csv(label_path)
    dataset['subDirectory_filePath'] = dataset['subDirectory_filePath'].apply(lambda x: x.split('/')[-1])
    dataset['subDirectory_filePath'] = dataset['subDirectory_filePath'].str.replace('.png', '', regex=False)
    
    au = pd.read_csv(au_file_path)
    au['file'] = au['file'].apply(lambda x: x.split('/')[-1])
    au.rename(columns={'file': 'subDirectory_filePath'}, inplace=True)

    # Merge the emotion and AU dataframes based on the 'subDirectory_filePath'
    dataset = pd.merge(dataset, au, on='subDirectory_filePath', how='inner')

    # Drop rows with missing values
    dataset = dataset.dropna()

    # Extract features and labels from the dataset
    features = dataset.drop(columns=['valence', 'arousal', 'expression', 'face', 'subDirectory_filePath'])
    labels= dataset[['valence', 'arousal', 'expression']]
    train_in, test_in, train_out, test_out = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
    # Split dataset into training and testing sets
    return (train_in, test_in, train_out, test_out,features,labels)


# Train model for Valence prediction using XGBoost Regressor
def train_valence_model(train_in, test_in, train_out, test_out):
    # Define hyperparameter grid for XGBoost Regressor
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    # Set up GridSearchCV for hyperparameter tuning
    xgb_regressor = XGBRegressor(random_state=42)
    xgb_grid_search = GridSearchCV(
        estimator=xgb_regressor,
        param_grid=xgb_param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    # Fit the model and tune hyperparameters
    xgb_grid_search.fit(train_in, train_out)
    xgb_best_model_valence = xgb_grid_search.best_estimator_

    # Print best hyperparameters and evaluate the model
    print(f"Best XGBoost Regression Hyperparameters: {xgb_grid_search.best_params_}")
    xgb_test_accuracy = train_and_eval_regressor(xgb_best_model_valence, test_in, test_out)
    return xgb_best_model_valence


# Train model for Arousal prediction using XGBoost Regressor
def train_arousal_model(train_in, test_in, train_out, test_out):
    # Define hyperparameter grid for XGBoost Regressor
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    # Set up GridSearchCV for hyperparameter tuning
    xgb_regressor = XGBRegressor(random_state=42)
    xgb_grid_search = GridSearchCV(
        estimator=xgb_regressor,
        param_grid=xgb_param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    # Fit the model and tune hyperparameters
    xgb_grid_search.fit(train_in, train_out)
    xgb_best_model_arousal = xgb_grid_search.best_estimator_

    # Print best hyperparameters and evaluate the model
    print(f"Best XGBoost Regression Hyperparameters: {xgb_grid_search.best_params_}")
    xgb_test_accuracy = train_and_eval_regressor(xgb_best_model_arousal, test_in, test_out)
    return xgb_best_model_arousal


# Train model for Emotion prediction using XGBoost Classifier
def train_emotion_model(train_in, test_in, train_out, test_out):
    xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')

    # Define hyperparameter grid for XGBoost Classifier
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    # Use GridSearchCV for hyperparameter tuning
    xgb_grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_param_grid,
        scoring='f1_weighted',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    # Fit the model and tune hyperparameters
    xgb_grid_search.fit(train_in, train_out)
    print(train_in.head())
    # Print best hyperparameters and evaluate the model
    print(f"Best XGBoost Hyperparameters: {xgb_grid_search.best_params_}")
    xgb_best_model = xgb_grid_search.best_estimator_
    xgb_test_accuracy = train_and_eval(xgb_best_model, test_in, test_out)
    return xgb_best_model

# Save the models to pickle files
def save_models():
    # Get action unit and labels
    au_file_path = './aus.csv'
    label_path = './DiffusionFER/DiffusionEmotion_S/dataset_sheet.csv'  

    # Get the train and test data
    train_in, test_in, train_out, test_out,features,labels = preprocessing(au_file_path,label_path)

    # Train the models for valence, arousal, and emotion
    xgb_best_model_arousal = train_arousal_model(train_in, test_in, train_out['arousal'], test_out['arousal'])
    xgb_best_model_valence = train_valence_model(train_in, test_in, train_out['valence'], test_out['valence'])

    predicted_valence = xgb_best_model_valence.predict(features)
    predicted_arousal = xgb_best_model_arousal.predict(features)

    predictions_df = pd.DataFrame({
        'valence': predicted_valence,
        'arousal': predicted_arousal
    })

    # Combine the features and predictions into a new DataFrame
    merged_dataset = pd.concat([features.reset_index(drop=True), predictions_df], axis=1)


    train_in, test_in, train_out, test_out = train_test_split(
    merged_dataset,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels # balances labels across the sets
    )
    print(merged_dataset.head)
    xgb_best_model_emotion = train_emotion_model(train_in, test_in, train_out['expression'], test_out['expression'])

    # Save models to pickle files
    with open('model_arousal.pkl', 'wb') as f:
        pickle.dump(xgb_best_model_arousal, f)
    with open('model_valence.pkl', 'wb') as f:
        pickle.dump(xgb_best_model_valence, f)
    with open('model_emotion.pkl', 'wb') as f:
        pickle.dump(xgb_best_model_emotion, f)

# Call save_models to train and save the models
save_models()
