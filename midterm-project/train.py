
# 1. Data preparation

# Load the dataset

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# parameters

n_estimators = 21
max_depth = 5
min_samples_leaf = 1

# data preparation
data = './diabetes_dataset.csv'
df = pd.read_csv(data)
df.head()
target_label = 'Outcome'
numeric_columns = df._get_numeric_data()
object_columns = df.select_dtypes(include=['object'])
X = numeric_columns.drop(target_label, axis=1)
y = df[[target_label]]
features = X.columns.to_list()

"""### Remove Outliers"""


def remove_outliers_iqr(df, column):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Filtering out the rows that are outside of the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]


# Remove outliers
df = remove_outliers_iqr(df, 'Glucose')
df = remove_outliers_iqr(df, 'BloodPressure')
df = remove_outliers_iqr(df, 'SkinThickness')
df = remove_outliers_iqr(df, 'Insulin')
df = remove_outliers_iqr(df, 'BMI')
df = remove_outliers_iqr(df, 'Age')

# Feature Engineering
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, np.inf],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Encode BMI_Category
data = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)

# Split dataset to train and test
categorical = [
    'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese'
]
# for c in categorical:
#     data[c] = data[c].apply(lambda x: 'yes' if x else 'no')
numerical = data.columns.drop(categorical)
numerical = numerical.to_list()
numerical.remove('BMI')
numerical.remove(target_label)
global_obese = data.Outcome.mean()
global_obese

target_label = "Outcome"


def generate_data(data):
    df_full_train, df_test = train_test_split(
        data, test_size=0.2, random_state=1)
    df_train = df_full_train[categorical+numerical]
    df_val = df_test[categorical+numerical]
    y_train = df_full_train[target_label].values
    y_val = df_test[target_label].values
    del df_full_train[target_label]
    del df_test[target_label]
    dv = DictVectorizer(sparse=False)
    train_dicts = df_train.fillna(0).to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    val_dicts = df_val.fillna(0).to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    return df_train, df_val, y_train, y_val, dv, X_train, X_val


def classification_evaluation_report(y_test, y_pred, title, best_params, t=0.5):
    print(f"{title} Classification Report: {best_params}")
    # Adjust threshold as needed
    print(classification_report(y_test, y_pred >= t))
    # Calculate each metric individually
    precision = precision_score(y_test, y_pred >= t)
    recall = recall_score(y_test, y_pred >= t)
    f1 = f1_score(y_test, y_pred >= t)
    auc_roc = roc_auc_score(y_test, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC-ROC Score: {auc_roc}")


random_state = 42


# Use 'f1' for imbalanced classes or 'roc_auc' for overall performance,# cv=5: 5-fold cross-validation
def evaluation_model(title, model, param_grid, cv=5, scoring='f1'):

    # Configure GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    # Fit the model with GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best estimator
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print()
    print("Best Hyperparameters:", best_params)
    print()
    # Step 5: Evaluate the best model
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    classification_evaluation_report(
        y_val, y_pred, "LogisticRegression", best_params)
    return (best_params, best_model)

# validation


df_train, df_val, y_train, y_val, dv, X_train, X_val = generate_data(data)

model_lr = LogisticRegression(solver='liblinear', random_state=random_state)

# Set up the hyperparameter grid
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Penalty type (L1 or L2 regularization)
    'class_weight': ['balanced', None]
}
best_params_lr, best_model_lr = evaluation_model(
    "LogisticRegression", model_lr, param_grid_lr)

# print(best_params_lr, best_model_lr)


def save_model(dv, model):
    output_file = f'model.bin'
    f_out = open(output_file, 'wb')
    pickle.dump((dv, model), f_out)
    f_out.close()
    return output_file


# Save the model
output_file = save_model(dv, best_model_lr)
print()
print(f'The model is saved to {output_file}')
