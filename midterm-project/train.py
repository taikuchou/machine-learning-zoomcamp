
# 1. Data preparation

# Load the dataset

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
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


def generate_data(data):
    df_full_train, df_test = train_test_split(
        data, test_size=0.2, random_state=1)
    df_train = df_full_train[categorical+numerical]
    df_val = df_test[categorical+numerical]
    y_train = df_full_train[target_label].values
    y_val = df_test[target_label].values
    del df_full_train[target_label]
    del df_test[target_label]
    return df_train, df_val, y_train, y_val


df_train, df_val, y_train, y_val = generate_data(data)

# validation

dv = DictVectorizer(sparse=False)
train_dicts = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)


scores = []

for n in tqdm(range(1, 26, 2)):
    rf = RandomForestClassifier(n_estimators=n,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((max_depth, n, auc))

print('validation results:')
print('n_estimators= %.0f ,max_depth= %.0f, min_samples_leaf = %.0f' %
      (n_estimators, max_depth, min_samples_leaf))
columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
print(df_scores)

# training the final model
df_train, df_val, y_train, y_val = generate_data(data)
dv = DictVectorizer(sparse=False)
train_dicts = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)
n_estimators = 21
max_depth = 5
min_samples_leaf = 1
rf = RandomForestClassifier(n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f'auc={auc}')


def save_model(dv, model, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf):
    output_file = f'rfc_model_n_estimators={n_estimators}_max_depth={max_depth}_min_samples_leaf={min_samples_leaf}.bin'
    f_out = open(output_file, 'wb')
    pickle.dump((dv, model), f_out)
    f_out.close()
    return output_file


# Save the model
output_file = save_model(dv, rf)
print(f'the model is saved to {output_file}')
