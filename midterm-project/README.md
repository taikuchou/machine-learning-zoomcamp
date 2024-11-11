### 1. Problem Identification

- **Problem**: Predict the likelihood of a patient developing diabetes based on health metrics.
- **Goal**: Build a classification model to predict diabetes (binary outcome: 1 for diabetic, 0 for non-diabetic).
- **Diabetes Prediction Dataset**

- **Overview**
  This dataset contains detailed medical diagnostic measurements aimed at predicting the onset of diabetes based on several health factors. The data comprises 768 records of female patients, with each record described by 8 health-related attributes. The `Outcome` variable indicates whether the patient has diabetes (1) or not (0). The dataset is suitable for training and testing machine learning models in classification tasks focused on diabetes prediction.

- **Dataset Details**

  - **Total Records**: 768
  - **Target Variable**: `Outcome` (0 or 1)
  - **Use Cases**:
    - Building classification models to predict diabetes onset.
    - Performing exploratory data analysis to discover trends and correlations among health indicators.
    - Feature engineering and selection for healthcare-related datasets.

- **Columns Description**

| Column                       | Description                                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Pregnancies**              | Number of times the patient has been pregnant                                                            |
| **Glucose**                  | Plasma glucose concentration after a 2-hour oral glucose tolerance test                                  |
| **BloodPressure**            | Diastolic blood pressure (mm Hg)                                                                         |
| **SkinThickness**            | Triceps skinfold thickness (mm)                                                                          |
| **Insulin**                  | 2-hour serum insulin (mu U/ml)                                                                           |
| **BMI**                      | Body mass index (weight in kg/(height in m)^2)                                                           |
| **DiabetesPedigreeFunction** | A function representing the patient’s diabetes pedigree (likelihood of diabetes based on family history) |
| **Age**                      | Age of the patient (years)                                                                               |
| **Outcome**                  | Binary outcome where 1 indicates diabetes presence and 0 indicates its absence                           |

- **Source**
  This dataset is adapted from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and is widely used in machine learning research on healthcare and medical diagnostics.

### 2. Exploratory Data Analysis (EDA)

- **Initial Analysis**:
  - Display summary statistics and basic dataset information.
  - Check for missing values.
- **Visualizations**:
  - Plot distributions for numerical columns (e.g., `Glucose`, `Age`).
  - Display a correlation heatmap to analyze relationships between features.
  - Create box plots to identify outliers in continuous variables.
- **Outcome Analysis**:
  - Visualize the balance of the target variable `Outcome`.

### 3. Data Preparation

- **Handle Missing Values**:
  - Replace zero values in relevant columns with `NaN` and impute them using median values.
- **Feature Scaling**:
  - Normalize or standardize numerical features.

### 4. Feature Engineering

- **Additional Features**:
  - Create `BMI_Category` based on BMI ranges.
- **Feature Importance**:
  - Use correlation and feature selection techniques to identify important features.

### 5. Model Training

- **Python environment**:
  ```console
  pip install -r requirements.txt
  ```
- **Models**:
  - Train models such as Logistic Regression, Decision Trees, and Random Forests.
  - See `mid_project_notebook.ipynb`
- **Evaluation Metrics**:

  - Use accuracy, precision, recall, F1-score, and AUC-ROC for evaluation.

  - `Class 0` indicates non-diabetes

  - `Class 1` means non-diabetes

  - Logistic Regression Classification Report {C=5}

    | Metric        | Class 0 | Class 1 | Average |
    | ------------- | ------- | ------- | ------- |
    | **Precision** | 0.93    | 0.53    | -       |
    | **Recall**    | 0.75    | 0.83    | -       |
    | **F1 Score**  | 0.83    | 0.65    | -       |
    | **Support**   | 83      | 29      | 112     |

    **Overall Metrics**:

    - **Accuracy**: 0.77
    - **Macro Avg Precision**: 0.73
    - **Macro Avg Recall**: 0.79
    - **Macro Avg F1 Score**: 0.74
    - **Weighted Avg Precision**: 0.82
    - **Weighted Avg Recall**: 0.77
    - **Weighted Avg F1 Score**: 0.78

    **Additional Metrics**:

    - **Precision (Class 1)**: 0.5454
    - **Recall (Class 1)**: 0.8276
    - **F1 Score (Class 1)**: 0.6575
    - **AUC-ROC Score**: 0.870

  - Decision Trees {max_depth=5, min_samples_leaf=10}

    | Metric        | Class 0 | Class 1 | Average |
    | ------------- | ------- | ------- | ------- |
    | **Precision** | 0.93    | 0.55    | -       |
    | **Recall**    | 0.76    | 0.83    | -       |
    | **F1 Score**  | 0.83    | 0.66    | -       |
    | **Support**   | 83      | 29      | 112     |

    **Overall Metrics**:

    - **Accuracy**: 0.78
    - **Macro Avg Precision**: 0.74
    - **Macro Avg Recall**: 0.79
    - **Macro Avg F1 Score**: 0.75
    - **Weighted Avg Precision**: 0.83
    - **Weighted Avg Recall**: 0.78
    - **Weighted Avg F1 Score**: 0.79

    **Additional Metrics**:

    - **Precision (Class 1)**: 0.5641
    - **Recall (Class 1)**: 0.7856
    - **F1 Score (Class 1)**: 0.6470
    - **AUC-ROC Score**: 0.8151

  - Random Forests {max_depth=5, min_samples_leaf=3, n_estimators=140}

    | Metric        | Class 0 | Class 1 | Average |
    | ------------- | ------- | ------- | ------- |
    | **Precision** | 0.83    | 0.86    | -       |
    | **Recall**    | 0.98    | 0.41    | -       |
    | **F1 Score**  | 0.90    | 0.56    | -       |
    | **Support**   | 83      | 29      | 112     |

    **Overall Metrics**:

    - **Accuracy**: 0.83
    - **Macro Avg Precision**: 0.84
    - **Macro Avg Recall**: 0.69
    - **Macro Avg F1 Score**: 0.73
    - **Weighted Avg Precision**: 0.83
    - **Weighted Avg Recall**: 0.83
    - **Weighted Avg F1 Score**: 0.81

    **Additional Metrics**:

    - **Precision (Class 1)**: 0.8571
    - **Recall (Class 1)**: 0.4138
    - **F1 Score (Class 1)**: 0.5581
    - **AUC-ROC Score**: 0.8799

    ### Model Selection Reasoning

    The **`Decision Tree Classifier`** is recommended as the best model due to its balance between recall and precision for the minority class (class `1`). This balance is reflected in the high F1 score for class `1`. Here’s why the Decision Tree stands out:

    - **`Balanced Recall and Precision`**: With a recall of 0.83 and a precision of 0.55 for class `1`, the Decision Tree captures a high proportion of true positives without generating too many false positives. This balance is ideal for scenarios where both precision and recall are important.

    - **`Interpretability`**: Decision Trees provide interpretability, making it easy to understand feature importance and the paths leading to each decision. This transparency is valuable when explaining the model’s decisions.

    - **`Comparable AUC-ROC`**: Although the AUC-ROC score for the Decision Tree is slightly lower than that of the Random Forest, it remains competitive, indicating that the model discriminates reasonably well between classes.

    While the **`Random Forest`** has the highest AUC-ROC and precision for class `1`, its low recall (0.41) may be inadequate if the goal is to identify as many positive cases as possible. The **`Logistic Regression`** model also performs well, but the Decision Tree provides a slightly better F1 score for class `1` and added interpretability, making it the more balanced choice overall.

- **Model Training Python Script**:
  ```console
  python train.py
  ```
  - Will save the model to `model.bin`

### 6. Export Notebook to Script

- **Python script**:
  - Local Service: Python script (`predict.py`).
  ```console
  python predict.py
  ```
  - Client Code: Python script (`prediction-test.py`).
  ```console
  python prediction-test.py
  ```

### 7. Model Deployment with Docker

- **Create a Web Service**:
  - Develop a Flask or FastAPI app (`predict.py`) to handle prediction requests.
- **Dockerize**:
  - Write a `Dockerfile` to set up the environment and run the service.
- **Run Locally**:
  - Build and run the Docker container to test the service locally.
  - Build docker file
  ```console
  docker build -t mid-pro-diabetes .
  ```
  - Run docker service
  ```console
  docker run  -it --rm -p 9696:9696 mid-proj-diabetes:latest
  ```

### 8. Testing Scenario 1: Local Service

- Open a terminal and run prediction local service: (`predict.py`).

```console
python predict.py
```

![Run local service](./images/01_local_service.jpg)

- Open another terminal and run local client: (`prediction-test.py`).

```console
python prediction-test.py
```

![Run local client to connect local service](./images/02_local_client_local_service.jpg)

- In first terminal, press CTRL+C to stop local service

### 9. Testing Scenario 2: Docker Service

- Open a terminal and build docker service: (`Dockerfile`).

```console
docker build -t mid-proj-diabetes .
```

![Build docker service](./images/03_build_docker_service.jpg)

- Run docker service in the same terminal

```console
docker run  -it --rm -p 9696:9696 mid-proj-diabetes:latest
```

![Run docker service](./images/04_run_docker_service.jpg)

- Modify code (Change `json=patient2` to `json=patient`in line 34)

![Code modification](./images/05_code_modification.jpg)

- Open another terminal. Run local client: (`prediction-test.py`).

```console
python prediction-test.py
```

![Run local client to connect docker service](./images/06_local_client_docker_service.jpg)
