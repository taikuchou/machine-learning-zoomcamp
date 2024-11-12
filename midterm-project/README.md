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
  - Tuning: Use GridSearchCV to tune hyperparameters.
  - See `mid_project_notebook.ipynb`
- **Evaluation Metrics**:

  - Use accuracy, precision, recall, F1-score, and AUC-ROC for evaluation.

  - `Class 0` indicates non-diabetes

  - `Class 1` means non-diabetes

  - Logistic Regression Model Report

    - Best Hyperparameters

      - `C`: 10
      - `class_weight`: balanced
      - `penalty`: l1

    - Classification Report

      | Metric        | Class 0 | Class 1 | Average |
      | ------------- | ------- | ------- | ------- |
      | **Precision** | 0.93    | 0.56    | -       |
      | **Recall**    | 0.77    | 0.83    | -       |
      | **F1 Score**  | 0.84    | 0.67    | -       |
      | **Support**   | 83      | 29      | 112     |

    - Overall Metrics:

      - **Accuracy**: 0.79
      - **Macro Avg Precision**: 0.74
      - **Macro Avg Recall**: 0.80
      - **Macro Avg F1 Score**: 0.75
      - **Weighted Avg Precision**: 0.83
      - **Weighted Avg Recall**: 0.79
      - **Weighted Avg F1 Score**: 0.80

    - Class 1 Metrics:

      - **Precision**: 0.5581
      - **Recall**: 0.8276
      - **F1 Score**: 0.6667
      - **AUC-ROC Score**: 0.8695

    - Interpretation

      The Logistic Regression model with these hyperparameters achieves a balanced performance, with a high recall for class `1` (0.83) and a good AUC-ROC score, suggesting effective discrimination between classes. This model is well-suited for identifying positive cases (class `1`) in the dataset.

  - Decision Trees Classifier Model Report

    - Best Hyperparameters

      - `class_weight`: balanced
      - `criterion`: gini
      - `max_depth`: 10
      - `min_samples_leaf`: 4
      - `min_samples_split`: 2

    - Classification Report

      | Metric        | Class 0 | Class 1 | Average |
      | ------------- | ------- | ------- | ------- |
      | **Precision** | 0.82    | 0.43    | -       |
      | **Recall**    | 0.76    | 0.52    | -       |
      | **F1 Score**  | 0.79    | 0.47    | -       |
      | **Support**   | 83      | 29      | 112     |

    - Overall Metrics:

      - **Accuracy**: 0.70
      - **Macro Avg Precision**: 0.62
      - **Macro Avg Recall**: 0.64
      - **Macro Avg F1 Score**: 0.63
      - **Weighted Avg Precision**: 0.72
      - **Weighted Avg Recall**: 0.70
      - **Weighted Avg F1 Score**: 0.70

    - Class 1 Metrics:

      - **Precision**: 0.4286
      - **Recall**: 0.5172
      - **F1 Score**: 0.4688
      - **AUC-ROC Score**: 0.6610

    - Interpretation

      The Decision Tree Classifier with these hyperparameters achieves moderate performance, with a focus on balanced class weights. While it has good precision and recall for class `0`, the performance for class `1` is lower, which might impact its ability to detect positive cases. The AUC-ROC score of 0.6610 suggests limited discriminative power between classes, indicating that further tuning or an alternative model might improve results.

  - Random Forests Classifier Model Report

    - Best Hyperparameters

      - `class_weight`: balanced
      - `max_depth`: 10
      - `min_samples_leaf`: 4
      - `min_samples_split`: 10
      - `n_estimators`: 100

    - Classification Report

      | Metric        | Class 0 | Class 1 | Average |
      | ------------- | ------- | ------- | ------- |
      | **Precision** | 0.91    | 0.58    | -       |
      | **Recall**    | 0.81    | 0.76    | -       |
      | **F1 Score**  | 0.85    | 0.66    | -       |
      | **Support**   | 83      | 29      | 112     |

    - Overall Metrics:

      - **Accuracy**: 0.79
      - **Macro Avg Precision**: 0.74
      - **Macro Avg Recall**: 0.78
      - **Macro Avg F1 Score**: 0.76
      - **Weighted Avg Precision**: 0.82
      - **Weighted Avg Recall**: 0.79
      - **Weighted Avg F1 Score**: 0.80

    - Class 1 Metrics:

      - **Precision**: 0.5789
      - **Recall**: 0.7586
      - **F1 Score**: 0.6567
      - **AUC-ROC Score**: 0.8538

    - Interpretation

      The Random Forest Classifier with these hyperparameters performs well, with high recall (0.76) and a good balance between precision and recall for class `1`. This model also achieved a strong AUC-ROC score of 0.8538, indicating good discriminative power. Its ability to handle class imbalance with balanced class weights makes it suitable for identifying positive cases in the dataset.

    ### Model Selection Reasoning

    #### Summary of Model Performance

    | Model                     | Class 1 Precision | Class 1 Recall | Class 1 F1 Score | Accuracy | AUC-ROC |
    | ------------------------- | ----------------- | -------------- | ---------------- | -------- | ------- |
    | **`Logistic Regression`** | `0.56`            | `0.83`         | `0.67`           | `0.79`   | `0.870` |
    | **Random Forest**         | 0.58              | 0.76           | 0.66             | 0.79     | 0.854   |
    | **Decision Tree**         | 0.43              | 0.52           | 0.47             | 0.70     | 0.661   |

    - **Best Model**: **`Logistic Regression`** is recommended if high recall and strong overall performance are priorities. With the highest recall and AUC-ROC score, it effectively captures positive cases while maintaining good precision.
    - **Alternative Choice**: **Random Forest Classifier** is a strong alternative if a more balanced precision-recall trade-off is desired. It has a comparable F1 score and strong accuracy, with added robustness from the ensemble approach.

    - **Least Preferred Model**: **Decision Tree Classifier** is the least suitable choice due to lower performance across all metrics compared to Logistic Regression and Random Forest.

    ### Conclusion

    Overall, **`Logistic Regression`** is the recommended model due to its high recall, solid F1 score, and the highest AUC-ROC score, providing effective balance and discriminative power. **Random Forest** can be considered as an alternative for more balanced performance, while **Decision Tree** is not recommended due to weaker performance.

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
