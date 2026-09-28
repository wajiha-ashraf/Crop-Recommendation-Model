# Crop Recommendation System

## 📌 Project Overview

This project uses machine learning to recommend a suitable crop based on soil and environmental conditions.

The problem is treated as a **multi-class classification problem**, where the model predicts one of 22 possible crop classes from the given input features.

The project covers data inspection, preprocessing, feature scaling, Random Forest model training, evaluation, visualization, and a sample crop recommendation.

---

## 📊 Dataset

The project uses the dataset:

`Crop_Recommendation.csv`

### Dataset Size

- **Rows:** 2,200
- **Columns:** 8
- **Target:** `Crop`

### Input Features

| Feature | Description |
|---|---|
| `Nitrogen` | Nitrogen content |
| `Phosphorus` | Phosphorus content |
| `Potassium` | Potassium content |
| `Temperature` | Temperature |
| `Humidity` | Humidity |
| `pH_Value` | Soil pH value |
| `Rainfall` | Rainfall |
| `Crop` | Target crop class |

The dataset contains **22 crop classes**, with 100 samples for each class.

### Crop Classes

- Rice
- Maize
- ChickPea
- KidneyBeans
- PigeonPeas
- MothBeans
- MungBean
- Blackgram
- Lentil
- Pomegranate
- Banana
- Mango
- Grapes
- Watermelon
- Muskmelon
- Apple
- Orange
- Papaya
- Coconut
- Cotton
- Jute
- Coffee

---

## 🔍 Data Inspection

The notebook performs the following initial checks:

- `head()`
- `tail()`
- `describe()`
- Data types
- Dataset shape
- Crop class distribution
- Missing-value check
- Class proportions
- Unique crop classes

### Missing Values

The dataset contains **no missing values** in any of the 8 columns.

### Class Distribution

Each of the 22 crop classes contains **100 samples**, making the dataset evenly distributed across the target classes.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Target Encoding

The categorical `Crop` target was converted into numerical labels using:

```python
LabelEncoder()
```

### 2. Feature and Target Separation

```python
X = df.drop('Crop', axis=1)
y = df['Crop']
```

### 3. Train-Test Split

The dataset was divided using:

- **80% training data**
- **20% testing data**
- `random_state=42`

This resulted in:

- Training set: 1,760 samples
- Test set: 440 samples

### 4. Feature Scaling

`StandardScaler` was applied to the input features.

The scaler was fitted on the training data and then used to transform both training and test data.

---

## 🤖 Machine Learning Model

A **Random Forest Classifier** was used.

The model was configured with:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The model was then trained using the scaled training data.

---

## 📈 Model Performance

The model achieved:

### **Test Accuracy: 99.32%**

The test set contained **440 samples**.

The classification report produced:

- **Macro average F1-score:** 0.99
- **Weighted average F1-score:** 0.99

Most crop classes achieved precision, recall, and F1-scores of 1.00. A few classes had slightly lower values.

---

## 📊 Visualizations

The notebook includes the following visualizations:

### 1. Confusion Matrix

A confusion matrix was generated to inspect actual versus predicted crop classes.

### 2. Feature Importance

Random Forest feature importance was visualized to examine the contribution of the input features to the model.

### 3. Crop Distribution

A bar chart was created to visualize the distribution of the 22 crop classes.

### 4. Feature Correlation Heatmap

A correlation heatmap was generated to examine relationships between the numerical features.

### 5. ROC Curves

ROC curves were generated for the 22 encoded crop classes using a one-vs-rest approach.

---

## 🌱 Sample Prediction

A sample input was provided to the trained model:

```python
[90, 60, 43, 20.8, 82.0, 6.5, 39]
```

The model returned:

```text
Recommended Crop: Watermelon
```

The encoded prediction was then converted back to the original crop name using `LabelEncoder`.

---

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Scikit-learn Components Used

- `LabelEncoder`
- `StandardScaler`
- `train_test_split`
- `RandomForestClassifier`
- `classification_report`
- `confusion_matrix`
- `roc_curve`
- `auc`
- `label_binarize`

---

## 📂 Project Structure

```text
Crop-Recommendation/
│
├── Crop_Recommendation.csv
├── main.ipynb
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Open the notebook

```bash
jupyter notebook
```

### 4. Run `main.ipynb`

Make sure `Crop_Recommendation.csv` is available in the project directory.

---

## 🎯 Project Objective

The objective of this project is to build a machine learning classification model that can recommend a crop based on soil nutrients and environmental conditions.

The project demonstrates a complete workflow from **data exploration → preprocessing → model training → evaluation → visualization → prediction**.

---

## 👩‍💻 Author

**Wajiha Ashraf**

BS Computer Science | AI/ML & Data Science Enthusiast
