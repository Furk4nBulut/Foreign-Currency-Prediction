
****************************************************************************************************************************
## License and Acknowledgments
ALL RIGHT BELONGS TO PROF. DR. MUHAMMET GÖKHAN CİNSDİKİCİ. Special thanks to Prof. Dr. Muhammet Gökhan Cinsdikici for developing the problem statement and ensuring its relevance.

## Problem Statement

**Objective**: Predict the value of a foreign currency using the values of other foreign currencies.

**Event Details**:
- **Date and Time**: December 5, 2023, 09:00 - 11:00
- **Institution**: Manisa Celal Bayar University
- **Event**: Artificial Neural Network Hackathon
- **Focus**: Predictive modeling without machine learning libraries.

**Acknowledgment**: This problem was developed by Prof. Dr. Muhammet Gökhan Cinsdikici.

****************************************************************************************************************************

# Foreign Currency Prediction Using Multi-Currency Analysis

## Abstract

This research explores the prediction of foreign currency values using other foreign currencies as input data. Conducted during the "Artificial Neural Network Hackathon" at Manisa Celal Bayar University, the challenge focused on developing predictive models without specialized machine learning libraries. This document presents a methodology for improving currency prediction accuracy by leveraging multi-currency relationships.

## Introduction
Foreign exchange markets play an important role in the 
global economy, serving as the cornerstone of international 
trade and investment. The ability to accurately predict foreign 
exchange values is of great importance to businesses, investors 
and policy makers. In this context, the challenge ahead is to 
develop effective models to predict currency movements. 
Specifically, the focus is on leveraging the relationships 
between various foreign currencies to estimate the value of the 
target currency. Our aim is to improve forecasting methods 
and deepen our understanding of the ever-changing global 
financial landscape by exploring the intricacies of predicting 
currency values through the relationships between different 
currencies.

# Methodology

## Data Preprocessing

This document outlines the data preprocessing steps applied to a financial dataset. The procedures encompass data loading, structural refinement, column renaming, and data quality assessments. The dataset was cleaned by removing unnecessary records, handling non-numeric values, and addressing missing data, culminating in a standardized, numeric representation suitable for analysis.

**Figure 1:** Sample Data From Given Dataset

### 1. Data Loading and Initial Inspection
The dataset was successfully loaded, and an initial inspection was conducted, with potential loading errors addressed through exception handling.

### 2. Cleaning the Dataset
- **Structural Refinement:** Descriptive initial rows were skipped, enhancing dataset clarity.
- **Column Renaming:** Columns were standardized for improved interpretability.
- **Record Removal:** Redundant information from the first row was dropped for dataset streamlining.

### 3. Data Quality Assessment
- **Identification of Non-Numeric and Missing Values:** Non-numeric columns and missing values were identified to gauge data quality.

### 4. Handling Non-Numeric and Missing Values
- **Non-Numeric Values:** Columns with non-numeric data types were converted to numeric, ensuring uniformity.
- **NaN Handling:** Rows with missing values were removed to eliminate noise in the analysis.

### 5. Data Type Standardization
- **Numeric Conversion:** Relevant columns were uniformly converted to numeric data types for consistency.

### 6. Final Data Quality Check
- **Post-Processing Data Quality:** A final check ensured the elimination of non-numeric values and missing entries.

### 7. Conclusion
The outlined preprocessing steps contribute to data quality, preparing the dataset for meaningful analysis by addressing structural inconsistencies, non-numeric values, and missing entries.

## Feature Selection

In the realm of predictive modeling, feature selection plays a pivotal role in crafting robust and efficient models. It involves identifying and retaining the most influential variables while discarding redundant or irrelevant ones. Its primary objectives include enhancing model performance, mitigating the risk of overfitting, and expediting model training.

### 1. Independent Variables (Features)
The following currency exchange rates were discerned as potential predictors:
- USD (TP DK USD S YTL)
- EUR (TP DK EUR S YTL)
- GBP (TP DK GBP S YTL)
- SEK (TP DK SEK S YTL)
- CHF (TP DK CHF S YTL)
- CAD (TP DK CAD S YTL)
- KWD (TP DK KWD S YTL)

### 2. Dependent Variable (Target)
The target variable, embodying the currency exchange rate for Saudi Riyal (SAR), was defined as:
- SAR (TP DK SAR S YTL) (Output)

### 3. Significance of Separation
The dataset was meticulously partitioned into independent variables (X) and the dependent variable (y). This segregation sets the stage for subsequent model training and evaluation.

### 4. Preview of Variables
A sneak peek into the initial five entries of both independent variables (X) and the dependent variable (y) is provided, offering a glimpse into the dataset's structure.

## Cost/Loss Function

The Mean Squared Error (MSE) is a fundamental metric in regression analysis, providing a quantitative measure of the accuracy of a predictive model. The MSE is computed by taking the average of the squared differences between actual and predicted values. This penalizes larger errors more heavily, offering a comprehensive assessment of the model's performance.

The `mean_squared_error` function presented here is a concise implementation of MSE calculation. It takes the true values (`y_true`) and the predicted values (`y_pred`) as inputs and returns the average of the squared differences. In the specific context of the code snippet, the MSE is calculated for a set of actual and predicted values stored in a DataFrame (`results_df`). The variables 'Actual' and 'Predicted' within the DataFrame correspond to the true and predicted values, respectively. The calculated MSE (`mse_all`) provides a consolidated measure of the model's performance across the entire dataset.

## Derivative of Cost/Loss Function

Understanding the derivative of the Mean Squared Error (MSE) function is crucial in machine learning optimization, especially in gradient descent-based algorithms. The derivative represents the slope or rate of change of the MSE with respect to the model parameters. The `derivative_mse` function provided here computes this derivative, facilitating the optimization of model coefficients.

### 1. Function Explanation
The `derivative_mse` function takes three parameters:
- `y_true` (true values)
- `y_pred` (predicted values)
- `X` (independent variables)

It employs a formula to compute the gradient of the MSE function with respect to the model parameters, providing insights into the direction and magnitude of adjustments needed during optimization.

### 2. Application to Data
In the code snippet, the `derivative_mse` function is applied to the actual and predicted values stored in the DataFrame (`results_df`). The result (`mse_derived`) signifies the gradient of the MSE function with respect to the model parameters for the given dataset.

### 3. Interpretation
The negative sign in the formula indicates the direction of steepest descent. The computed derivative guides the adjustment of model parameters, minimizing the MSE and refining the model's predictive capabilities.

## Optimizer Function

In the quest for robust machine learning models, regularization techniques become paramount to mitigate overfitting and enhance generalization. The `MultipleLinearRegressionRegularized` class presented here encapsulates a Multiple Linear Regression (MLR) model augmented with regularization, specifically a combination of L1 (Lasso) and L2 (Ridge) regularization.

### 1. Model Parameters
The model is equipped with several hyperparameters:
- `l1_ratio`: Proportion of L1 regularization
- `alpha`: Regularization strength
- `epochs`: Number of training iterations
- `learning_rate`: Step size for gradient descent

### 2. Training Process
The `fit` method employs gradient descent to iteratively update model coefficients. Noteworthy steps include:
- Adding a bias column to the independent variables (X)
- Initializing coefficients
- Iteratively updating coefficients using gradient descent
- Incorporating L1 and L2 penalties to mitigate overfitting

### 3. Regularization Mechanism
- **L1 Penalty:** Enforces sparsity in the coefficients by introducing the absolute values of the coefficients.
- **L2 Penalty:** Penalizes large coefficients to prevent overfitting.
- Both penalties are integrated into the gradient calculations and contribute to the iterative coefficient updates.

### 4. Predictions
The `predict` method utilizes the trained coefficients to make predictions. Similar to the training phase, a bias column is added to the input variables (X) before applying the dot product with the coefficients.

### 5. Implementation
The code snippet initializes and trains the regularized MLR model (`mlr_Regularized`) on the provided data (X, y). Predictions are then made on a subset of the training set (`X.values[:5]`).

### 6. Optimization Mechanism
Regularization aids in model optimization by constraining the coefficients, preventing them from becoming excessively large.

## Creating a Machine Learning Model

In machine learning, the development of predictive models is a cornerstone of data-driven decision making. The model is trained on a dataset comprising selected independent variables (features) and a corresponding dependent variable (target).

### 1. Model Architecture
The MLR model is encapsulated within the `MultipleLinearRegression` class, offering a modular and organized structure. The model's core attribute, `coefficients`, serves as a repository for the calculated regression coefficients.

### 2. Training the Model
The `fit` method is employed to train the model using the least squares method. This involves:
- Adding a bias column to the independent variables (X)
- Applying normal equations to determine the optimal coefficients (`beta`)

### 3. Making Predictions
The `predict` method utilizes the trained coefficients to make predictions for new data. Similar to the training phase, a bias column is added to the input variables (X) before applying the dot product with the coefficients.

### 4. Implementation
The code snippet initializes the MLR model, fits it to the training data (X, y), and subsequently makes predictions on a subset of the training set (`X.values[:5]`). The predicted values are stored in the `predictions` variable.

**Figure 2:** Multiple Linear Regression Predictions and Actual Values Graph

### 5. Insights from Predictions
The `predictions` variable encapsulates the model's output, providing insights into how well the trained MLR model performs on a subset of the training data.

## Model Evaluation and Testing

### 1. User Interaction
- Users are prompted to input values for each independent variable, creating a dynamic and personalized forecasting experience.
- The script ensures the user's input is of numeric type, promoting robustness and preventing potential errors.

### 2. Model Prediction
- The user-provided values are converted into a numpy array and used as input for the pre-trained MLR model.
- The model processes the input and generates a predicted output, representing the forecasted currency exchange rate.

### 3. Result Presentation
- The predicted output is then displayed to the user, offering insights into the anticipated currency exchange rate based on their input.

## Discussion

Although our current approach is promising, ensuring the robustness of our findings warrants exploration of alternative methods. An effective alternative involves utilizing time series analysis specifically designed for foreign exchange forecasting. This method allows for a nuanced examination of historical currency data over sequential time periods, capturing patterns and trends that our current multiple linear regression model does not fully capture. By adopting time series analysis, we improve our ability to account for temporal dependencies in currency fluctuations and potentially improve forecast accuracy


## License and Acknowledgments

ALL RIGHT BELONGS TO PROF. DR. MUHAMMET GÖKHAN CİNSDİKİCİ. Special thanks to Prof. Dr. Muhammet Gökhan Cinsdikici for developing the problem statement and ensuring its relevance.

## Contact

Furkan Bulut  
Manisa Celal Bayar University Faculty of Engineering  
Manisa, TURKEY  
[210316011@ogr.cbu.edu.tr](mailto:210316011@ogr.cbu.edu.tr)
