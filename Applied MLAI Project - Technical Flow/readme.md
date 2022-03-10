## Part 1: Preliminary Data Analysis
 
 This task comprises three primary subtasks. 

#### Task 1: Labeling, Assessment and Data segment Selection

If label is not available, among various ways of labeling, clarify one. Observation target ratio among various customer groups to assess the feasibility and choose the potential customer group with equivalent dataset used for building machine learning (ML) model.

#### Task 2: Feature Exploring and Research Model Define

Feature Exploring: Based on the understanding of stakeholder about the business, referring relating research/scientific papers, defining the suitable independence variables (features) which might deliver the effect to dependent variable (target/label).

#### Task 3: Exploratory Data Analysis

Collect data of all above metrics, implementing EDA with below analysis.

**Missing value**: define the suitable way to deal with unknown or missing data at any column. It could choose to fill special value into missing location to mark it. Hot-desk imputation technique could be considered. 

**Duplicate data**: remove the duplicate data, just keep one for the training dataset. For the prediction dataset, depending on the data meaning, the duplicate could be removed or kept.

**Data Adjustment**: to be ensure all data of particular metric is same format, and unique categorical value for particular meaning.

**Anomaly/Outlier detection**: figure out the data points which are significant different with the other normal ones. 

For single or small number of data features, the statistic concepts such as normal distribution 3-sigma, or boxplot 1.5*IQR outliers detection technique could be used.
For the data with large number of metrics, the unsupervised learning method could be more advantageous, such as density-based method (DBScan Clustering), or Isolation Forest.

**Data visualization**: for data understanding, get insights and support for above tasks and feature engineering in the next step.

**Univariate analysis**:
   *	Categorical (Frequency Table, Bar Chart, Pie Chart)
   *	Quantitative (Histogram, Boxplot, Outliers)

## Part 2: Feature Engineering

Multivariate analysis: 

   *	Feature correlation matrix performs the direction and strength of the relationship among features.
   *	Relationship of all features and label expressed by multiple scatter plot, colored by label.  

Based on the data understanding and insights getting from EDA and multivariate analysis, feature engineering applies different methods to select, transform, and provide final package of [input features, label] which is used as input data to train the ML model. 

Below is some issues, and recommend techniques for solving.

**Feature selection**: features which have correlation with label. The logical explained relationship between features and label was considered already in Part1.Task2.
**Feature reduction**: Information Value (IV), Principal Component Analysis (PCA).
**Feature crossing**: create new feature based on transformation of available features.
**Data Normalization/Standardization/Scaling**: transform data features to approximate normal distribution, standard distribution or scale down data value. It is the final transformation step before putting data to model to train. Technique: min-max feature scaling, z-scoring.
**Imbalanced Data process**: if the label data is skew, especially in almost logistic regression classification quest, the imbalance is usually significant. Undersampling and Oversampling technique could be considered as the solution for imbalanced issue. For Oversampling, recommend BorderlineSMOTE algorithm.

## Part 3: Build, Train, and Predict ML model

#### Task 1: Train Data Preparation

Investigate in preparing data for training model. Following that, data will be divided into 3 groups: train, valid and test.
Train and valid data are used concurrently in train period, as known as cross validation. This is great technique to protect the model from the overfitting.
The test data is used to evaluate model independently when training completed.
The important here is to be ensure train, valid, and test group keeping the same rates among features and label inside. This is known as stratified data splitting.

#### Task 2: Model Structure

Based on the issue that need solving, the appropriate ML algorithm is selected.
They could be linear regression, classification, or clustering model. The estimator with relevant hyper-parameters is configured in this task. Technique: GridSearchCV supports hyper-parameter selection, and StratifiedKFold for cross validation.
Deep learning also could be used for advance. 
The structure of Neural Network need defining, includes number of layers, number of nodes in layers, type of nodes (Dense, LSTM, Conv1D, etc.) and kind of activation.
Hyper-parameters configure: epoch number, batch size, learning rate, loss function, optimizer, evaluating metric, early stop, cross validation.

#### Task 3: Model Fit and Evaluation

Fitting and the model is train. 
Using test data to evaluate the model when training completed.
If evaluation result is good, saving model. If not, adjust model structure, or hyper-parameter and retrain. If the test result still is not good, review Part 1, Part 2.

#### Task 4: Predict Trained ML Model 

Prediction is the operation of using trained model. The new input data will be fed into trained model to get the forecast result. Refer below figure for more information.

## Coding Flow

The diagram below shows the coding flow to implement the ML project. The flow includes two separate processes, TRAIN and PREDICT. The TRAIN process has responsibility for training model, evaluation and save versions, whereas PREDICT process loads trained model for using. 
