#!/usr/bin/env python
# coding: utf-8

# ## Building Predictive models to predict the target variable of the dataset
# 
# #### Machine Learning Algorithms:
# 
# - K-Nearest Neighbors
# - Decision Trees

# ### Import libraries

# In[1]:


import pandas as pd # Import pandas library for data manipulation and analysis
import numpy as np # Import numpy library for array operations
import matplotlib.pyplot as plt # Import pyplot module for creating visualizations

get_ipython().run_line_magic('config', 'Completer.use_jedi=False')


# ### Load the dataset

# In[2]:


# Read the CSV file into pandas DataFrame
data = pd.read_csv('risk_factors.csv')
data # Display the first few rows (5) of the dataset


# Based on the overview of the dataset above, there are <b>858 examples</b> and each example describes a patient with <b>32 features</b> and <b>4 target variables</b>. However, in this study, we will only take <b>'Biopsy'</b> as our target variable because the other 3 target variables namely 'Hinselmann', 'Schiller' and	'Citology' are non-invasive tests that a patient has to undergo before a Biopsy as well as to ease the processing. Therefore, there will be <b>35 features</b> and <b>1 target variable</b> which is 'Biopsy'.
# 
# However, not all the features are used in the algorithms as the importance of the features will also be considered differently for each algorithm. These features contribute to the target or predicted outcome, which is the result of Biopsy test and thereby confirming the presence/non-presence of cervical cancer in the patients. Therefore, our objective for this experiment is to forecast whether a patient is <b>associated with cervical cancer or not based on Biopsy test result</b>.

# ### Split the dataset
# The target column which is Biopsy is separated from the columns of features as a preparation before training, validation and testing the dataset.

# In[3]:


features = data.drop(['Biopsy'], axis=1)
X = features
y = data['Biopsy']


# In[4]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training set into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)


# ### Data preprocessing
# ______________________________________________________________________________________
# Description:
# 
# <b>Step 1: Missing Data Handling</b> 

# It can be seen that this dataset uses '?' to represent null values. Therefore, we will replace '?' with 'NaN' to ease the processing.

# In[5]:


# Replace '?' with null
X_train = X_train.replace('?', np.nan)
X_val = X_val.replace('?', np.nan)
X_test = X_test.replace('?', np.nan)


# After replacing with null, we count how many null values there are in each column of the dataset.

# In[6]:


X_val.isnull().sum()


# From the above step, it can be observed that the parameters <b>'STD: Time since the first diagnosis'</b> and <b>'STD: Time since last diagnoses'</b> had many null values, approximately 90% for all splitted data sets. Replacing these null values would make the classifier useless. Hence, these two features were <b>dropped</b> for each training, validation and test set.

# In[7]:


# Remove columns in training set
X_train = X_train.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)


# In[8]:


# Check if columns were removed
X_train.isnull().sum()


# In[9]:


# Remove columns in validation set
X_val = X_val.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)


# In[10]:


# Check if columns were removed
X_val.isnull().sum()


# In[11]:


# Remove columns in test set
X_test = X_test.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)


# In[12]:


# Check if columns were removed
X_test.isnull().sum()


# In[13]:


# Check the types of data for each column
X.info()


# Since there are multiple columns that have the 'object' data type, the values in the columns for each set were converted to numerical values to ease the processing.

# In[14]:


# Convert all features to numeric values

X_train = X_train.apply(pd.to_numeric)
X_val = X_val.apply(pd.to_numeric)
X_test = X_test.apply(pd.to_numeric)


# The columns in the dataset have both categorical and numerical attributes. Hence, we separate the categorical and numerical attributes for each set as they both require different processes/formulas in imputing the null values.

# In[15]:


# Separate the categorical and numerical values for each set

X_train_cat = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis',
               'STDs:cervical condylomatosis','STDs:vaginal condylomatosis',
               'STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
               'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV',
               'STDs:Hepatitis B','STDs:HPV','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology']
X_train_num = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies',
               'Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives (years)',
               'IUD (years)','STDs (number)','STDs: Number of diagnosis']

X_val_cat = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis',
            'STDs:cervical condylomatosis','STDs:vaginal condylomatosis',
            'STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
            'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV',
            'STDs:Hepatitis B','STDs:HPV','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology']
X_val_num = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies',
             'Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives (years)',
             'IUD (years)','STDs (number)','STDs: Number of diagnosis']

X_test_cat = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis',
            'STDs:cervical condylomatosis','STDs:vaginal condylomatosis',
            'STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
            'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV',
            'STDs:Hepatitis B','STDs:HPV','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology']
X_test_num = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies',
              'Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives (years)',
              'IUD (years)','STDs (number)','STDs: Number of diagnosis']


# <b>Descriptive statistics</b> was used to replace the missing values. The most typical metrics for this task are mean, median and mode. The median was used to replaced numerical attributes, and categorical attributes were replaced by the mode. Mean value imputation was avoided since they highly influence the extreme values/outliers in the data.

# In[16]:


# Replace null values of numerical attributes with the median
# Replace null values of categorical attributes with the mode

for feature in X_train[X_train_num]:
    X_train[feature].fillna((X_train[feature].median()), inplace=True)
    
for feature in X_train[X_train_cat]:
    X_train[feature].fillna((X_train[feature].mode()[0]), inplace=True)
    
for feature in X_val[X_val_num]:
    X_val[feature].fillna((X_val[feature].median()), inplace=True)
    
for feature in X_val[X_val_cat]:
    X_val[feature].fillna((X_val[feature].mode()[0]), inplace=True)
    
for feature in X_test[X_test_num]:
    X_test[feature].fillna((X_test[feature].median()), inplace=True)
    
for feature in X_test[X_test_cat]:
    X_test[feature].fillna((X_test[feature].mode()[0]), inplace=True)


# In[17]:


# Check null values again for training set
X_train.isnull().sum()


# In[18]:


# Check null values again for validation set
X_val.isnull().sum()


# In[19]:


# Check null values again for test set
X_test.isnull().sum()


# <b>Step 2: Feature Scaling/Normalisation</b>

# Feature scaling is a method used to normalize the range of independent variables or features of data. In this process, standard scalar was used to normalise the data. This is because it uses the standard normal distribution. All the means of the attributes are made zero, and the variance is scaled to one.

# In[20]:


from sklearn.preprocessing import StandardScaler

saved_cols = X_train.columns # Save column names for later use
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.fit_transform(X_val)
X_test_norm = scaler.fit_transform(X_test)


# In[21]:


# Insert the standardised values in a dataframe for each set

X_train = pd.DataFrame(X_train_norm, columns = saved_cols)
X_val = pd.DataFrame(X_val_norm, columns = saved_cols)
X_test = pd.DataFrame(X_test_norm, columns = saved_cols)

X_test


# #### Feature Selection
# ______________________________________________________________________________________
# <b>Step 1: Determine the feature importance with Tree Based Classifier</b>

# The importance of each feature is determined by using a Tree Based Classifier, namely the `Extra Trees Classifier`. The normalized total reduction in the mathematical criteria used in the decision of feature of split is computed. This value is called the Gini Importance of the feature.

# In[22]:


from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X_train,y_train)
print(model.feature_importances_)

# A graph of feature importances is plotted for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(33).plot(kind='barh')
plt.figure(figsize=(10, 8))
plt.show()


# From the above bar graph, we can see that there are 7 features that shows no importance. Hence, these 7 features are dropped from each set.

# In[23]:


# Drop columns that have 0 importance
X_train = X_train.drop(columns = ['STDs:cervical condylomatosis','STDs:AIDS','STDs:molluscum contagiosum',
                                  'STDs:Hepatitis B','STDs:pelvic inflammatory disease','STDs:vaginal condylomatosis',
                                  'STDs:HPV'])
X_val = X_val.drop(columns = ['STDs:cervical condylomatosis','STDs:AIDS','STDs:molluscum contagiosum',
                              'STDs:Hepatitis B','STDs:pelvic inflammatory disease','STDs:vaginal condylomatosis',
                              'STDs:HPV'])
X_test = X_test.drop(columns = ['STDs:cervical condylomatosis','STDs:AIDS','STDs:molluscum contagiosum',
                                'STDs:Hepatitis B','STDs:pelvic inflammatory disease','STDs:vaginal condylomatosis',
                                'STDs:HPV'])


# <b>Step 2: Use Pearson’s correlation feature selection technique</b>

# Pearson’s correlation feature selection technique was utilized to find redundant features. This feature selection technique compares the degree of association among all variables. When there is a high correlation between two independent attributes, one of these attributes can be removed since both features contribute the same to the ML model.

# In[24]:


# Find correlation

corr = X_train.corr()


# A Pearson’s correlation heat map is plotted below.

# In[25]:


import seaborn as sns

plt.figure(figsize=(20, 8))
heatmap = sns.heatmap(corr, vmin=0, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


# Based on this heatmap, we can drop the features that have a high correlation with a value higher than 0.8.

# In[26]:


# Get upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# Find features with correlation greater than 0.80
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop highly correlated features
X_train.drop(to_drop, axis=1, inplace=True)
print (X_train)

X_val.drop(to_drop, axis=1, inplace=True)
print (X_val)

X_test.drop(to_drop, axis=1, inplace=True)
print (X_test)


# ### Data modeling
# ______________________________________________________________________________________
# <b>Model 1: k-Nearest Neighbours</b>

# The code begins by importing the `KNeighborsClassifier` class from the `sklearn.neighbors` module. It then initializes variables for storing classification scores (`scores`), the maximum score found (`max_score`), and the corresponding value of k (`best_k`). 
# 
# The subsequent loop tests different values of k from 1 to 20 for the k-nearest neighbors algorithm. For each value of k, a `KNeighborsClassifier` object is created, fitted to the training data, and used to calculate the classification accuracy on the validation data. If the current score is higher than the previous maximum score, the maximum score and corresponding value of k are updated. The scores are recorded in a list. 
# 
# Finally, a line graph is plotted to visualize the scores for different values of k, and the maximum score and best k value are printed.

# In[27]:


from sklearn.neighbors import KNeighborsClassifier

scores = []
max_score = 0
best_k = 0
for k in range(1,20):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    score = model_knn.score(X_val, y_val)
    if score > max_score:
        max_score = score
        best_k = k
    scores.append(score)
plt.plot(np.arange(1,20), scores)
plt.xticks(np.arange(1,20))
plt.grid()
plt.show()
print(max(scores))
print(best_k)


# In[28]:


# The model is then used to predict the test set
model_knn = KNeighborsClassifier(n_neighbors=best_k)
model_knn.fit(X_train, y_train)
y_pred_knn_test = model_knn.predict(X_test)
y_pred_knn_train = model_knn.predict(X_train)


# The F1 scores between the training and testing datasets are compared to prevent overfitting. In this case, overfitting will happen if the F1 score of training dataset is 1.0 or its value is too high as compared to the testing dataset.

# In[29]:


# Compare the F1 scores between the training and testing datasets
from sklearn import metrics

test_f1_score = metrics.f1_score(y_test, y_pred_knn_test)
train_f1_score = metrics.f1_score(y_train, y_pred_knn_train)

print('F1 Score (test):', test_f1_score)
print('F1 Score (train):', train_f1_score)


# From the comparison between both F1 scores, we can say that the data is not overfitting.

# <b>Model 2: Decision Tree</b>

# The code for Decision Tree modelling begins by importing the `DecisionTreeClassifier` class from the `sklearn.tree` module. We then create an empty list to store the models (`dt_model`). We will test 5 depths (`depths`) to see which depth will produce the best models by looping the models starting from a tree with depth 3 until until depth 7. The model is fit to the training data and will append them to the list (`dt_models`).
# 
# To validate our models, the accuracy of the current models are calculated using the validation data. It iterates each model to check whether the current accuracy is better then the previous accuracy and update it to the best accuracy (`best_acc`) and its corresponding model (`model_best`). The decision tree models and its depth are printed to showcase the models accuracy corresponding to the model.
# 
# Finally, using the best model, a decision tree graph is plotted. From our project, we determine that a decision tree with depth 3 produces the highest accuracy and is determined as the best model.

# In[30]:


from sklearn.tree import DecisionTreeClassifier

# Create an empty list to store the models
dt_models = []

# The depths will start at 3 and increment by 1 until it reaches 8
# In this model, the depths determined are 3, 4, 5, 6, 7
depths = np.arange(3, 8, 1)

# Iterate over each depth value
for d in depths:
    # Create a decision tree model with specified parameters
    model = DecisionTreeClassifier(criterion='gini', max_depth=d, random_state=42)
    model.fit(X_train, y_train) # Fit the model to the training data
    dt_models.append(model) # Append the model to the list


# In[31]:


best_acc = 0
model_best = None

# Iterate over each decision tree model and its corresponding depth
for m,d in zip(dt_models, depths):
    acc = m.score(X_val, y_val) # Calculate the accuracy of the current model on the validation data
    
    # Print the accuracy of the current model
    print(f'Decision Tree classifier with max_depth={d} achieves a mean accuracy of {acc}')
    
    # Check if the current accuracy is better than the previous best accuracy
    if acc > best_acc:
        # Update the best accuracy and the corresponding best model
        # For our data, the best accuracy obtained is when the tree has a depth of 3
        best_acc = acc
        model_best = m


# In[32]:


from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

# Set the figure size for the plot
plt.figure(figsize=(50, 30))

# Plot the decision tree using the best model (depth: 3)
# feature_names parameter is used to label the tree nodes with feature names
plot_tree(model_best, feature_names=list(X.columns))
plt.show()


# In[33]:


# The model is then used to predict the test set
y_pred_dt_test = model.predict(X_test)
y_pred_dt_train = model.predict(X_train)


# As mentioned above, the F1 scores between the training and testing datasets are compared to prevent overfitting. In this case, overfitting will happen if the F1 score of training dataset is 1.0 or its value is too high as compared to the testing dataset.

# In[34]:


from sklearn import metrics

test_f1_score = metrics.f1_score(y_test, y_pred_dt_test)
train_f1_score = metrics.f1_score(y_train, y_pred_dt_train)

print('F1 Score (test):', test_f1_score)
print('F1 Score (train):', train_f1_score)


# ### Model evaluation
# ______________________________________________________________________________________
# Each of the model will be evaluated via the performance metrics namely F1 Score, Accuracy, Precision, Recall and Specificity.

# #### 1. k-Nearest Neighbours 

# In[35]:


from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

print('k-Nearest Neighbours')
print('----------------------')
print('F1 Score: %.3f' % f1_score(y_test, y_pred_knn_test))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_knn_test))
print('Precision: %.3f' % precision_score(y_test, y_pred_knn_test))
print('Recall: %.3f' % recall_score(y_test, y_pred_knn_test))
print('Specificity: %.3f' % recall_score(y_test, y_pred_knn_test, pos_label = 0))


# In[36]:


# Print the confusion matrix using Matplotlib
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_knn_test)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix: kNN', fontsize=18)
plt.show()


# #### 2. Decision Tree

# In[37]:


from sklearn.metrics import f1_score,precision_score,recall_score
print('Decision Tree')
print('----------------------')
print('F1 Score: %.3f' % f1_score(y_test, y_pred_dt_test))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_dt_test))
print('Precision: %.3f' % precision_score(y_test, y_pred_dt_test))
print('Recall: %.3f' % recall_score(y_test, y_pred_dt_test))
print('Specificity: %.3f' % recall_score(y_test, y_pred_dt_test, pos_label = 0))


# In[38]:


# Print the confusion matrix using Matplotlib
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_dt_test)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix: Decision Tree', fontsize=18)
plt.show()


# As displayed and evaluated via the performance metrics above, the justifications are as below:
# 
# - The <b>F1 score</b> using k-Nearest Neighbour algorithm (0.556) is <b>lower</b> than of the decision tree algorithm (0.714). Since the closer the value of F1 score to the value 1, denotes a better classifier, hence the decision tree algorithm is performs better in this case. In this case, based on the formulae, F1-score is more sensitive to false negative and will penalize models that produce too many false negatives as compared to accuracy.
# 
# - The <b>accuracy</b> of both k-Nearest Neighbour algorithm and decision tree algorithm are similar (0.953) which also shows that they both have a high value of accuracy.
# 
# - The <b>precision</b> using k-Nearest Neighbour algorithm (0.714) is <b>higher</b> than of the decision tree algorithm (0.588). Eventhough the higher precision values denote a better performance of algorithm, but in this case, it does not because it includes the number of false positives, whereby the model incorrectly labels as positive that are actually negative, hence the person who will be needing biopsy will be at risk of not receiving one. Thus, the decision tree algorithm works best for this scenario.
# 
# - The <b>recall</b> using k-Nearest Neighbour algorithm (0.455) is <b>lower</b> than of the decision tree algorithm (0.909). Recall is the measure of the model correctly identifying True Positives whereby it predicts correctly the patients who needs a biopsy as compared to the actual scenario. Hence, the decision tree algorithm works best in predicting the patients and their actualness of needing a biopsy.
# 
# - The <b>specificity</b> using k-Nearest Neighbour algorithm (0.988) is <b>slightly higher</b> than of the decision tree algorithm (0.957). Specificity mentions about how negative records are correctly predicted. Hence, with the high specificity of both the models, it will help in evaluating which patient does not need a biopsy correctly.
# 
# Among all the performane metrics, <b>F1-score</b> should be prioritised to compare and determine the most optimal algorithm in this dataset. To justify this decision, firstly, accuracy is not suitable to be used because it is only optimal for classes which are balanced and there is no serious flaw to predicting false negatives (FN). In this case, the <b>Decision Tree</b> model has a <b>higher</b> value of F1 score (0.714) compared to the kNN model (0.556). 
# 
# Additionally, the <b>Decision Tree</b> model also has the <b>lowest</b> value of <b>False Negatives</b> (1) compared to the kNN model (6). This is significant when related to the human health. As an example, if a patient is suffering from cervical cancer, but the trained model predicts that the patient does not have cervical cancer, it would result in false negative which is bad as the sick patient is predicted to be healthy and ends up not receiving cervical biopsy. Therefore, the more the false negatives, ultimately it will cause more patient to not undergo a biopsy when needed.
# 
# In conclusion, the <b>Decision Tree</b> model is our champion model.
# 
