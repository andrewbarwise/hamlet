# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:10:00 2023

@author: Christine Sheldon

Machine Learning and AI for Humanities
Week 2 - Class 2: Supervised Classification
In class tutorial and assignment

### CODING TUTORIAL

Today we will apply two classification algorithms, visualize their performance, and calculate their accuracy. 

Data to be used: The Library of Garfunkel - Cleaned & Merged

"""
#%% Importing necessary libraries

import os # working directory management
import pandas as pd # dataframe management
import numpy as np # Numers Library 
from sklearn.model_selection import train_test_split # Function to split the data into training and test subsets
from sklearn.preprocessing import StandardScaler # Function to scale variables
from sklearn.neighbors import KNeighborsClassifier # KNN Classifier
import matplotlib.pyplot as plt # Plotting Data
from sklearn.linear_model import LogisticRegression # Logistic Regression Classifier
from sklearn.metrics import accuracy_score # Function to calculate accuracy scores
from sklearn.utils import resample # Function to duplicate observations, used to balance the data

#%% Set working directory (wd) and Load the data 

os.getcwd()                                                                     # What is the current working directory? 
os.chdir(r"C:\Users\Christine Sheldon\Dropbox (Nuffield College)\Oxford\Teaching\ML & AI for Humanities\Week 2") # Set the directory to the location the data is stored
data = pd.read_csv('data_merged.csv')                                           # load the .csv file into a panda's dataframe

#%% Balancing the data

# a dataset is considered balanced, when there are equally as many observations per class (another word for category or label). 
# If a dataset is unbalanced, it means that there is much more information (more rows) for some classes, when compared to others. 
# Since more information == better classification, this is an issue: some classes will classify better than others. 
# As a result, this needs to be addressed before running the classification algorithm.  

# Investigate class balance
print(data['Favorite'].value_counts())

# One way to fix this, is to upsample the minority class
# first split the favourite and non-favourite books into separate datasets
minority_class = data[data['Favorite'] == 1]
majority_class = data[data['Favorite'] == 0]

# Then, resample the minority, favourite dataset. Here we are randomly duplicating rows until it is the same length as the non-favourites group
upsampled_minority = resample(minority_class,
                             replace=True,
                             n_samples=len(majority_class),
                             random_state=42) # We set a random_state, or a seed, to ensure replication is consistent. 

data_balanced = pd.concat([majority_class, upsampled_minority]) # Then we add the resampled minority, and the original majority class back together again. Each contributing 516 rows
print(data_balanced['Favorite'].value_counts())

# This does inflate the information in the minority class (i.e. add more variation)
# Other strategies include synthetic minority over sampling (SMOTE) or specifing unbalanced data in the classifier model itself

#%% Prepare balanced data for classification: Scaling

# Separate our features (X) from our labels (y)
X = data_balanced[['Pages', 'year_published']]
y = data_balanced['Favorite']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the variables: It's generally a good idea to scale the features for kNN especially
# Note that we scale our training and test data seperately: this is crucial to prevent data leakage. 
# Data leakage occurs when information from the test set is inadvertently used to inform the training process, 
# leading to overoptimistic estimates of model performance.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Why do we scale: Some algorithms can be sensitive to the scale of the input features, especially those that rely on optimization algorithms 
# to find the best fit (like gradient descent). Having features on different scales can lead to longer convergence times or convergence to suboptimal solutions. 
# Scaling can help with faster and more stable convergence.


#%% Apply k-nearest neighbors classifier

knn = KNeighborsClassifier(n_neighbors=3)                                       # Specify the model, using 3 neighbors for this example
knn.fit(X_train_scaled, y_train)                                                # Fit the model on the scaled X variables, and the y variable of the training data. 

# Predict the outcome using the specified model both for train and test data
y_train_pred_knn = knn.predict(X_train_scaled)
y_test_pred_knn = knn.predict(X_test_scaled)

# Calculate training and testing accuracy for kNN
train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)

print(f"kNN Training Accuracy: {train_accuracy_knn}")
print(f"kNN Testing Accuracy: {test_accuracy_knn}")

# If the test and training accuracy are too close, it is an indication the model is underfitting. 
# The accuracy on the test data should always be lower than the accuracy on the training data

#%% How many neighbours do we need? 

# Let's iterate through 10 different kNearest Neighbours models and increase the number of neighbours by one each time
# For each of these models we will save the train and test accuracy and then plot these together.  

# Create empty lists to hold accuracies
train_accuracies = []
test_accuracies = []

# Set the number of neighbours to check
max_neighbours = 10

# Looping through and fitting KNN classifiers
for neighbors in range(1, max_neighbours + 1):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train_scaled, y_train)
    train_accuracy = knn.score(X_train_scaled, y_train)
    test_accuracy = knn.score(X_test_scaled, y_test)
    
    # Append accuracies to lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Lets plot how our two types of accuracy vary across the number of neighbours
plt.figure(figsize=(10, 7))
plt.plot(range(1, max_neighbours + 1), train_accuracies, marker='o', label='Training Accuracy') # Plotting training accuracy
plt.plot(range(1, max_neighbours + 1), test_accuracies, marker='o', linestyle='--', color='red', label='Test Accuracy') # plotting test accuracy
plt.xlabel('Number of Neighbors')                                               # Adding labels & title
plt.ylabel('Accuracy')
plt.title('KNN Classifier: Training vs. Test Accuracy')
plt.legend()                                                                    # Add a legend indicating which colour is which accuracy
plt.xticks(np.arange(1, max_neighbours + 1))                                    # Set number of X-axis tick labels
plt.grid(True)                                                                  # Add a grid for easy interpretations
plt.show()


#%% Extra Code if interested: 
# Creating a grid of values to evaluate the classifier's decision boundary
# You do not need to know this code, but are very welcome to use it!

x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the outcome using the specified model both for train and test data
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(10, 7))

# Plotting the decision boundary (contour)
plt.contourf(xx, yy, Z, alpha=0.3)

# Plotting the original data points
scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, edgecolor='k', s=100)
legend1 = plt.legend(*scatter.legend_elements(), loc="upper left", title="Favorite")
plt.gca().add_artist(legend1)
plt.xlabel('Pages')
plt.ylabel('Year Published')
plt.title('KNN Decision Boundary for Favourites')
plt.show()

#%% Let's Apply a logistic regression classifier too
logreg = LogisticRegression()                                                   # Specify the model. Some things to add: solver = 'liblinear', random_state = 45
logreg.fit(X_train_scaled, y_train)                                             # Fit the model on the scaled X variables, and the y variable of the training data. 

# Predict the specified model on both train and test data
y_train_pred_logreg = logreg.predict(X_train_scaled)
y_test_pred_logreg = logreg.predict(X_test_scaled)

# Calculate training and testing accuracy for logistic regression
train_accuracy_logreg = accuracy_score(y_train, y_train_pred_logreg)
test_accuracy_logreg = accuracy_score(y_test, y_test_pred_logreg)

print(f"Logistic Regression Training Accuracy: {train_accuracy_logreg}")
print(f"Logistic Regression Testing Accuracy: {test_accuracy_logreg}")

#%% Extra code: Plot logistic regression decision boundary as well

# Create a grid of values to evaluate the classifier's decision boundary
x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Get the decision boundary (where probability is 0.5)
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision line within the scatterplot. 
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3) 
scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, edgecolor='k', s=100)
legend1 = plt.legend(*scatter.legend_elements(), loc="upper left", title="Favorite")
plt.gca().add_artist(legend1)
plt.xlabel('Pages')
plt.ylabel('Year Published')
plt.title('Logistic Regression Decision Boundary with Original Data Points')
plt.show()

#%% How certain is the model about its decisions? Let's look at predicted probabilities 

# Save the predicted probabilities of a book being a favourite
predicted_probs = logreg.predict_proba(X_test_scaled)[:, 1]

# Plotting the predicted probabilities
plt.figure(figsize=(10, 6))
scatter = plt.scatter(predicted_probs, y_test, c=y_test, cmap='coolwarm', alpha=0.6)
label_0 = plt.scatter([],[], alpha=0.6, color=scatter.cmap(0), label="Hmm Maybe not") # Dummy scatter plots for legend
label_1 = plt.scatter([],[], alpha=0.6, color=scatter.cmap(255), label="LOVED IT")
plt.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')          # Vertical threshold line
plt.xlabel('Predicted Probability')                                             # Labels & Title
plt.ylabel('True Label + Jitter')
plt.title('Predicted Probabilities vs True Labels')
plt.legend(handles=[label_0, label_1])                                          # Legend
plt.grid(True)
plt.show()

#%%
"""
### IN CLASS ASSIGNMENT
Now it is your turn
    
1. For this in-class assignment please work with the data you prepared during the previous classes. 

2. Choose a binary variable, which will be your outcome/label. If your data does not contain binary variables, create one using a condition.

3. Choose two other variables (or more) which will be your features. Check if the data is balanced, if not, fix this using a method of your choosing. 

4. Split the data into X and y variables, and scale if you think it necessary. 

5. Choose a supervised classifier, I would highly recommend you pick a different one than the ones used here. 

    5a. look at the classifier's documentation and decide which model specifications would be best for you. 
    5b. Fit the model 

6. Calculate the accuracy for the training and test set. 

7. Vary some specification in the model defined above. This can be the way the data is balanced, scaling, train-test split, or model specification etc. 
    and assess how it affects the accuracy of your model. 
    
    7a. Do this a number of times (for the same metric) and think of a way to visualize the findings. Such a figure can be used to substantiate your design choices in the assignment report. 

"""

