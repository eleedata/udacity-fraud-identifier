#!/usr/bin/python

# Import libraries
import sys
import pickle
import pprint
from time import time

# Import libraries for data exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Import Udacity tools
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###
### Task 1: Data Exploration
###

# How many data points?
total = len(data_dict) + 0.0
print 'Number of datapoints: '+str(total)

# What features are in the dataset?
print ''
print 'Features inside dataset: '
pprint.pprint(data_dict['METTS MARK'])

# How many features are there?
print ''
print 'Number of features: '+str(len(data_dict['METTS MARK']))

# Create a dataframe with dictionary for easier data processing
df = pd.DataFrame(data_dict)
df = df.T

# Find number of POIs vs. non POIs
print ''
print 'Allocation across classes (POI/non-POI): '
print df['poi'].value_counts()

# Get percentages of NaN for every feature
features = df.columns.values.tolist()
results = {}
for feature in features:
    percentage = df[feature][df[feature] == 'NaN'].count()/total
    results[feature] = percentage
print ''
print 'Percentage of NaN values for each feature: '
pprint.pprint(results)

# Convert dictionary into a Pandas series
s = pd.Series(results)

# Show features with high (i.e. > 50%) NaN values
print ''
print 'Features with over 50% NaN values: '
print s[s > 0.50]

###
### Task 2: Remove outliers
###

# Remove rows where all features were missing values
print ''
print 'Datapoint for LOCKHART EUGENE E consists of all missing values: '
print df.loc['LOCKHART EUGENE E']

# Function that will output rows that are in the top or bottom 10% of values
def show_outliers(df, feature):
    # Get upper and lower limits
    s = df[feature][df[feature] != 'NaN']
    lower_lim = s.quantile(0.1)
    upper_lim = s.quantile(0.9)
    # Return dataframe
    return s[(s < lower_lim) | (s > upper_lim)]

# Show outliers for restricted_stock
print ''
print 'Outliers for restricted_stock (particularly large negative value for BHATNAGAR SANJAY): '
print show_outliers(df, 'restricted_stock')

# Show outliers for deferral_payments
print ''
print 'Outliers for deferral_payments (particularly large negative value for BELFER ROBERT): '
print show_outliers(df, 'deferral_payments')

# Function takes in a dataframe, applies cleaning to the data and returns a new dataframe
def clean_df(df):
    # Remove rows where all features were missing values
    df = df.drop('LOCKHART EUGENE E')
    # Remove "THE TRAVEL AGENCY IN THE PARK" and "TOTAL" because they are not people
    df = df.drop(['THE TRAVEL AGENCY IN THE PARK', 'TOTAL'])
    # Update total_stock_value for BELFER ROBERT
    df['total_stock_value']['BELFER ROBERT'] = 'NaN'
    # Update restricted_stock for BHATNAGAR SANJAY
    df['restricted_stock']['BHATNAGAR SANJAY'] = 2604490
    # Update restricted_stock for BELFER ROBERT
    df['deferral_payments']['BELFER ROBERT'] = 'NaN'
    # Fix values for BELFER ROBERT
    df['deferred_income']['BELFER ROBERT'] = -102500
    df['expenses']['BELFER ROBERT'] = 3285
    df['director_fees']['BELFER ROBERT'] = 102500
    df['total_payments']['BELFER ROBERT'] = 3285
    df['restricted_stock']['BELFER ROBERT'] = 44093
    df['restricted_stock_deferred']['BELFER ROBERT'] = -44093
    df['exercised_stock_options']['BELFER ROBERT'] = 'NaN'
    # Return dataframe
    return df

# Clean dataframe of outliers
print ''
print 'Cleaning function clean_df() applied on dataset. '
df_clean = clean_df(df)

###
### Task 3: Select what features you'll use
###

# Function for testing using StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

def test_stratified_shuffle_split(clf, dataset, feature_list, folds = 1000, scale_features = True):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
 
    # Scale features
    if(scale_features):
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print 'Total predictions: '+str(total_predictions)
        print 'Accuracy: '+str(accuracy)
        print 'Precision: '+str(precision)
        print 'Recall: '+str(recall)
        print 'F1: '+str(f1)
        print 'F2: '+str(f2)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

# Convert dataframe back into a dictionary
df_clean_t = df_clean.T
my_dictionary = df_clean_t.to_dict()

### Store to my_dataset for easy export below.
my_dataset = my_dictionary

# Base feature list
features_list = ['poi',
                 'bonus',
                 'deferral_payments',
                 'deferred_income',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'loan_advances',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'salary',
                 'shared_receipt_with_poi',
                 'to_messages',
                 'total_payments',
                 'total_stock_value']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Use Decision Tree algorithm to determine feature importances
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features, labels)
importances = clf.feature_importances_

print ''
print 'Decision Tree feature importances: '
for i in range(len(importances)):
    print "{}\t{}".format(features_list[i+1], importances[i])

# Compare performance of different feature lists based on feature importances
features_list_test = ['poi']
features_addon = ['exercised_stock_options',
                  'total_payments',
                  'bonus',
                  'long_term_incentive',
                  'shared_receipt_with_poi',
                  'expenses',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'salary',
                  'other',
                  'restricted_stock']

for f in features_addon:
    features_list_test.append(f)
    print ''
    print 'Performance with feature list: '+str(features_list_test)
    clf = DecisionTreeClassifier()
    test_stratified_shuffle_split(clf, my_dataset, features_list_test)

# Convert features and labels into numpy array
features_np = np.asarray(features)
labels_np = np.asarray(labels)

# Select features using SelectKBest 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

fs = SelectKBest(f_classif, k='all')
features_np_new = fs.fit_transform(features_np, labels_np)
fs_scores = fs.scores_

print ''
print 'SelectKBest feature scores: '
for i in range(len(fs_scores)):
    print "{}\t{}".format(features_list[i+1], fs_scores[i])

# New feature list with highest Decision Tree feature importances
features_list = ['poi',
                 'exercised_stock_options',
                 'total_payments',
                 'bonus',
                 'long_term_incentive',
                 'shared_receipt_with_poi',
                 'expenses',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'salary']

###
### Task 4: Create new feature(s)
###

# Replace all string of 'NaN' to be proper NaN value
df_clean = df_clean.replace('NaN', np.nan, regex=True)

# Create percentage_salary feature
df_clean['percentage_salary'] = df_clean['salary']/(df_clean['total_payments'] + df_clean['total_stock_value'])

# Replace all proper NaN values to string 'NaN' because featureFormat() converts 'NaN' to 0.0
df_clean = df_clean.replace(np.nan, 'NaN', regex=True)

# Convert dataframe back into a dictionary
df_clean_t = df_clean.T
my_dictionary = df_clean_t.to_dict()

### Store to my_dataset for easy export below.
my_dataset = my_dictionary

# Compare results of classifier with and without engineered features
print ''
print 'Performance WITHOUT engineered feature: '
clf = DecisionTreeClassifier()
test_stratified_shuffle_split(clf, my_dataset, features_list)

# Include new features into feature list
features_list.append('percentage_salary')
print 'Performance WITH engineered feature: '
test_stratified_shuffle_split(clf, my_dataset, features_list)

### Task 5: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Revert feature list back to highest Decision Tree feature importances
features_list = ['poi',
                 'exercised_stock_options',
                 'total_payments',
                 'bonus',
                 'long_term_incentive',
                 'shared_receipt_with_poi',
                 'expenses',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'salary']

# Second feature list selected with SelectKBest for testing
features_list_2 = ['poi',
                    'exercised_stock_options',
                    'total_stock_value',
                    'bonus', 
                    'salary', 
                    'deferred_income']

# Import libraries for algorithms to test
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Test performance with features_list
clf_dt = DecisionTreeClassifier()
clf_knn = KNeighborsClassifier()

print ''
print 'Decision Tree classifier with Feature List selected by Decision Tree feature importance: '
test_stratified_shuffle_split(clf_dt, my_dataset, features_list)

print ''
print 'KNN classifier with Feature List selected by Decision Tree feature importance (feature scaling applied): '
test_stratified_shuffle_split(clf_knn, my_dataset, features_list)

print ''
print 'KNN classifier with Feature List selected by Decision Tree feature importance (feature scaling NOT applied): '
test_stratified_shuffle_split(clf_knn, my_dataset, features_list, scale_features = False)

# Test performance with features_list_2
clf_dt = DecisionTreeClassifier()
clf_knn = KNeighborsClassifier()

print ''
print 'Decision Tree classifier with Feature List selected with SelectKBest: '
test_stratified_shuffle_split(clf_dt, my_dataset, features_list_2)

print ''
print 'KNN classifier with Feature List selected with SelectKBest (feature scaling applied): '
test_stratified_shuffle_split(clf_knn, my_dataset, features_list_2)

print ''
print 'KNN classifier with Feature List selected with SelectKBest (feature scaling NOT applied): '
test_stratified_shuffle_split(clf_knn, my_dataset, features_list_2, scale_features = False)


### Task 6: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Tune with GridSearchCV
from sklearn.grid_search import GridSearchCV
parameters = {'min_samples_split':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'max_depth':[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters, cv=10)
clf.fit(features, labels)
print ''
print 'Best parameter settings for DecisionTreeClassifier using GridSearchCV: '
print clf.best_params_

# Manually tune min_samples_split
print "DecisionTreeClassifier(min_samples_split=1)"
clf = DecisionTreeClassifier(min_samples_split=1)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=2)"
clf = DecisionTreeClassifier(min_samples_split=2)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=5)"
clf = DecisionTreeClassifier(min_samples_split=5)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=10)"
clf = DecisionTreeClassifier(min_samples_split=10)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=20)"
clf = DecisionTreeClassifier(min_samples_split=20)
test_stratified_shuffle_split(clf, my_dataset, features_list)

# Manually tune max_depth
print "DecisionTreeClassifier(max_depth=1)"
clf = DecisionTreeClassifier(max_depth=1)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(max_depth=2)"
clf = DecisionTreeClassifier(max_depth=2)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(max_depth=5)"
clf = DecisionTreeClassifier(max_depth=5)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(max_depth=10)"
clf = DecisionTreeClassifier(max_depth=10)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(max_depth=15)"
clf = DecisionTreeClassifier(max_depth=15)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(max_depth=25)"
clf = DecisionTreeClassifier(max_depth=25)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(max_depth=35)"
clf = DecisionTreeClassifier(max_depth=35)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(max_depth=50)"
clf = DecisionTreeClassifier(max_depth=50)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(max_depth=100)"
clf = DecisionTreeClassifier(max_depth=100)
test_stratified_shuffle_split(clf, my_dataset, features_list)

# Combine manually tuned parameters
print "DecisionTreeClassifier(min_samples_split=1, max_depth=15)"
clf = DecisionTreeClassifier(min_samples_split=1, max_depth=15)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=2, max_depth=15)"
clf = DecisionTreeClassifier(min_samples_split=2, max_depth=15)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=3, max_depth=15)"
clf = DecisionTreeClassifier(min_samples_split=3, max_depth=15)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=2, max_depth=5)"
clf = DecisionTreeClassifier(min_samples_split=2, max_depth=5)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=2, max_depth=6)"
clf = DecisionTreeClassifier(min_samples_split=2, max_depth=6)
test_stratified_shuffle_split(clf, my_dataset, features_list)

print "DecisionTreeClassifier(min_samples_split=3, max_depth=5)"
clf = DecisionTreeClassifier(min_samples_split=3, max_depth=5)
test_stratified_shuffle_split(clf, my_dataset, features_list)

# Final tuned classifier
clf = DecisionTreeClassifier(min_samples_split=2, max_depth=15)
test_stratified_shuffle_split(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
