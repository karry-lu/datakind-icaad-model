##########################################
### DataKind UPR Classifier: "Winterfell"
##########################################


import pandas as pd
import numpy as np
import re
import copy
import os
import string
import xgboost as xgb
import keyword_dict_v2 as rebecca

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=np.nan, precision = 3, linewidth = 125)
r_state = 104
keywords = rebecca.load_keyword_dict()


def feature_engineering(train_text, ngrams):
    # remove stop words first
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = ngrams)
    train_tfidf = tfidf_vectorizer.fit_transform(train_text)
    return train_tfidf, tfidf_vectorizer

def process_text_string(input_string):
    input_string = re.sub(r'[^a-zA-Z0-9 ]', '', input_string)
    input_string = input_string.strip()
    input_string = input_string.lower()
    return input_string

# not used
def split_into_test_train(labeled_df, test_fraction, r_state):
    train, test = train_test_split(labeled_df, test_size = test_fraction, random_state = r_state)
    train_text = train['text'].tolist()
    train_target = train['label'].tolist()
    test_text = test['text'].tolist()
    test_target = test['label'].tolist()
    # returns X and Y for both train and test sets
    return train_text, train_target, test_text, test_target

# not used
def process_y_datakind_labeled_data(data):
    y_train = []
    for index, row in data.iterrows():
        r = row.tolist()[2:][0]
        r2 = [int(x) for x in r.split(',')]
        y_train.append(r2)
    return y_train

# not used
def process_y_icaad_labeled_data(data):
    y_train = []
    for index, row in data.iterrows():
        r = row.tolist()[2:]
        r2 = [int(x) for x in r if x != 0]
        y_train.append(r2)
    return y_train

def process_y_all_labeled_data(data):
    y_train = []
    for index, row in data.iterrows():
        print index
        r = row.tolist()[2:][0]
        r2 = [int(x) for x in r.split(',') if int(x) != 0]
        y_train.append(r2)
    return y_train

def process_text_and_split_sentences_for_external(data):
    # accepts as input a 2 column df: text, label
    data = data.dropna(how='any')
    list_of_sentences = []
    list_of_labels = []
    for i in range(len(data)):
        # split text blob on periods
        split_sentences = data['text'].tolist()[i].split(".")
        # process text
        split_sentences = [process_text_string(x) for x in split_sentences]
        split_sentences = filter(None, split_sentences)
        sdg = data['label'].tolist()[i]
        array_sdg_label = [sdg]*len(split_sentences)
        list_of_sentences.extend(split_sentences)
        list_of_labels.extend(array_sdg_label)
    processed_df = pd.DataFrame(data = {'Recommendation' : list_of_sentences, 'SDGs': list_of_labels}, columns = ['Recommendation', 'SDGs'])
    return processed_df

# not used
def examine_prediction_matrix(preds, x_test):
    np.savetxt("pred_logit_matrix.csv", preds, delimiter=",")
    test_df = pd.DataFrame(data = {'text': x_test}, columns = ['text'])
    test_df.to_csv("pred_text.csv", index=False)

# not used
def parse_predictions(original_df, preds_matrix):
    SDG_list = []
    for row_index in range(0,len(preds_matrix)):
        row = preds_matrix[row_index,]
        index = np.where(row==1)[0]+1
        index_str = np.array_str(index)
        SDG_list.append(index_str)
    new_df = original_df.copy()
    new_df['predicted_SDGs'] = SDG_list
    return new_df

def format_predictions(original_df, preds_matrix):
    df = pd.DataFrame(preds_matrix)
    df.columns = ["SDG_" + str(i) for i in range(1,18)]
    df['Recommendation'] = original_df['Recommendation']
    df['SuR'] = original_df['SuR']
    df['RS'] = original_df['RS']
    df['Response'] = original_df['Response']
    df['Action'] = original_df['Action']
    df['Issue'] = original_df['Issue']
    df['Session'] = original_df['Session']
    return df

def keyword_rules_1(keywords, x_test, preds):
    row_sums = np.apply_along_axis(lambda x: sum(x), axis=1, arr= preds).tolist()
    indices = [i for i, x in enumerate(row_sums) if x == 0]
    print len(indices)
    # copy over prediction matrix
    preds_copy = copy.copy(preds)
    for row_index in indices:
        inserted_row = np.repeat(0, 17)
        for SDG in keywords.keys():
            all_results = []
            for item in keywords[SDG]:
                regex_pattern = re.compile(item)
                record = x_test[row_index]
                regex_result = re.findall(regex_pattern, record)
                all_results += regex_result
            if len(all_results) > 0:
                SDG_number = re.findall(r'\d+', SDG)[0]
                SDG_number = int(SDG_number)
                inserted_row[(SDG_number-1)] = 1
        preds_copy[row_index,] = inserted_row

    row_sums = np.apply_along_axis(lambda x: sum(x), axis=1, arr= preds_copy).tolist()
    indices_recheck = [i for i, x in enumerate(row_sums) if x == 0]
    print len(indices_recheck)
    return preds_copy

def keyword_rules_2(keywords, x_test, preds):
    # initialize empty matrix
    preds_rules = np.zeros(shape=preds.shape)
    for row_index in range(0,len(preds_rules)):
        print row_index
        inserted_row = np.repeat(0, 17)
        for SDG in keywords.keys():
            all_results = []
            for item in keywords[SDG]:
                regex_pattern = re.compile(item)
                record = x_test[row_index]
                regex_result = re.findall(regex_pattern, record)
                all_results += regex_result
            if len(all_results) > 0:
                SDG_number = re.findall(r'\d+', SDG)[0]
                SDG_number = int(SDG_number)
                inserted_row[(SDG_number-1)] = 1
        preds_rules[row_index,] = inserted_row
    preds_with_keywords_2 = np.maximum(preds, preds_rules)
    return preds_with_keywords_2


##### Import all labeled data #####


### load ICAAD labeled data
f = "data/manual_ICAAD_Labeling_latest.xlsx"
icaad_spreadsheet = pd.read_excel(f, sheetname = None)

# process each sheet
temp_data_hold = pd.DataFrame()
for entry in icaad_spreadsheet.keys():
    tab = icaad_spreadsheet[entry]
    # select columns to keep.  keep only text, issues and 4 SDG labels
    cols = list(tab.columns.values)
    # extract SDG number
    tab['SDG_1'] = tab['SDG Goal I'].str.split('-').str.get(0).str.strip()
    tab['SDG_2'] = tab['SDG Goal II'].str.split('-').str.get(0).str.strip()
    tab['SDG_3'] = tab['SDG Goal III'].str.split('-').str.get(0).str.strip()
    tab['SDG_4'] = tab['SDG Goal IV'].str.split('-').str.get(0).str.strip()
    keep = ['Recommendation', 'Issue', 'SDG_1', 'SDG_2', 'SDG_3', 'SDG_4']
    tab2 = tab[keep]
    print entry
    print len(tab2)
    temp_data_hold = temp_data_hold.append(tab2)

# drop all still-unlabeled rows
icaad_labeled_data = temp_data_hold[~temp_data_hold['SDG_1'].isnull()]
icaad_labeled_data = icaad_labeled_data.fillna(0)
# drop SDG labels that are just long ass text strings
sdg_list = [str(x) for x in range(0,18)]
icaad_labeled_data = icaad_labeled_data[icaad_labeled_data['SDG_1'].isin(sdg_list)]
icaad_labeled_data = icaad_labeled_data[~(icaad_labeled_data['SDG_1'].str.len() > 3)]
icaad_labeled_data = icaad_labeled_data[~(icaad_labeled_data['SDG_2'].str.len() > 3)]
icaad_labeled_data = icaad_labeled_data[~(icaad_labeled_data['SDG_3'].str.len() > 3)]
icaad_labeled_data = icaad_labeled_data[~(icaad_labeled_data['SDG_4'].str.len() > 3)]
# combine into one column
icaad_labeled_data['SDGs'] = icaad_labeled_data['SDG_1'].astype(str) + ", "  + icaad_labeled_data['SDG_2'].astype(str) + ", " +  icaad_labeled_data['SDG_3'].astype(str) + ", " + icaad_labeled_data['SDG_4'].astype(str)
icaad_labeled_data = icaad_labeled_data.drop(['SDG_1', 'SDG_2','SDG_3', 'SDG_4'], 1)


### load DataKind labeled data
manual_datakind_data = pd.read_csv("data/manual_datakind_labels.csv", sep=',')
keep = ['text', 'Issue', 'SDGs']
manual_datakind_data = manual_datakind_data[keep]
manual_datakind_data.rename(columns={'text':'Recommendation'}, inplace=True)
# drop all still unlabeled rows
our_labeled_data = manual_datakind_data[~manual_datakind_data['SDGs'].isnull()]


### load external data: UN topics
topics_data = pd.read_csv("data/external_un_topics.csv", sep=',', header=0, names=['text', 'label'])
topics_data_processed = process_text_and_split_sentences_for_external(topics_data)
topics_data_processed['SDGs'] = topics_data_processed['SDGs'].astype(str)
topics_data_processed['Issue'] = "no issue"
topics_data_processed = topics_data_processed[topics_data_processed['Recommendation'].str.len() > 3]
# keep observations tagged with the following low-occurrence SDGs
low_sdgs = [str(x) for x in [7, 11, 12, 13, 14, 15, 17]]
topics_data_processed_subset = topics_data_processed[topics_data_processed['SDGs'].isin(low_sdgs)]


"""
### load external data: partnership
partner = pd.read_csv("data/external_partnerships.csv", sep=",")
partner['SDGs'] = partner['SDGs'].str.replace(r'[\(\)\',]', '')
partner['SDGs'] = partner['SDGs'].str.strip()
partner_desc_text = partner[['SDGs', 'Description']]
partner_desc_text.rename(columns={'Description': 'text', 'SDGs': 'label'}, inplace=True)
# keep only short text
partner_desc_text_short = partner_desc_text[partner_desc_text['text'].str.len() < 500]
# split into single sentences, process each sentence
partner_processed = process_text_and_split_sentences_for_external(partner_desc_text)
# create clean DF
partner_processed_clean = partner_processed[partner_processed['Recommendation'].str.len() > 3]
"""

# combine UPR data sources
all_labeled_data = our_labeled_data.append(icaad_labeled_data)
# all_labeled_data = all_labeled_data.append(topics_data_processed_subset)


# split into train and test
train, test = train_test_split(all_labeled_data, test_size = 0.25, random_state = r_state)
# augment training set with UN topics data; keep test set solely UPR data
train = train.append(topics_data_processed_subset)


##### Feature Engineering #####


# process X
x_train = [process_text_string(x) for x in train['Recommendation'].tolist()]
x_test = [process_text_string(x) for x in test['Recommendation'].tolist()]
x_train_tfidf, tfidf_vectorizer = feature_engineering(x_train, (1,2))
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# process Y
mlb = MultiLabelBinarizer()
y_train = process_y_all_labeled_data(train)
y_test = process_y_all_labeled_data(test)
y_train_tf = mlb.fit_transform(y_train)
y_test_tf = mlb.fit_transform(y_test)
# get freq count on the labels in training and test set
print np.sum(y_train_tf, axis = 0)
print np.sum(y_test_tf, axis = 0)


##### Train Models #####


# linear SVM
clf_linear_svc = LinearSVC(C = 1, loss = "squared_hinge", dual = True, penalty = "l2", random_state = r_state)
multi_clf = OneVsRestClassifier(clf_linear_svc)
multi_clf.fit(x_train_tfidf, y_train_tf)
preds_svm = multi_clf.predict(x_test_tfidf)
preds_scores_svm = multi_clf.decision_function(x_test_tfidf)


# evaluation metrics
preds = copy.copy(preds_svm)
preds_probs = copy.copy(preds_scores_svm)
delta = np.absolute(preds - y_test_tf)
print(np.sum(delta, axis = 0)/float(len(delta)))
target_names = [str(x) for x in range(1,18)]
hamming_loss(y_test_tf, preds)
print(metrics.classification_report(y_test_tf, preds, target_names=target_names))



"""

# XGBoost
clf_xgb = xgb.XGBClassifier(max_depth = 3, n_estimators = 200, learning_rate = 0.1, silent = False, seed = r_state, objective = "binary:logistic")
multi_clf = OneVsRestClassifier(clf_xgb)
multi_clf.fit(x_train_tfidf, y_train_tf)
preds_xgb = multi_clf.predict(x_test_tfidf)
hamming_loss(y_test_tf, preds_xgb)

preds_probs_xgb = multi_clf.predict_proba(x_test_tfidf)

# logistic regression
clf_logit = LogisticRegression(C = 10, penalty = 'l2', class_weight = None, random_state = r_state, solver='liblinear')
multi_clf = OneVsRestClassifier(clf_logit)
multi_clf.fit(x_train_tfidf, y_train_tf)
preds_logit = multi_clf.predict(x_test_tfidf)
hamming_loss(y_test_tf, preds_logit)

preds_probs_logit = multi_clf.predict_proba(x_test_tfidf)


# random forest
clf_random_forest = RandomForestClassifier(n_estimators=500, random_state=r_state)
multi_clf = OneVsRestClassifier(clf_random_forest)
multi_clf.fit(x_train_tfidf, y_train_tf)
preds_rf = multi_clf.predict(x_test_tfidf)
hamming_loss(y_test_tf, preds_rf)

preds_probs_rf = multi_clf.predict_proba(x_test_tfidf)


"""



# Observations:
# not surprisingly, f1 score and performance for SDGs that have no samples is terrible
# classifiers don't predict any labels for some observations (b/c predicted prob is too low)
# svm performs the best with total f1 score of ~0.76
# partnership data doesn't seem that helpful


##### Keyword Rules  #####

"""
keyword rules 1:

for each record that has no prediction (defined in the list "indices"):
    for each SDG in the keyword_dict:
        for each regex pattern:
            find all instances of the regex in the record
        sum all the instances together
        if len(instances) > 0:
            then assign SDG to this record
            RETURN: a np array with dimensions 1x17, with 1 for each assigned SDG and 0 otherwise
            INSERT: this new row into the original "preds" matrix

keyword rules 2:

for each record in the test set, generate SDG labels based on keyword regex matching.
then overlap keyword prediction matrix with model prediction matrix.
"""


# keyword rules #1

preds_with_keywords_1 = keyword_rules_1(keywords, x_test, preds)
print(metrics.classification_report(y_test_tf, preds_with_keywords_1, target_names=target_names))
hamming_loss(y_test_tf, preds_with_keywords_1)

# keyword rules #2

preds_with_keywords_2 = keyword_rules_2(keywords, x_test, preds)
print(metrics.classification_report(y_test_tf, preds_with_keywords_2, target_names=target_names))
hamming_loss(y_test_tf, preds_with_keywords_2)


##### Predict on Unlabeled Data #####


# load and transform unlabeled UPRs
upr_unlabeled = pd.read_csv("data/total_upr_unlabeled.csv", sep=',')
upr_unlabeled_list = [process_text_string(x) for x in upr_unlabeled['Recommendation'].tolist()]
upr_unlabeled_test_tfidf = tfidf_vectorizer.transform(upr_unlabeled_list)

# predict on new data with SVM object and keyword rules
upr_unlabeled_preds = multi_clf.predict(upr_unlabeled_test_tfidf)
upr_unlabeled_preds_with_keywords_v1 = keyword_rules_1(keywords, upr_unlabeled_list, upr_unlabeled_preds)
upr_unlabeled_preds_with_keywords_v2 = keyword_rules_2(keywords, upr_unlabeled_list, upr_unlabeled_preds)

# use keywords v2 for final output
upr_unlabeled_final_preds_matrix = format_predictions(upr_unlabeled, upr_unlabeled_preds_with_keywords_v2)
upr_unlabeled_final_preds_matrix.to_csv("upr_unlabeled_final_preds_draft.csv", sep=",", index = False)


"""
# quick check of India
answers = pd.read_csv("/Users/karrylu/Documents/DataKind/india_answer_key.csv", sep=',')
answers_matrix = answers.as_matrix()
india = upr_unlabeled_final_preds_matrix[upr_unlabeled_final_preds_matrix['SuR']=="India"]
india_matrix = india[list(india.columns.values)[0:17]].as_matrix()

print(metrics.classification_report(answers_matrix, india_matrix, target_names=target_names))
india.to_csv("upr_unlabeled_final_preds_india.csv", sep=",", index = False)
hamming_loss(answers_matrix, india_matrix)

hamming loss of 0.027 and precision/recall/f scores of 0.92/0.82/0.86

# quick check of Ghana
answers = pd.read_csv("/Users/karrylu/Documents/DataKind/ghana_answer_key.csv", sep=',')
answers_matrix = answers.as_matrix()
ghana = upr_unlabeled_final_preds_matrix[upr_unlabeled_final_preds_matrix['SuR']=="Ghana"]
ghana_matrix = ghana[list(ghana.columns.values)[0:17]].as_matrix()

print(metrics.classification_report(answers_matrix, ghana_matrix, target_names=target_names))
hamming_loss(answers_matrix, ghana_matrix)

hamming loss of 0.029 and precision/recall/f scores of 0.96/0.79/0.86

"""

