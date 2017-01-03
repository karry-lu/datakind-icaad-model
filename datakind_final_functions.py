##########################################
### DataKind UPR Classifier: "Winterfell"
##########################################


import pandas as pd
import numpy as np
import re
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=np.nan, precision = 3, linewidth = 125)
r_state = 104


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

def process_y_all_labeled_data(data):
    y_train = []
    for index, row in data.iterrows():
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

def process_X_matrix(train, test):
    x_train = [process_text_string(x) for x in train['Recommendation'].tolist()]
    x_test = [process_text_string(x) for x in test['Recommendation'].tolist()]
    x_train_tfidf, tfidf_vectorizer = feature_engineering(x_train, (1,2))
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    return x_train, x_test, x_train_tfidf, x_test_tfidf, tfidf_vectorizer

def process_Y_array(train, test):
    mlb = MultiLabelBinarizer()
    y_train = process_y_all_labeled_data(train)
    y_test = process_y_all_labeled_data(test)
    y_train_tf = mlb.fit_transform(y_train)
    y_test_tf = mlb.fit_transform(y_test)
    return y_train_tf, y_test_tf

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