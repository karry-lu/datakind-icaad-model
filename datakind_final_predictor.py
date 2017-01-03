##########################################
### DataKind UPR Classifier: "Winterfell"
##########################################


import pandas as pd
import numpy as np
import sys
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
import keyword_dict_v3 as rebecca
import datakind_final_functions as function_library
import datakind_final_dataload as loaded_data

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=np.nan, precision = 3, linewidth = 125)
r_state = 104
keywords = rebecca.load_keyword_dict()
target_names = [str(x) for x in range(1,18)]


def main(args):

    input_file = args[0]
    output_file = args[1]
    keyword_lookup_selection = int(args[2])

    ##### Train best model on full set #####

    full_data = loaded_data.all_labeled_data.append(loaded_data.topics_data_processed_subset)
    x_full_data = [function_library.process_text_string(x) for x in full_data['Recommendation'].tolist()]
    x_full_data_tfidf, tfidf_vectorizer = function_library.feature_engineering(x_full_data, (1,2))
    mlb = MultiLabelBinarizer()
    y_full_data = function_library.process_y_all_labeled_data(full_data)
    y_full_data_tf = mlb.fit_transform(y_full_data)

    clf_linear_svc = LinearSVC(C = 2, loss = "squared_hinge", dual = False, penalty = "l1", random_state = r_state)
    multi_clf = OneVsRestClassifier(clf_linear_svc)
    multi_clf.fit(x_full_data_tfidf, y_full_data_tf)

    ##### Predict on Unlabeled Data #####

    # load and transform unlabeled UPRs
    upr_unlabeled = pd.read_csv(input_file, sep=',')
    upr_unlabeled_list = [function_library.process_text_string(x) for x in upr_unlabeled['Recommendation'].tolist()]
    upr_unlabeled_test_tfidf = tfidf_vectorizer.transform(upr_unlabeled_list)

    # predict on new data with SVM object and keyword rules (v1 takes a few minutes, v2 takes roughly half an hour)
    upr_unlabeled_preds = multi_clf.predict(upr_unlabeled_test_tfidf)
    if keyword_lookup_selection == 1:
        print "Selected keyword lookup version 1"
        upr_unlabeled_preds_with_keywords = function_library.keyword_rules_1(keywords, upr_unlabeled_list, upr_unlabeled_preds)
    else:
        print "Selected keyword lookup version 2"
        upr_unlabeled_preds_with_keywords = function_library.keyword_rules_2(keywords, upr_unlabeled_list, upr_unlabeled_preds)

    # recommend keywords v2 for final output
    upr_unlabeled_final_preds_matrix = function_library.format_predictions(upr_unlabeled, upr_unlabeled_preds_with_keywords)
    upr_unlabeled_final_preds_matrix.to_csv(output_file, sep=",", index = False)


if __name__ == "__main__":
    main(sys.argv[1:])


"""
# quick check of India
answers = pd.read_csv("/Users/karrylu/Documents/DataKind/india_answer_key.csv", sep=',')
answers_matrix = answers.as_matrix()
india = upr_unlabeled_final_preds_matrix[upr_unlabeled_final_preds_matrix['SuR']=="India"]
india_matrix = india[list(india.columns.values)[0:17]].as_matrix()

print(metrics.classification_report(answers_matrix, india_matrix, target_names=target_names))
metrics.hamming_loss(answers_matrix, india_matrix)

hamming loss of 0.024 and precision/recall/f scores of 0.91/0.87/0.89

# quick check of Ghana
answers = pd.read_csv("/Users/karrylu/Documents/DataKind/ghana_answer_key.csv", sep=',')
answers_matrix = answers.as_matrix()
ghana = upr_unlabeled_final_preds_matrix[upr_unlabeled_final_preds_matrix['SuR']=="Ghana"]
ghana_matrix = ghana[list(ghana.columns.values)[0:17]].as_matrix()

print(metrics.classification_report(answers_matrix, ghana_matrix, target_names=target_names))
metrics.hamming_loss(answers_matrix, ghana_matrix)

hamming loss of 0.026 and precision/recall/f scores of 0.94/0.86/0.89

"""
