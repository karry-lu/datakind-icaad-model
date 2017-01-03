##########################################
### DataKind UPR Classifier: "Winterfell"
##########################################


import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_fscore_support
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


##### Cross Validation #####


precision_array = []
recall_array = []
f_score_array = []
kf = KFold(len(loaded_data.all_labeled_data), n_folds=4, shuffle = True, random_state = r_state)
for train_index, test_index in kf:
    train = loaded_data.all_labeled_data.iloc[train_index]
    test = loaded_data.all_labeled_data.iloc[test_index]
    train = train.append(loaded_data.topics_data_processed_subset)

    ##### Feature Engineering #####

    x_train, x_test, x_train_tfidf, x_test_tfidf, tfidf_vectorizer = function_library.process_X_matrix(train, test)
    y_train_tf, y_test_tf = function_library.process_Y_array(train, test)

    ##### Train SVM #####

    clf_linear_svc = LinearSVC(C = 2, loss = "squared_hinge", dual = False, penalty = "l1", random_state = r_state)
    multi_clf = OneVsRestClassifier(clf_linear_svc)
    multi_clf.fit(x_train_tfidf, y_train_tf)
    preds = multi_clf.predict(x_test_tfidf)
    preds_scores = multi_clf.decision_function(x_test_tfidf)

    target_names = [str(x) for x in range(1,18)]
    hamming_loss(y_test_tf, preds)
    print(metrics.classification_report(y_test_tf, preds, target_names=target_names))

    ##### Keyword Rules  #####

    # # keyword rules #1
    # preds_with_keywords_1 = function_library.keyword_rules_1(keywords, x_test, preds)
    # print(metrics.classification_report(y_test_tf, preds_with_keywords_1, target_names=target_names))
    # hamming_loss(y_test_tf, preds_with_keywords_1)

    # # keyword rules #2
    # preds_with_keywords_2 = function_library.keyword_rules_2(keywords, x_test, preds)
    # print(metrics.classification_report(y_test_tf, preds_with_keywords_2, target_names=target_names))
    # hamming_loss(y_test_tf, preds_with_keywords_2)

    ##### Evaluation Metrics #####

    p, r, f, s = precision_recall_fscore_support(y_test_tf, preds, average = "weighted")
    precision_array.append(p)
    recall_array.append(r)
    f_score_array.append(f)

eval_metrics = pd.DataFrame(
    data = {'k': range(1,5), 'precision': precision_array, 'recall': recall_array, 'f-score': f_score_array},
    columns = ['k', 'precision', 'recall', 'f-score'])
print ("Cross-validated Precision:", round(np.mean(eval_metrics['precision']),3))
print ("Cross-validated Recall:", round(np.mean(eval_metrics['recall']),3))
print ("Cross-validated F-score:", round(np.mean(eval_metrics['f-score']),3))


