# DataKind UPR Classifier: "Winterfell"

## Data
In this algorithm we load three data sources: two sets of hand-labeled UPR records (ICAAD team and DataKind team) and one set of external data (collected from the UN SDG topics website, which includes descriptions and briefs of each SDG).  A fourth set of external data, "parternships", was not used for this iteration.  
For a training set, we incorporate all three data sources, while our test set contains only UPR records labeled by ICAAD or DataKind.
For the ICAAD labeled records, we keep the first 4 SDG labels for simplicity.  Datakind labeled records contain at most 3 SDG labels.  
For the external "UN topics" data, the raw text blobs were split on ending punctuation in order to create individual observations or records.  All of the resulting observations are assigned the SDG labels of the original blob.  We only keep observations that have been identified as "low occurrence" SDGs as defined in the code.  This is done to minimize the syntactic disparity between UPR records and external data, while still helping the classifier pick up rare SDGs.  
String processing includes removing whitespace, non-alphanumeric characters, and coercing to lowercase.  

## ML Model
Training data was transformed into a TF-IDF matrix, which incorporates a weighting scheme for occurrences of each word token.  We set the TF-IDF vectorizer function to remove common stop words and create features using 1-gram and 2-grams.  We process the response variable (the SDG labels) using the MultiLabelBinarizer function in order to allow the classifier to generate multiple predictions using a One-vs-rest classification regime.
We tried the following models in this code: support vector machine with linear kernel, logistic regression, random forest, and gradient boosting, with linear SVM performing the best in terms of F1-score, and gradient boosting supplying similar performance but notably taking longer to run.

## Keyword Rules
In addition to modeling, we also incorporate a regex-matching approach utilize keywords that ICAAD has helped provide.  We test two implementations of this method: 1) keyword predictions are generated for instances where the model is not able to predict anything, and 2) keyword predictions are generated for all test observations and consolidated with the model based predictions.  Either way, the final product is a two-step ensemble approach that tries to maximize recall without suffering much in precision.      

## Included Files
* data
	* folder including all labeled training data
* datakind_draft.py
	* rough code
* datakind_final_dataload.py
	* loads and processes all raw training data
* datakind_final_functions.py
	* convenience functions
* datakind_final_predictor.py
	* trains final model on all data, then predicts on new data
* datakind_final_validation.py
	* code for cross-validation
* keyword_dict_v1.py, keyword_dict_v2.py, keyword_dict_v3.py
	* dictionaries for keyword lookups
* README.md
	* this file
* requirements.txt
	* required libraries
* total_upr_unlabeled.csv
	* scraped UPRs from UN site
* upr_unlabeled_final_preds_draft.csv
	* initial set of predictions
* upr_unlabeled_final_preds_final
	* final set of predictions

## Execution
In Terminal: `python2.7 datakind_final_predictor.py total_upr_unlabeled.csv upr_unlabeled_final_preds_final.csv 2`