##########################################
### DataKind UPR Classifier: "Winterfell"
##########################################


import pandas as pd
import numpy as np
import datakind_final_functions as function_library

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=np.nan, precision = 3, linewidth = 125)
r_state = 104


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
    # print entry
    # print len(tab2)
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
topics_data_processed = function_library.process_text_and_split_sentences_for_external(topics_data)
topics_data_processed['SDGs'] = topics_data_processed['SDGs'].astype(str)
topics_data_processed['Issue'] = "no issue"
topics_data_processed = topics_data_processed[topics_data_processed['Recommendation'].str.len() > 3]
# keep observations tagged with the following low-occurrence SDGs
low_sdgs = [str(x) for x in [7, 11, 12, 13, 14, 15, 17]]
topics_data_processed_subset = topics_data_processed[topics_data_processed['SDGs'].isin(low_sdgs)]


# combine UPR data sources
all_labeled_data = our_labeled_data.append(icaad_labeled_data)
