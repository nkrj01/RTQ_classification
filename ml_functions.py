import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizing import pre_processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


##### Groups different lables together to reduce the number of tags and eliminate tags that are too low in numbers ####
def assign_value(dataframe):
    if dataframe["Count"] < 20:
        return "others"
    else:
        return dataframe["L1"]


def group_others(dataframe):
    if int(dataframe["Count"]) == 0:
        return "others"
    else:
        return dataframe["L1"]


def grouping(df, column):
    df3 = df.groupby(column).count()
    df3.reset_index(inplace=True)
    df3 = df3.rename(columns={'Question Text': 'Count'})

    df3["L1"] = df3.apply(assign_value, axis=1)
    df4 = df3.groupby(column).sum()
    df4.reset_index(inplace=True)
    df4 = df4[[column, "Count"]]

    df_merged = df.merge(df4, how='left')
    df_merged['Count'] = df_merged['Count'].fillna(0)
    df_merged[column] = df_merged.apply(group_others, axis=1)
    return df_merged
    # df_merged = df_merged[["Question Text", "L1", "L2"]]


######################################################################################################################

###### Create single-label, multi-label and multi-task labels. Converting labels from text to integers.
def create_labels(df, label_type='single_label'):
    df = df.replace(np.nan, '', regex=True)
    L1 = df["L1"].tolist()
    L2 = df["L2"].tolist()
    if label_type == "multi_label":
        labels = []
        for i in range(len(L1)):
            if L2[i]:
                labels.append({L1[i], L2[i]})
            else:
                labels.append({L1[i]})
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels).tolist()
        df['label'] = pd.Series(labels)
        df = df[["Question Text", "label"]]
        print(mlb.classes_)

    elif label_type == 'single_label':
        le = preprocessing.LabelEncoder()
        le.fit(L1)
        labels = le.transform(L1)
        print(le.classes_)
        df["label"] = labels
        df = df[["Question Text", "label"]]

    else: # multi_task
        pde1 = df["PDE1"].tolist()
        le = preprocessing.LabelEncoder()
        label1 = le.fit_transform(L1)
        label2 = le.fit_transform(pde1)
        labels = []
        for i in range(len(label2)):
            labels.append([label2[i], label1[i]])
        df["label"] = pd.Series(labels)
        df = df[["Question Text", "label"]]

    return df


#######################################################################################################################


##### Removes a list of class from the dataframe ######################################################################
def remove_class(df, class_to_remove_PDE, class_to_remove_L1, class_type):
    for i in class_to_remove_L1:
        df = df[df["L1"] != i]

    for i in class_to_remove_PDE:
        df = df[df["PDE1"] != i]

    if class_type == "multi_label":
        for i in class_to_remove_L1:
            df = df[df["L2"] != i]

        for i in class_to_remove_PDE:
            df = df[df["PDE2"] != i]

    return df


#######################################################################################################################


###### Train-test-split, word-to-vector for texts, possible preprocess X(stopwrords, lemmetaie etc) ###################
def train_val_data(df):
    X = df["Question Text"]
    # X = pre_processing(X)
    y = df["label"].tolist()
    X_train_text, X_val_text, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_text = np.hstack((np.array(X_train_text), np.array(X_val_text)))
    vec = CountVectorizer(binary=True, max_df=0.5, min_df=0.01)
    # vec = TfidfVectorizer(binary=True, max_df=0.5, min_df=0.01)
    vec.fit(X_train_text)
    X_train = vec.transform(X_train_text).toarray()
    X_val = vec.transform(X_val_text).toarray()
    y_val = np.array(y_val)
    y_train = np.array(y_train)
    return X_text, X_train, X_val, y_train, y_val


######## Calclates multi_task accuracy ################################################################################
def multi_task_accuracy(y_val, y_pred):
    n = len(y_val[:, 0])
    count_PDE = y_val[:, 0] == y_pred[:, 0]
    accuracy_PDE = np.count_nonzero(count_PDE) / n
    count_L1 = y_val[:, 1] == y_pred[:, 1]
    accuracy_L1 = np.count_nonzero(count_L1) / n
    count_total = np.multiply(count_PDE, count_L1)
    accuracy_total = np.count_nonzero(count_total) / n
    return accuracy_PDE, accuracy_L1, accuracy_total


#######################################################################################################################


##### Cleans text sentence. Removes stopwrods, digits, characters, capitilization #####################################
def cleaner(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]+", " ", text)
    tokenizer = RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(text)
    tokens_without_sw = [word for word in text_tokens if word not in stopwords.words()]
    filtered_sentence = " ".join(tokens_without_sw)
    return filtered_sentence


############# cleaner - stopwords #####################################################################################
def remove_characters(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]+", " ", text)
    tokenizer = RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(text)
    filtered_sentence = " ".join(text_tokens)
    return filtered_sentence


#######################################################################################################################


##### Spearate big questions that have many bullet points into individual questiosn ###################################
def qseparator(text):
    if len(text) >400:
        regex = ["~[A-Za-z0-9]+\.", "~[A-Za-z0-9]+\)", "(•)", "\(\d\)"]
        l = []
        for ex in regex:
            l.append(len(re.findall(ex, text)))

        max_value = max(l)
        if max_value >= 2:
            index = l.index(max_value)
            split_list = re.compile(regex[index]).split(text)

            short_split_list = [i for i in split_list if len(i) > 80]
            if len(short_split_list) > 1:
                if short_split_list[0][-1] == ":" or short_split_list[0][-2] == ":" or short_split_list[0].find(
                        "following") != -1:
                    final_list = []
                    for i in range(1, len(short_split_list)):
                        final_list.append(short_split_list[0] + short_split_list[i])
                    return final_list

                else:
                    return short_split_list
            else:
                return short_split_list

        else:
            return [text]

    else:
        return [text]
#######################################################################################################################
