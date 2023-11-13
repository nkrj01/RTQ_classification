import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
import statistics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, r'C:\Users\nraj01\PycharmProjects\pythonProject1\RTQ')
import ml_functions as fc # import helper functions class


if __name__ == "__main__":
    class_type = "multi_label"
    df = pd.read_excel("RTQs analysis_raw.xlsx") # read the data
    df = df[["Question Text", "PDE1", "PDE2", "L1", "L2"]]
    df = df.loc[(df['PDE1'] == 'Filling') | (df['PDE1'] == 'Sterile Filtration')]
    df = df.reset_index(drop=True)
    class_to_remove_L1 = ["Others", "Justify", "Correction", "Data"]
    class_to_remove_PDE = ["Sterile Filtration"]
    df = fc.remove_class(df, class_to_remove_PDE, class_to_remove_L1, class_type)
    df = df.reset_index(drop=True)
    df_labeled = fc.create_labels(df, class_type)

    X_train, X_val, y_train, y_val = fc.train_val_data(df_labeled)

    # Models
    clf = RandomForestClassifier(random_state=0)
    multi_target_forest = MultiOutputClassifier(clf, n_jobs=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    if class_type == "multi_task":
        print("Accuracy: ", fc.multi_task_accuracy(y_val, y_pred))
    else:
        print("f1 score: ", f1_score(y_val, y_pred, average='macro'))
        print("Accuracy: ", clf.score(X_val, y_val))
