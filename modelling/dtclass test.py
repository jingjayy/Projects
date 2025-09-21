########## INSTRUCTIONS ##########
# 1. Only add or modify codes within the blocks enclosed with
#    ########## student's code ##########
# 
# ######################################
######################################


########## student's code ##########
# If you need to import any library, you may do it here
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
######################################
# Task 1
# 1. Update the name and id to your name and student id
def get_student_info():
    ################## student's code ##################
    studentName = "Hong Jing Jay"
    studentId = "22008338"
    ####################################
    return studentName, studentId

def load_file():
    ################## student's code ##################
    # Task 2
    ####################################
    # 1. Load the Iris dataset from sklearn into a pandas DataFrame called 'df'
    # 2. The DataFrame should have the columns:
    #    sepal_length, sepal_width, petal_length, petal_width, target
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    df = pd.read_csv('iris.csv', header=None, names=column_names)
    ####################################
    return df

def train_decision_tree(df):
    ################## student's code ##################
    # Task 3
    ####################################
    # 1. Initialise a DecisionTreeClassifier with maximum depth of 4 using variable name: "dtModel"
    # 2. Train the decision tree model to classify the species (target column)
    #    using the features:
    #       sepal_length, sepal_width, petal_length, petal_width
    dtModel = DecisionTreeClassifier(max_depth=4)
    features_for_dt = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    target_pred = df["target"]
    dtModel.fit(features_for_dt, target_pred)
    ####################################
    return dtModel

def test_decision_tree(df, dtModel):
    ################## student's code ##################
    # Task 4
    ####################################
    # 1. Predict the class for all rows in df using the trained decision tree
    # 2. Add the predicted outcome as a new column of df called "dresult"
    features_for_dt = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    predictions = dtModel.predict(features_for_dt)
    df["dresult"] = predictions
    ###############################################
    return df

def save_to_file(df):
    ################## student's code ##################
    # Task 5
    ####################################
    # 1. Save the dataframe "df" to a csv file with the name of "iris_results.csv"
    df.to_csv("iris_results.csv", index=False)
    ###################################################

if __name__ == "__main__":
    print("Only add or modify codes within the blocks enclosed with")
    print("################## student's code ##################")
    print("")
    print("###################################################")
    print("DO NOT REMOVE OR MODIFY CODES FROM OTHER SECTIONS")
    print("")

    sname, sid = get_student_info()
    print(f"You are {sname} with student ID {sid}")

################## student's code ##################
# you do not need to change the code of this section
# but you may modify the code for debugging purpose
    df = load_file()
    dtModel = train_decision_tree(df)
    df = test_decision_tree(df, dtModel)
    save_to_file(df)

###################################################
    # ---- Added Debug Output ---- (addon)
    print("\nSample of final results:")
    print(df.head(10))
    print("\niris_results.csv has been saved!")
