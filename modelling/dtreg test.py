########## INSTRUCTIONS ##########
# 1. Only add or modify codes within the blocks enclosed with
#    ########## student's code ##########
# 
# ######################################
######################################


########## student's code ##########
# If you need to import any library, you may do it here
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
######################################
# Task 1
# 1. Update the name and id to your name and student id
def get_student_info():
    ################## student's code ##################
    studentName = 'Hong Jing Jay'
    studentId = '22008338'
    ####################################
    return studentName, studentId

def load_file():
    ################## student's code ##################
    # Task 2
    ####################################
    # 1. Load the California Housing dataset (CSV file) into a pandas DataFrame called 'df'
    #    (note that the csv file has no header)
    # 2. Add/set the headers of the columns (from left to right) to be:
    #    longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity']
    column_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 
                    'households', 'median_income', 'median_house_value', 'ocean_proximity']
    df = pd.read_csv("housing.csv", header=None, names = column_names)
    ####################################
    return df

def train_decision_tree(df):
    ################## student's code ##################
    # Task 3
    ####################################
    # 1. Initialise a DecisionTreeRegressor with maximum depth of 5 using variable name: "dtModel"
    # 2. Train the decision tree model to predict MedHouseVal
    #    using the features: median_income, housing_median_age, total_rooms
    dtModel = DecisionTreeRegressor(max_depth=5)
    features_for_dt = df[["median_income", "housing_median_age", "total_rooms"]]
    target_pred = df["median_house_value"]
    dtModel.fit(features_for_dt, target_pred)
    ####################################
    return dtModel

def test_decision_tree(df, dtModel):
    ################## student's code ##################
    # Task 4
    ####################################
    # 1. Predict the target for all rows in df using the trained decision tree
    # 2. Add the predicted outcome as a new column of df called "dresult"
    features_for_dt = df[["median_income", "housing_median_age", "total_rooms"]]
    predictions = dtModel.predict(features_for_dt)
    df['dresult'] = predictions
    ###############################################
    return df

def save_to_file(df):
    ################## student's code ##################
    # Task 5
    ####################################
    # 1. Save the dataframe "df" to a csv file with the name of "california_results.csv"
    df.to_csv("california_results.csv", index=False)
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
    print("\ncalifornia_results.csv has been saved!")
