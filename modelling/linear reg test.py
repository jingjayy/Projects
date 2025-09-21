########## INSTRUCTIONS ##########
# 1. Only add or modify codes within the blocks enclosed with
#    ########## student's code ##########
#
# ######################################
######################################


########## student's code ##########
# If you need to import any library, you may do it here
import pandas as pd
from sklearn.linear_model import LinearRegression
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
    # 1. Load the California Housing dataset CSV into a pandas DataFrame called df
    #    (CSV has no header)
    # 2. Columns (from left to right) are:
    #    longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
    #    population, households, median_income, median_house_value, ocean_proximity
    # 3. Drop the "ocean_proximity" column because it is categorical
    column_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                    'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity']
    df = pd.read_csv('housing.csv', header=None, names=column_names)
    df = df.drop(columns=['ocean_proximity'])
    ####################################
    return df

def train_linear_regression_model(df):
    ################## student's code ##################
    # Task 3
    ####################################
    # 1. Initialise a LinearRegression model using variable name linModel
    # 2. Train the model to predict median_house_value
    #    using features: longitude, latitude, median_income
    features_for_lin = df[['longitude', 'latitude', 'median_income']]
    linModel = LinearRegression()
    linModel.fit(features_for_lin, df['median_house_value'])
    ####################################
    return linModel

def test_linear_regression_model(df, linModel):
    ################## student's code ##################
    # Task 4
    ####################################
    # 1. Predict the target for any 10 rows in df
    # 2. Save the result in a variable named outcome
    test_data = df[['longitude', 'latitude', 'median_income']].head(10)
    outcome = linModel.predict(test_data)
    ####################################
    return outcome

def add_prediction_result_to_data(df, linModel):
    ################## student's code ##################
    # Task 5
    ####################################
    # 1. Predict the target for all rows in df
    # 2. Add the result as a new column 'dresult'
    features_for_lin = df[['longitude', 'latitude', 'median_income']]
    df['dresult'] = linModel.predict(features_for_lin)
    ####################################
    return df

def save_to_file(df):
    ################## student's code ##################
    # Task 6
    ####################################
    # 1. Save df to 'california_linreg_results.csv'
    df.to_csv("california_linreg_results.csv", index=False)
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
    linModel = train_linear_regression_model(df)
    results = test_linear_regression_model(df, linModel)
    df = add_prediction_result_to_data(df, linModel)
    save_to_file(df)

###################################################
    # ---- Added Debug Output ---- (addon)
    print("\nFirst 10 Linear Regression predictions:", results)
    print("\nSample of final results:")
    print(df.head(10))
    print("\ncalifornia_linreg_results.csv has been saved!")
