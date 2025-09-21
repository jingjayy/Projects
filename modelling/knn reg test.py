########## INSTRUCTIONS ##########
# 1. Only add or modify codes within the blocks enclosed with
#    ########## student's code ##########
# 
# ######################################
######################################


########## student's code ##########
# If you need to import any library, you may do it here
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
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
    ####################################
    return df

def train_knn_regression_model(df):
    ################## student's code ##################
    # Task 3
    ####################################
    # 1. Initialise a KNeighborsRegressor with k=5 using variable name knnModel
    # 2. Train the model to predict median_house_value
    #    using features: longitude, latitude, median_income
    features_for_knn = df[['longitude', 'latitude', 'median_income']]
    knnModel = KNeighborsRegressor(n_neighbors=5)
    knnModel.fit(features_for_knn, df['median_house_value'])
    ####################################
    return knnModel

def test_knn_regression_model(df, knnModel):
    ################## student's code ##################
    # Task 4
    ####################################
    # 1. Predict the target for any 10 rows in df
    # 2. Save the result in a variable named outcome
    test_data = df[['longitude', 'latitude', 'median_income']].head(10)
    outcome = knnModel.predict(test_data)
    ####################################
    return outcome

def add_prediction_result_to_data(df, knnModel):
    ################## student's code ##################
    # Task 5
    ####################################
    # 1. Predict the target for all rows in df
    # 2. Add the result as a new column 'dresult'
    features_for_knn = df[['longitude', 'latitude', 'median_income']]
    df['dresult'] = knnModel.predict(features_for_knn)
    ####################################
    return df

def save_to_file(df):
    ################## student's code ##################
    # Task 6
    ####################################
    # 1. Save df to 'california_knn_results.csv'
    df.to_csv("california_knn_results.csv", index=False)
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
    knnModel = train_knn_regression_model(df)
    results = test_knn_regression_model(df, knnModel)
    df = add_prediction_result_to_data(df, knnModel)
    save_to_file(df)

###################################################
    # ---- Added Debug Output ---- (addon)
    print("\nFirst 10 KNN regression predictions:", results)
    print("\nSample of final results:")
    print(df.head(10))
    print("\ncalifornia_knn_results.csv has been saved!")
