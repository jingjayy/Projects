########## INSTRUCTIONS ##########
# 1. Only add or modify codes within the blocks enclosed with
#    ########## student's code ##########
# 
# ######################################
######################################


########## student's code ##########
# If you need to import any library, you may do it here
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
    # 1. Load the Wine Classification dataset CSV into a pandas DataFrame called df
    #    (CSV has no header)
    # 2. Columns (from left to right) are:
    #    class, alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
    #    total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
    #    color_intensity, hue, od280_od315, proline
    column_names = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                    'color_intensity', 'hue', 'od280_od315', 'proline']
    df = pd.read_csv('wine.csv', header=None, names=column_names)
    ####################################
    return df

def train_knn_model(df):
    ################## student's code ##################
    # Task 3
    ####################################
    # 1. Initialise a KNeighborsClassifier with k=5 using variable name knnModel
    # 2. Train using all feature columns except 'class' as input
    features_for_knn = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                    'color_intensity', 'hue', 'od280_od315', 'proline']]
    knnModel = KNeighborsClassifier(n_neighbors=5)
    knnModel.fit(features_for_knn, df['class'])
    ####################################
    return knnModel

def test_knn_model(df, knnModel):
    ################## student's code ##################
    # Task 4
    ####################################
    # 1. Predict the class for any 10 rows from df
    # 2. Save the result in a variable named outcome
    test_data = df.drop(columns=['class']).head(10)
    outcome = knnModel.predict(test_data)
    ####################################
    return outcome

def add_classification_result_to_data(df, knnModel):
    ################## student's code ##################
    # Task 5
    ####################################
    # 1. Predict the class for all rows in df
    # 2. Add a new column 'dresult' with the predicted class
    features_for_knn = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                    'color_intensity', 'hue', 'od280_od315', 'proline']]
    df['dresult'] = knnModel.predict(features_for_knn)
    ####################################
    return df

def save_to_file(df):
    ################## student's code ##################
    # Task 6
    ####################################
    # 1. Save df to 'wine_knn_results.csv'
    df.to_csv("wine_knn_results.csv", index=False)
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
    knnModel = train_knn_model(df)
    results = test_knn_model(df, knnModel)
    df = add_classification_result_to_data(df, knnModel)
    save_to_file(df)

###################################################
    # ---- Added Debug Output ---- (addon)
    print("\nFirst 10 KNN predictions:", results)
    print("\nSample of final results:")
    print(df.head(10))
    print("\nwine_knn_results.csv has been saved!")
