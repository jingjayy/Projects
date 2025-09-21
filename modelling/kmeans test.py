########## INSTRUCTIONS ##########
# 1. Only add or modify codes within the blocks enclosed with
#    ########## student's code ##########
# 
# ######################################
######################################


########## student's code ##########
# If you need to import any library, you may do it here
import pandas as pd
from sklearn.cluster import KMeans
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
    # 1. Load the Wine dataset (CSV file) into a pandas DataFrame called 'df'
    #    (note that the CSV file has no header)
    # 2. Add/set the headers of the columns (from left to right) to be:
    #    Alcohol, Malic_Acid, Ash, Alcalinity_of_Ash, Magnesium, Total_Phenols, 
    #    Flavanoids, Nonflavanoid_Phenols, Proanthocyanins, Color_Intensity, 
    #    Hue, OD280_OD315, Proline
    #    (13 numeric columns in total)
    column_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                    'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity',
                    'Hue', 'OD280_OD315', 'Proline']
    df = pd.read_csv('wine.csv', header=None, names=column_names)
    ####################################
    return df

def train_clustering_model(df):
    ################## student's code ##################
    # Task 3
    ####################################
    # 1. Initialise a KMeans model with 3 clusters using variable name: "kmModel"
    # 2. Train the KMeans model using the first four columns only:
    #       Alcohol, Malic_Acid, Ash, Alcalinity_of_Ash
    features_for_clustering = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash']]
    kmModel = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmModel.fit(features_for_clustering)
    ####################################
    return kmModel

def test_clustering_model(df, kmModel):
    ################## student's code ##################
    # Task 4
    ####################################
    # 1. Use any 10 rows from df to identify/predict their clusters
    # 2. Save the identified cluster indices with variable name: "outcome"
    test_data = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash']].head(10)
    outcome = kmModel.predict(test_data)
    ####################################
    return outcome

def add_clustering_result_to_data(df, kmModel):
    ################## student's code ##################
    # Task 5
    ####################################
    # 1. Predict the clusters of every row in df
    # 2. Convert cluster numbers (0,1,2) to alphabets (a,b,c)
    # 3. Add the cluster outcome as a new column called "cresult"
    features_for_clustering = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash']]
    cluster_predictions = kmModel.predict(features_for_clustering)

    cluster_map = {0:'a', 1:'b', 2:'c'}
    df['cresult'] = pd.Series(cluster_predictions).map(cluster_map)
    ####################################
    return df


def save_to_file(df):
    ################## student's code ##################
    # Task 6
    ####################################
    # 1. Save the dataframe "df" to a csv file with the name of "wine_cluster_results.csv"
    df.to_csv("wine_cluster_results.csv", index=False)
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
    kmModel = train_clustering_model(df)
    results = test_clustering_model(df, kmModel)  # ✅ store numeric clusters in results
    df = add_clustering_result_to_data(df, kmModel)  # ✅ update df with cresult
    save_to_file(df)

###################################################
    # ---- Added Debug Output ---- (addon)
    print("\nSample of final results:")
    print(df.head(10))
    print("\nwine_cluster_results.csv has been saved!")
