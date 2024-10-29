# For this particular project we will not be taking `disgust` and `fear` emotion into account

import pandas as pd
import tarfile

def filter_dataframe(df, conditions):
    filtered_df = df.copy()

    for column, value in conditions.items():
        filtered_df = filtered_df[~filtered_df[column].isin(value)]

    return filtered_df

def extract_file(tar_path, extract_path):
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)
        print(f"Successfully extracted to {extract_path}")
    except Exception as e:
        print(f"Error extracting tar file: {str(e)}")


if __name__ == "__main__":
    extract_file("..\\challenges-in-representation-learning-facial-expression-recognition-challenge\\fer2013.tar.gz","..\\challenges-in-representation-learning-facial-expression-recognition-challenge\\")
    df = pd.read_csv("..\\challenges-in-representation-learning-facial-expression-recognition-challenge\\fer2013\\fer2013.csv")

    conditions = {'emotion': [1,2]}
    filtered_df = filter_dataframe(df, conditions)


    df.to_csv('.\\filtered_data.csv', index=False)
    print("Data Filtered Successfully.")