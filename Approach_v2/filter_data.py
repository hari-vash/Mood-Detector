# For this particular project we will not be taking `disgust` and `fear` emotion into account

import pandas as pd

def filter_dataframe(df, conditions):
    filtered_df = df.copy()

    for column, value in conditions.items():
        filtered_df = filtered_df[~filtered_df[column].isin(value)]

    return filtered_df

if __name__ == "__main__":
    df = pd.read_csv("..//challenges-in-representation-learning-facial-expression-recognition-challenge//icml_face_data.csv")

    conditions = {'emotion': [1,2]}
    filtered_df = filter_dataframe(df, conditions)


    df.to_csv('.//filtered_data.csv', index=False)
    print("Data Filtered Successfully.")