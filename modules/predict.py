import json

import dill
import os
import dill
from datetime import datetime
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


filename = '/Users/yaandreev/airflow_hw/data/models/cars_pipe_202210231944.pkl'


def predict():
    import pandas as pd
    with open(filename, 'rb') as file:
        object_to_load = dill.load(file)
    df_predicted = pd.DataFrame()
    PATH = "/Users/yaandreev/airflow_hw/data/test/"
    for test_name_data in os.listdir(PATH):
        with open('/Users/yaandreev/airflow_hw/data/test/'+test_name_data, 'r') as j:
            import pandas as pd
            data = json.load(j)
            df = pd.DataFrame([data])
            pred = object_to_load.predict(df)
            x = {'car_id': df.id, 'prediction': pred}
            df_predicted_chapter = pd.DataFrame.from_dict(x)
            df_predicted = pd.concat([df_predicted_chapter, df_predicted], axis=0)

    df_predicted.to_csv(f'/Users/yaandreev/airflow_hw/data/predictions/df_predicted_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
