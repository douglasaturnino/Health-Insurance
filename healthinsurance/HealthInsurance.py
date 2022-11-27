import os
import pickle
import numpy as np
import pandas as pd


class HealthInsurance():
    def __init__(self):
        self.home_path = os.getcwd()
        self.annual_premiun_scaler = pickle.load(open(os.path.join(
            self.home_path, 'features', 'annual_premiun_scaler.pkl'), 'rb'))
        self.age_scaler = pickle.load(
            open(os.path.join(self.home_path, 'features', 'age_scaler.pkl'), 'rb'))
        self.vintage_scaler = pickle.load(
            open(os.path.join(self.home_path, 'features', 'vintage_scaler.pkl'), 'rb'))
        self.target_encode_gender_scaler = pickle.load(open(os.path.join(
            self.home_path, 'features', 'target_encode_gender_scaler.pkl'), 'rb'))
        self.target_encode_region_code_scaler = pickle.load(open(os.path.join(
            self.home_path, 'features', 'target_encode_region_code_scaler.pkl'), 'rb'))
        self.fe_policy_sales_channel_scaler = pickle.load(open(os.path.join(
            self.home_path, 'features', 'fe_policy_sales_channel_scaler.pkl'), 'rb'))

    def data_cleaning(self, data):
        # 1.1. Rename Columns
        cols_new = ['annual_premium', 'vintage', 'age', 'region_code', 'previously_insured', 'vehicle_damage', 'policy_sales_channel']
                   
        
        return data[cols_new]

    def feature_engineering(self, data):
        # vehicle age
        #data['vehicle_age'] = data['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else                                                        'between_1_2_year' if x == '1-2 Year' else 'below_1_year')
        # vehicle damage
        data['vehicle_damage'] = data['vehicle_damage'].apply(
            lambda x: 1 if x == 'Yes' else 0)
        return data

    def data_preparation(self, data):
        # annual_premium
        data['annual_premium'] = self.annual_premiun_scaler.transform(
            data[['annual_premium']].values)

        # Age
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # vintage
        data['vintage'] = self.vintage_scaler.transform(
            data[['vintage']].values)

        # gender
        # data.loc[:, 'gender'] = data['gender'].map(self.target_encode_gender_scaler)

        # region_code
        data.loc[:, 'region_code'] = data['region_code'].map(
            self.target_encode_region_code_scaler)

        # vehicle_age
        # data = pd.get_dummies(data, prefix='vehicle_age', columns=['vehicle_age'])

        # policy_sales_channel
        data.loc[:, 'policy_sales_channel'] = data['policy_sales_channel'].map(
            self.fe_policy_sales_channel_scaler)

        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'previously_insured', 'vehicle_damage', 'policy_sales_channel']
                        

        return data[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # model prediction
        pred = model.predict_proba(test_data)
        # join prediction into original data
        original_data['response'] = pred[:, 1].tolist()

        return original_data.to_json(orient='records', date_format='iso')
