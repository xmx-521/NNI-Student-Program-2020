import pandas as pd
import numpy as np
from nni.feature_engineering.gradient_selector import FeatureGradientSelector

if __name__ == '__main__':

    # MARK: - Read dataset as df
    datasetAsDf = pd.read_csv('dataset/training.csv')

    # MARK: - Split dataset into input features and labels
    datasetFeaturesAsDf = datasetAsDf.iloc[:, 1:3595]
    datasetLabelsAsDf = datasetAsDf.iloc[:, 3595:3600]

    datasetCaAsDf = datasetLabelsAsDf.loc[:, 'Ca']
    datasetPAsDf = datasetLabelsAsDf.loc[:, 'P']
    datasetpHAsDf = datasetLabelsAsDf.loc[:, 'pH']
    datasetSOCAsDf = datasetLabelsAsDf.loc[:, 'SOC']
    datasetSandAsDf = datasetLabelsAsDf.loc[:, 'Sand']

    # MARK: - One-hot encode for feature Depth
    datasetFeaturesAsDf['Depth'] = pd.get_dummies(datasetFeaturesAsDf['Depth'])

    # MARK: - Transform df to array
    datasetFeaturesAsArray = datasetFeaturesAsDf.values

    fgs = FeatureGradientSelector()
