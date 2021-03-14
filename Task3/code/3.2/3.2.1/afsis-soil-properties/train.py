import pandas as pd
import numpy as np
from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
from sklearn.model_selection import train_test_split
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    # MARK: - Read dataset as df
    datasetAsDf = pd.read_csv('dataset/training.csv')

    # MARK: - Split dataset into input features and labels
    datasetFeaturesAsDf = datasetAsDf.iloc[:, 1:3595]
    datasetLabelsAsDf = datasetAsDf.iloc[:, 3595:3600].astype(float)

    datasetCaAsDf = datasetLabelsAsDf.loc[:, 'Ca'].astype(float)
    datasetPAsDf = datasetLabelsAsDf.loc[:, 'P'].astype(float)
    datasetpHAsDf = datasetLabelsAsDf.loc[:, 'pH'].astype(float)
    datasetSOCAsDf = datasetLabelsAsDf.loc[:, 'SOC'].astype(float)
    datasetSandAsDf = datasetLabelsAsDf.loc[:, 'Sand'].astype(float)

    # MARK: - One-hot encode for feature Depth
    datasetFeaturesAsDf['Depth'] = pd.get_dummies(datasetFeaturesAsDf['Depth'])
    datasetFeaturesAsDf = datasetFeaturesAsDf.astype(float)

    # MARK: - Transform df to array
    datasetFeaturesAsArray = datasetFeaturesAsDf.values
    datasetLabelsAsArray = datasetLabelsAsDf.values

    datasetCaAsArray = datasetCaAsDf.values
    datasetPAsArray = datasetPAsDf.values
    datasetpHAsArray = datasetpHAsDf.values
    datasetSOCAsArray = datasetSOCAsDf.values
    datasetSandAsArray = datasetSandAsDf.values

    # MARK: - Reshape array as Matrix
    datasetCaAsMatrix = np.matrix(datasetCaAsArray.reshape(
        (-1, 1))).astype(float)
    datasetPAsMatrix = np.matrix(datasetPAsArray.reshape(
        (-1, 1))).astype(float)
    datasetpHAsMatrix = np.matrix(datasetpHAsArray.reshape(
        (-1, 1))).astype(float)
    datasetSOCAsMatrix = np.matrix(datasetSOCAsArray.reshape(
        (-1, 1))).astype(float)
    datasetSandAsMatrix = np.matrix(datasetSandAsArray.reshape(
        (-1, 1))).astype(float)

    # MARK: - NNI Feature Engineering and Kernel Ridge Regression for every target

    # MARK: - Ca
    print('Selecting features for Ca\n')
    XTrainCa, XTestCa, yTrainCa, yTestCa = train_test_split(
        datasetFeaturesAsDf.copy(),
        datasetCaAsMatrix,
        test_size=0.2,
        random_state=42)

    print(XTrainCa.shape, yTrainCa.shape)

    fgsCa = FeatureGradientSelector(verbose=1,
                                    n_epochs=100,
                                    learning_rate=1e-1,
                                    shuffle=True)
    fgsCa.fit(XTrainCa, yTrainCa)
    print(fgsCa.get_selected_features())
    print('\n')

    CaSelectedFeatureIndices = fgsCa.get_selected_features()
    XTrainCa = XTrainCa.iloc[:, CaSelectedFeatureIndices]
    XTestCa = XTestCa.iloc[:, CaSelectedFeatureIndices]

    krCa = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
    krCa.fit(XTrainCa, yTrainCa)
    yPredCa = krCa.predict(XTestCa)
    print(mean_squared_error(yPredCa, yTestCa))
    print('\n')

    # MARK: - P
    print('Selecting features for P\n')
    XTrainP, XTestP, yTrainP, yTestP = train_test_split(
        datasetFeaturesAsDf.copy(),
        datasetPAsMatrix,
        test_size=0.2,
        random_state=42)

    print(XTrainP.shape, yTrainP.shape)

    fgsP = FeatureGradientSelector(verbose=1,
                                   n_epochs=100,
                                   learning_rate=1e-1,
                                   n_features=20,
                                   shuffle=True)
    fgsP.fit(XTrainP, yTrainP)
    print(fgsP.get_selected_features())
    print('\n')

    PSelectedFeatureIndices = fgsP.get_selected_features()
    XTrainP = XTrainP.iloc[:, PSelectedFeatureIndices]
    XTestP = XTestP.iloc[:, PSelectedFeatureIndices]

    krP = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
    krP.fit(XTrainP, yTrainP)
    yPredP = krP.predict(XTestP)
    print(mean_squared_error(yPredP, yTestP))
    print('\n')

    # MARK: - pH
    print('Selecting features for pH\n')
    XTrainpH, XTestpH, yTrainpH, yTestpH = train_test_split(
        datasetFeaturesAsDf.copy(),
        datasetpHAsMatrix,
        test_size=0.2,
        random_state=42)

    print(XTrainpH.shape, yTrainpH.shape)

    fgspH = FeatureGradientSelector(verbose=1,
                                    n_epochs=100,
                                    learning_rate=1e-1,
                                    shuffle=True)
    fgspH.fit(XTrainpH, yTrainpH)
    print(fgspH.get_selected_features())
    print('\n')

    pHSelectedFeatureIndices = fgspH.get_selected_features()
    XTrainpH = XTrainpH.iloc[:, pHSelectedFeatureIndices]
    XTestpH = XTestpH.iloc[:, pHSelectedFeatureIndices]

    krpH = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
    krpH.fit(XTrainpH, yTrainpH)
    yPredpH = krpH.predict(XTestpH)
    print(mean_squared_error(yPredpH, yTestpH))
    print('\n')

    # MARK: - SOC
    print('Selecting features for SOC\n')
    XTrainSOC, XTestSOC, yTrainSOC, yTestSOC = train_test_split(
        datasetFeaturesAsDf.copy(),
        datasetSOCAsMatrix,
        test_size=0.2,
        random_state=42)

    print(XTrainSOC.shape, yTrainSOC.shape)

    fgsSOC = FeatureGradientSelector(verbose=1,
                                     n_epochs=100,
                                     learning_rate=1e-1,
                                     shuffle=True)
    fgsSOC.fit(XTrainSOC, yTrainSOC)
    print(fgsSOC.get_selected_features())
    print('\n')

    SOCSelectedFeatureIndices = fgsSOC.get_selected_features()
    XTrainSOC = XTrainSOC.iloc[:, SOCSelectedFeatureIndices]
    XTestSOC = XTestSOC.iloc[:, SOCSelectedFeatureIndices]

    krSOC = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
    krSOC.fit(XTrainSOC, yTrainSOC)
    yPredSOC = krSOC.predict(XTestSOC)
    print(mean_squared_error(yPredSOC, yTestSOC))
    print('\n')

    # MARK: - Sand
    print('Selecting features for Sand\n')
    XTrainSand, XTestSand, yTrainSand, yTestSand = train_test_split(
        datasetFeaturesAsDf.copy(),
        datasetSandAsMatrix,
        test_size=0.2,
        random_state=42)

    print(XTrainSand.shape, yTrainSand.shape)

    fgsSand = FeatureGradientSelector(verbose=1,
                                      n_epochs=100,
                                      learning_rate=1e-1,
                                      shuffle=True)

    fgsSand.fit(XTrainSand, yTrainSand)
    print(fgsSand.get_selected_features())
    print('\n')

    SandSelectedFeatureIndices = fgsSand.get_selected_features()
    XTrainSand = XTrainSand.iloc[:, SandSelectedFeatureIndices]
    XTestSand = XTestSand.iloc[:, SandSelectedFeatureIndices]

    krSand = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
    krSand.fit(XTrainSand, yTrainSand)
    yPredSand = krSand.predict(XTestSand)
    print(mean_squared_error(yPredSand, yTestSand))
    print('\n')

    # Predict sorted_test
    test = pd.read_csv('dataset/sorted_test.csv')
    test = test.iloc[:, 1:3595]
    test['Depth'] = pd.get_dummies(test['Depth'])
    test = test.astype(float)

    testCa = test.iloc[:, CaSelectedFeatureIndices]
    testP = test.iloc[:, PSelectedFeatureIndices]
    testpH = test.iloc[:, pHSelectedFeatureIndices]
    testSOC = test.iloc[:, SOCSelectedFeatureIndices]
    testSand = test.iloc[:, SandSelectedFeatureIndices]

    testPredCa = krCa.predict(testCa)
    testPredP = krP.predict(testP)
    testPredpH = krpH.predict(testpH)
    testPredSOC = krSOC.predict(testSOC)
    testPredSand = krSand.predict(testSand)

    sub = pd.read_csv('dataset/sample_submission.csv')
    sub['Ca'] = testPredCa
    sub['P'] = testPredP
    sub['pH'] = testPredpH
    sub['SOC'] = testPredSOC
    sub['Sand'] = testPredSand

    sub.to_csv('final_answer.csv', index=False)
