import copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from decimal import Decimal

from scipy.interpolate import interp1d

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, classification_report, mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, confusion_matrix)

import pickle

from PySide6.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QRadioButton, QGridLayout, QLabel, \
    QButtonGroup
from sklearn.tree import DecisionTreeClassifier

from ui_main import Ui_MainWindow

from pathlib import Path

import os.path


class NonePrep():
    def fit(self, *args):
        pass

    def transform(self, X):
        return X

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.filePath = Path()
        self.data = pd.DataFrame()
        self.selectFileBtn.clicked.connect(self.selectFile)
        self.ecgCurveBtn.clicked.connect(self.showEcgGraph)
        self.heartRateBtn.clicked.connect(self.showHeartRateGraph)
        self.splitBtn.clicked.connect(self.splitCsv)
        self.learnBtn.clicked.connect(self.learn)

        self.modelsBtbGroup = QButtonGroup()
        self.modelsBtbGroup.idClicked.connect(self.changeSelectedModel)

        self.preprocessing = [
            NonePrep(),
            StandardScaler(),
            RobustScaler(),
            MinMaxScaler(),
            Normalizer(),
        ]
        self.selectedPreprocessingIndex = 0
        self.preprocessingBtbGroup = QButtonGroup()
        self.preprocessingBtbGroup.addButton(self.prepRadBtnNone)
        self.preprocessingBtbGroup.addButton(self.prepRadBtnStandardScaler)
        self.preprocessingBtbGroup.addButton(self.prepRadBtnRobustScaler)
        self.preprocessingBtbGroup.addButton(self.prepRadBtnMinMaxScaler)
        self.preprocessingBtbGroup.addButton(self.prepRadBtnNormalizer)
        self.preprocessingBtbGroup.idClicked.connect(self.changeSelectedPreprocessing)
        self.radBtnVersionModelProd.setChecked(True)

        self.selectedVersionModelIndex = 0
        self.versionModelBtnGroup = QButtonGroup()
        self.versionModelBtnGroup.addButton(self.radBtnVersionModelTest)
        self.versionModelBtnGroup.addButton(self.radBtnVersionModelProd)
        self.versionModelBtnGroup.addButton(self.radBtnVersionModelAll)
        self.versionModelBtnGroup.idClicked.connect(self.changeSelectedVersionModel)

        self.modelVersions = [
            'test',
            'prod',
            'all',
        ]

        def createModels(model):
            modelDataEmpty = {
                                'print': '',
                                'test': {
                                    'model': copy.deepcopy(model),
                                    'best_params': {},
                                    'accuracy': 0,
                                    'confusion_matrix': [],
                                    'recall': 0,
                                    'std': 0,
                                },
                                'prod': {
                                    'model': copy.deepcopy(model),
                                    'best_params': {},
                                    'accuracy': 0,
                                    'confusion_matrix': [],
                                    'recall': 0,
                                    'std': 0,
                                }
                            }
            return modelDataEmpty


        seed = 42
        n_classes = [0, 1, 2]
        modelsList = [
            {
                'models': createModels(RandomForestClassifier(random_state=seed, n_jobs=-1)),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [50, 100, 150, 200, 250],
                    'max_depth': [None, 10, 20],
                    'min_samples_leaf': [1, 2, 4],
                }
            },
            {
                'models': createModels(ExtraTreesClassifier()),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [50, 100, 150, 200, 250],
                    'max_depth': [None, 10, 20],
                    'min_samples_leaf': [1, 2, 4],
                }
            },
            {
                'models': createModels(DecisionTreeClassifier()),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_leaf': [1, 2, 4],
                }
            },
            {
                'models': createModels(KNeighborsClassifier(n_neighbors=10)),
                'params': {
                    'n_neighbors': np.arange(1, 51),
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                }
            },
            # {
            #     'models': createModels(MLPClassifier()),
            #     'params': {
            #         'solver': ['lbfgs', 'sgd', 'adam'],
            #         'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
            #         'alpha': 10.0 ** -np.arange(1, 10),
            #         # 'hidden_layer_sizes': np.arange(10, 15),
            #     }
            # },
            # {
            #     'models': createModels(BaggingClassifier()),
            #     'params': {
            #         'bootstrap': [True, False],
            #         'bootstrap_features': [True, False],
            #         'n_estimators': [5, 10, 15, 100, 200, 300],
            #         'max_samples': [0.6, 0.8, 1.0],
            #         'max_features': [0.6, 0.8, 1.0],
            #         # 'base_estimator__bootstrap': [True, False],
            #         # 'base_estimator__n_estimators': [100, 200, 300],
            #         # 'base_estimator__max_features': [0.6, 0.8, 1.0]
            #     }
            # },
            # {
            #     'models': createModels(AdaBoostClassifier(algorithm='SAMME')),
            #     'params': {
            #         'n_estimators': [50, 100, 200],
            #         'learning_rate': [0.1, 0.5, 1],
            #     }
            # },
            # {
            #     'models': createModels(GradientBoostingClassifier()),
            #     'params': {
            #         "loss": ["deviance"],
            #         "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            #         "min_samples_split": np.linspace(0.1, 0.5, 12),
            #         "min_samples_leaf": np.linspace(0.1, 0.5, 12),
            #         "max_depth": [3, 5, 8],
            #         "max_features": ["log2", "sqrt"],
            #         "criterion": ["friedman_mse", "squared_error"],
            #         "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            #         "n_estimators": [10, 50, 100],
            #     }
            # },
            # {
            #     'models': createModels(LogisticRegression()),
            #     'params': {
            #         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            #         'penalty': ['none', 'elasticnet', 'l1', 'l2'],
            #         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            #     }
            # },
            # {
            #     'models': createModels(LinearSVC()),
            #     'params': {
            #         'penalty': ['l1', 'l2'],
            #         'C': 0.01 * 10 ** np.arange(0, 5),
            #     }
            # },
            # {
            #     'models': createModels(SVC()),
            #     'params': {
            #         'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            #         'C': 0.01 * 10 ** np.arange(0, 5),
            #     }
            # },
            # {
            #     'models': createModels(GaussianNB()),
            #     'params': {
            #         'var_smoothing': np.logspace(0, -9, num=100),
            #         # 'priors': [None, [0.1, ] * len(n_classes), ],
            #     }
            # },
            # {
            #     'models': createModels(BernoulliNB()),
            #     'params': {
            #         'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
            #         'fit_prior': [True, False],
            #         # 'class_prior': [None, [0.1, ] * len(n_classes), ],
            #         'binarize': [None, 0.0, 8.5, 10.0]
            #     }
            # },
            # {
            #     'model': ComplementNB(),
            #     'data': copy.deepcopy(modelDataEmpty),
            #     'params': {
            #         'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, ],
            #         'fit_prior': [True, False],
            #         'norm': [True, False],
            #         'class_prior': [None, [0.1,]* len(n_classes), ]
            #     }
            # },
            # {
            #     'model': MultinomialNB(),
            #     'data': copy.deepcopy(modelDataEmpty),
            #     'params': {
            #         'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, ],
            #         'fit_prior': [True, False],
            #         'class_prior': [None, [0.1,]* len(n_classes), ]
            #     }
            # },
            # {
            #     'model': CategoricalNB(min_categories=200),
            #     'data': copy.deepcopy(modelDataEmpty),
            #     'params': {
            #         'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, ],
            #         'fit_prior': [True, False],
            #         'min_categories': [18, 25, 30],
            #         'class_prior': [None, [0.1,]* len(n_classes),]
            #     }
            # },
        ]
        self.models = [copy.deepcopy(modelsList) for i in range(len(self.preprocessing))]
        self.selectedModelIndex = 0

        if os.path.isfile('models.sav'):
            self.loadModels()
        else:
            self.learn()

    def loadModels(self):
        frames = []
        pathlist = Path('data/payload').glob('**/*.csv')
        for path in pathlist:
            frames.append(pd.read_csv(str(path)))
        data = pd.concat(frames, axis=0)
        data.reindex()

        X = data.iloc[:, 2:]
        y = data['stress']

        self.models = pickle.load(open('models.sav', 'rb'))
        column = 0
        for prep in self.preprocessing:
            prep.fit(X)
            X = prep.transform(X)

            models_list = self.models[column]
            row = 0
            for model_dict in models_list:
                if column == 0:
                    radBtn = QRadioButton(model_dict['models']['test']['model'].__class__.__name__)
                    self.modelsGridLayout.addWidget(radBtn, row, column)
                    self.modelsBtbGroup.addButton(radBtn)
                accuracyLabel = QLabel(model_dict['models']['print'])
                self.modelsGridLayout.addWidget(accuracyLabel, row, column + 1)
                row += 1
            column += 1

    def selectFile(self):
        dialog = QFileDialog(self)
        dialog.setNameFilter("CSV, TXT (*.csv *.txt)")
        if dialog.exec():
            fileNames = dialog.selectedFiles()
            self.fileName.setText(fileNames[0])
            self.filePath = Path(fileNames[0])
            if self.filePath.suffix == '.csv':
                self.readCsv()
            else:
                self.readTxt()

            time_start = self.data.iloc[0, 0]
            self.data['timestamp:'] -= time_start

            self.ecgCurveBtn.setEnabled(True)
            self.heartRateBtn.setEnabled(True)
            self.splitBtn.setEnabled(True)

    def readCsv(self):
        self.data = pd.read_csv(self.filePath)

    def readTxt(self):
        with open(self.filePath, 'r') as f:
            data_list = []
            rows = f.read().split('\n')
            columns = rows[0].split()
            rows = rows[1:]
            for row in rows:
                if row:
                    data_row = row.split()
                    data_row[0] = float(data_row[0][:-1].replace(',', '.'))
                    data_list.append(data_row)
            self.data = pd.DataFrame(data_list, columns=columns, dtype='float64')

    def showEcgGraph(self):
        all_time = np.array([])
        time_0 = self.data['timestamp:'][0]

        i = 1
        for index in range(len(self.data['timestamp:'])):
            time = self.data['timestamp:'][index]
            if time != time_0 or index == len(self.data['timestamp:']) - 1:
                if index != len(self.data['timestamp:']) - 1:
                    all_time = np.hstack([all_time, np.linspace(time_0, time, i)])
                else:
                    all_time = np.hstack([all_time, np.linspace(time_0, time_0 + 0.03, i)])
                time_0 = time
                i = 1
            else:
                i += 1

        fig, ax = plt.subplots()

        ax.plot(all_time, self.data['ADC'].to_numpy(dtype=int))
        ax.set_xlabel('Time')
        ax.set_ylabel('ADC')
        ax.grid(True)
        fig.canvas.manager.window.showMaximized()
        plt.subplots_adjust(bottom=0.059, top=0.985, left=0.037, right=0.992, hspace=0.2, wspace=0.2)
        plt.show()

    def showHeartRateGraph(self):
        orig_time = [self.data['timestamp:'][0]]

        all_time = np.array([])
        time_0 = self.data['timestamp:'][0]

        all_heart_rate_4s = [self.data['HeartRate4sAverage'][0]]
        all_heart_rate_30s = [self.data['HeartRate30sAverage'][0]]

        i = 1
        for index in range(len(self.data['timestamp:'])):
            time = self.data['timestamp:'][index]
            heart_rate_4s = self.data['HeartRate4sAverage'][index]
            heart_rate_30s = self.data['HeartRate30sAverage'][index]
            if time != time_0 or index == len(self.data['timestamp:']) - 1:
                if index != len(self.data['timestamp:']) - 1:
                    all_time = np.hstack([all_time, np.linspace(time_0, time, i)])
                    orig_time.append(time)
                else:
                    all_time = np.hstack([all_time, np.linspace(time_0, time_0 + 0.03, i)])
                    orig_time.append(time + 0.03)
                time_0 = time

                all_heart_rate_4s.append(heart_rate_4s)
                all_heart_rate_30s.append(heart_rate_30s)

                i = 1
            else:
                i += 1


        split_data, split_time = self.splitData()
        split_data = split_data.iloc[:, 2:]
        X = self.preprocessing[self.selectedPreprocessingIndex].transform(split_data)
        data_pred = []
        if self.modelVersions[self.selectedVersionModelIndex] != 'all':
            data_pred.append(self.models[self.selectedPreprocessingIndex][self.selectedModelIndex]['models']
                             [self.modelVersions[self.selectedVersionModelIndex]]['model'].predict(X))
        else:
            for version in self.modelVersions:
                if version != 'all':
                    data_pred.append(self.models[self.selectedPreprocessingIndex][self.selectedModelIndex]['models']
                                     [version]['model'].predict(X))

        stress_colors = ['r', 'orange']

        fig, ax = plt.subplots()
        fig.set_facecolor('#292929')
        ax.set_facecolor('#575757')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
        x_start = [0 for i in range(len(data_pred))]
        x_end = [0 for i in range(len(data_pred))]
        stress_type_prev = [0 for i in range(len(data_pred))]
        height_area = 110
        y0_area = 40
        for i in range(len(data_pred[0])):
            for j in range(len(data_pred)):
                stress = data_pred[j][i]
                if stress != stress_type_prev[j] and stress_type_prev[j] != 0:
                    x_end[j] = split_time[i] - x_start[j]
                    ax.add_patch(Rectangle((x_start[j], y0_area + height_area / len(data_pred) * j), x_end[j], height_area / len(data_pred), facecolor=stress_colors[j], alpha=stress_type_prev[j] * 0.1))
                    stress_type_prev[j] = stress
                    if stress != 0:
                        x_start[j] = split_time[i]

                elif stress != 0 and stress_type_prev[j] != stress:
                    x_start[j] = split_time[i]
                    stress_type_prev[j] = stress

        # ax[0].plot(all_time, all_heart_rate_4s, label='Heart Rate 4s Average')
        ax.plot(orig_time, all_heart_rate_4s, label='Heart Rate 4s Average')
        ax.plot(orig_time, all_heart_rate_30s, label='Heart Rate 30s Average')
        # ax[0].plot(split_time, data_pred, label="Stress")
        # ax[0].fill_between(split_time, y1=data_pred, y2=40)
        ax.set_xlabel('Time')
        ax.set_ylabel('HeartRate')
        ax.grid(True)
        ax.legend()
        fig.canvas.manager.window.showMaximized()
        plt.subplots_adjust(bottom=0.059, top=0.985, left=0.037, right=0.992, hspace=0.2, wspace=0.2)
        plt.show()

    def splitCsv(self):
        data, _ = self.splitData()
        print(data)
        data.to_csv(f'data/payload/{self.filePath.stem}.csv', index=False)

    def splitData(self):
        data = self.data.drop_duplicates(subset=['timestamp:'], ignore_index=True)
        data.reindex()
        data.drop('ADC', axis=1, inplace=True)

        split_step = 160

        split_time = []

        split_data_list = []
        columns = ['stress', 'time']
        for j in range(split_step):
            columns.append(f'HeartRate4sAverage{j}')
            columns.append(f'HeartRate30sAverage{j}')

        i = 0
        while (i+1) * split_step < len(data):
            split_time.append(data['timestamp:'][split_step*i])
            row_data = [0,
                        f'{round(data['timestamp:'][split_step*i], 3)}-{round(data['timestamp:'][split_step*(i+1)], 3)}']
            # print(split_step*(i+1))
            for ind in range(int(split_step*i), int(split_step*(i+1))):
                row_data.append(int(data['HeartRate4sAverage'][ind]))
                row_data.append(int(data['HeartRate30sAverage'][ind]))

            split_data_list.append(row_data)
            i += 0.5

        split_data = pd.DataFrame(split_data_list, columns=columns)
        return split_data, split_time

    def changeSelectedModel(self, id):
        print(abs(id) - 2)
        self.selectedModelIndex = abs(id) - 2

    def changeSelectedPreprocessing(self, id):
        print(abs(id) - 2)
        self.selectedPreprocessingIndex = abs(id) - 2

    def changeSelectedVersionModel(self, id):
        print(abs(id) - 2)
        self.selectedVersionModelIndex = abs(id) - 2

    # def confusion_matrix_scorer(self, clf, X, y):
    #     y_pred = clf.predict(X)
    #     cm = confusion_matrix(y, y_pred)
    #     recall_0 = cm[0][0] / (cm[0][0] + cm[0][1])
    #     recall_1 = cm[1][1] / (cm[1][0] + cm[1][1])
    #     return recall_0 * recall_1


    def learn(self, optim=False):
        def confusion_matrix_scorer(clf, X, y):
            y_pred = clf.predict(X)
            cm = confusion_matrix(y, y_pred)
            recall_0 = cm[0][0] / (cm[0][0] + cm[0][1] + cm[0][2])
            recall_1 = cm[1][1] / (cm[1][0] + cm[1][1] + cm[1][2])
            recall_2 = cm[2][1] / (cm[2][0] + cm[2][1] + cm[2][2])
            return recall_0 * recall_1 * recall_2
        print('Start learn')
        frames = []
        pathlist = Path('data/payload').glob('**/*.csv')
        for path in pathlist:
            frames.append(pd.read_csv(str(path)))
        data = pd.concat(frames, axis=0)
        data.reindex()

        X = data.iloc[:, 2:]
        y = data['stress']

        test_size = 0.5
        seed = 42

        # Настройка параметров оценивания алгоритма
        num_folds = 10
        n_estimators = 100
        scoring = 'recall'

        self.modelsGridLayout.children().clear()

        column = 0
        for prep in self.preprocessing:
            prep.fit(X)
            X = prep.transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                                stratify=y, shuffle=True)
            models_list = self.models[column]
            row = 0
            for i in range(len(models_list)):
                try:
                    model_dict = models_list[i]
                    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
                    model_name = model_dict['models']['test']['model'].__class__.__name__
                    params = model_dict['params']
                    print(model_name, prep)

                    if optim:
                        grid_search = GridSearchCV(model_dict['models']['test']['model'], params, cv=kfold, n_jobs=-1, scoring=confusion_matrix_scorer)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        model_dict['models']['test']['model'] = grid_search.best_estimator_
                        model_dict['models']['test']['best_params'] = best_params
                        print(best_params)

                    cv_results = cross_val_score(model_dict['models']['test']['model'], X_train, y_train, cv=kfold, n_jobs=-1, scoring=confusion_matrix_scorer)
                    model_dict['models']['test']['model'].fit(X_train, y_train)
                    y_pred = model_dict['models']['test']['model'].predict(X_test)
                    accuracy = model_dict['models']['test']['model'].score(X_test, y_test)
                    print(cv_results.mean(), cv_results)
                    model_dict['models']['test']['accuracy'] = accuracy
                    # model_dict['data']['test']['recall'] = cv_results.mean()
                    # model_dict['data']['test']['std'] = cv_results.std()
                    # print(accuracy)
                    classificationReport = classification_report(y_test, y_pred)
                    print(classificationReport)
                    matrix = confusion_matrix(y_test, y_pred)
                    model_dict['models']['test']['confusion_matrix'] = matrix
                    recall_0 = matrix[0][0]/(matrix[0][0]+matrix[0][1])
                    recall_1 = matrix[1][1]/(matrix[1][0]+matrix[1][1])
                    print(recall_0 * recall_1)
                    print(matrix)
                    # print("MAE", mean_absolute_error(y_test, y_pred))
                    # print("MAPE", mean_absolute_percentage_error(y_test, y_pred))
                    if column == 0:
                        radBtn = QRadioButton(model_name)
                        self.modelsGridLayout.addWidget(radBtn, row, column)
                        self.modelsBtbGroup.addButton(radBtn)

                    if optim:
                        grid_search = GridSearchCV(model_dict['models']['prod']['model'], params, cv=kfold, n_jobs=-1, scoring=confusion_matrix_scorer)
                        grid_search.fit(X, y)
                        best_params = grid_search.best_params_
                        model_dict['models']['prod']['model'] = grid_search.best_estimator_
                        model_dict['models']['prod']['best_params'] = best_params

                    # cv_results = cross_val_score(model_dict['model'], X, y, cv=kfold, n_jobs=-1, scoring=scoring)
                    model_dict['models']['prod']['model'].fit(X, y)
                    y_pred = model_dict['models']['prod']['model'].predict(X)
                    accuracy_full = model_dict['models']['prod']['model'].score(X, y)
                    # print(accuracy_full)
                    # print(classification_report(y, y_pred))
                    matrix_full = confusion_matrix(y, y_pred)
                    print(matrix_full)
                    recall_full_0 = matrix_full[0][0]/(matrix_full[0][0]+matrix_full[0][1])
                    recall_full_1 = matrix_full[1][1]/(matrix_full[1][0]+matrix_full[1][1])
                    model_dict['models']['print'] = (f'{recall_0 * recall_1:.3f} | {recall_0:.3f} / {recall_1:.3f} => '
                                  f'{recall_full_0 * recall_full_1:.3f} | {recall_full_0:.3f} / {recall_full_1:.3f}')
                    accuracyLabel = QLabel(model_dict['models']['print'])
                    self.modelsGridLayout.addWidget(accuracyLabel, row, column + 1)

                    model_dict['models']['prod']['accuracy'] = accuracy_full
                    # model_dict['models']['prod']['recall'] = cv_results.mean()
                    # model_dict['models']['prod']['std'] = cv_results.std()
                    model_dict['models']['prod']['confusion_matrix'] = matrix_full
                except Exception as e:
                    print(e)
                row += 1
            column += 1

        # print(self.models)
        filename = 'models.sav'
        pickle.dump(self.models, open(filename, 'wb'))


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()