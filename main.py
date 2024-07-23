import copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy import signal

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, classification_report, mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, confusion_matrix, recall_score, f1_score)

import pickle

from PySide6.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QRadioButton, QGridLayout, QLabel, \
    QButtonGroup
from sklearn.tree import DecisionTreeClassifier

from ui_main import Ui_MainWindow

from pathlib import Path

import os.path


class NonePrep:
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

        # Список предобработок
        self.preprocessing = [
            NonePrep(),
            StandardScaler(),
            RobustScaler(),
            MinMaxScaler(),
            Normalizer(),
        ]
        self.selectedPreprocessingIndex = 0
        self.selectedVersionModelIndex = 1

        self.init_ui()

        self.modelVersions = [
            'test',
            'prod',
            'all',
        ]

        def create_models(model):
            modelDataEmpty = {
                'print': '',
                'test': {
                    'model': copy.deepcopy(model),
                    'best_params': {},
                    'accuracy': 0,
                    'confusion_matrix': [],
                    'f1_macro': 0,
                },
                'prod': {
                    'model': copy.deepcopy(model),
                    'best_params': {},
                    'accuracy': 0,
                    'confusion_matrix': [],
                    'f1_macro': 0,
                }
            }
            return modelDataEmpty

        seed = 42
        # Список моделей
        modelsList = [
            {
                # {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 50}
                'models': create_models(RandomForestClassifier(random_state=seed, n_jobs=-1, verbose=0,
                                                              criterion='entropy', n_estimators=50,
                                                              min_samples_leaf=3, max_depth=5)),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [25, 50, 75, 100, 150, 200, 250],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 0.5],
                    'max_features': [None, 'sqrt', 'log2', 0.3, 0.5],
                    'min_samples_leaf': [1, 2, 3, 4, 0.5],
                    'bootstrap': [True, False]
                }
            },
            {
                'models': create_models(ExtraTreesClassifier()),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [25, 50, 75, 100, 150, 200, 250],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 3, 4, 5, 6],
                    'max_features': [None, 'sqrt', 'auto', 'log2', 0.3, 0.5],
                    'bootstrap': [True, False]
                }
            },
            {
                # {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 5}
                'models': create_models(DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5)),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 3, 4, 5, 6],
                }
            },
            {
                # {'algorithm': 'auto', 'n_neighbors': np.int64(3)}
                'models': create_models(KNeighborsClassifier(n_neighbors=3)),
                'params': {
                    'n_neighbors': np.arange(1, 50),
                    'algorithm': ['auto'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                }
            },
            {
                'models': create_models(MLPClassifier()),
                'params': {
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
                    'alpha': 10.0 ** -np.arange(1, 10),
                    # 'hidden_layer_sizes': np.arange(10, 15),
                }
            },
            {
                'models': create_models(BaggingClassifier()),
                'params': {
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False],
                    'n_estimators': [5, 10, 15, 100, 200, 300],
                    'max_samples': [0.6, 0.8, 1.0],
                    'max_features': [0.6, 0.8, 1.0],
                }
            },
            {
                'models': create_models(AdaBoostClassifier(algorithm='SAMME')),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.5, 1],
                }
            },
            {
                'models': create_models(GradientBoostingClassifier(verbose=1, n_estimators=150, learning_rate=0.075)),
                'params': {
                    "loss": ["deviance"],
                    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                    "min_samples_split": np.linspace(0.1, 0.5, 12),
                    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                    "max_depth": [None, 3, 5, 8],
                    "max_features": [None, "log2", "sqrt"],
                    "criterion": ["friedman_mse", "squared_error"],
                    "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                    "n_estimators": [10, 50, 100, 150],
                }
            },
            {
                'models': create_models(LogisticRegression()),
                'params': {
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'penalty': ['none', 'elasticnet', 'l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                }
            },
            {
                'models': create_models(LinearSVC()),
                'params': {
                    'penalty': ['l1', 'l2'],
                    'C': 0.01 * 10 ** np.arange(0, 5),
                }
            },
            {
                'models': create_models(SVC()),
                'params': {
                    # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'kernel': ['rbf'],
                    'C': 0.01 * 10 ** np.arange(0, 5),
                }
            },
            {
                'models': create_models(GaussianNB()),
                'params': {
                    'var_smoothing': np.logspace(0, -9, num=100),
                }
            },
            {
                'models': create_models(BernoulliNB()),
                'params': {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
                    'fit_prior': [True, False],
                    'binarize': [None, 0.0, 8.5, 10.0]
                }
            },
        ]
        self.selectedModelIndex = 0

        if os.path.isfile('models.sav'):
            self.load_models()
        else:
            self.models = [copy.deepcopy(modelsList) for i in range(len(self.preprocessing))]
            self.learn()

    def init_ui(self):
        self.selectFileBtn.clicked.connect(self.select_file)
        self.ecgCurveBtn.clicked.connect(self.show_ecg_graph)
        self.heartRateBtn.clicked.connect(self.show_heart_rate_graph)
        self.splitBtn.clicked.connect(self.split_csv)
        self.splitAllBtn.clicked.connect(self.split_all_csv)
        self.learnBtn.clicked.connect(self.learn)

        self.modelsBtnGroup = QButtonGroup()
        self.modelsBtnGroup.idClicked.connect(self.change_selected_model)

        self.preprocessingBtnGroup = QButtonGroup()
        self.preprocessingBtnGroup.addButton(self.prepRadBtnNone)
        self.preprocessingBtnGroup.addButton(self.prepRadBtnStandardScaler)
        self.preprocessingBtnGroup.addButton(self.prepRadBtnRobustScaler)
        self.preprocessingBtnGroup.addButton(self.prepRadBtnMinMaxScaler)
        self.preprocessingBtnGroup.addButton(self.prepRadBtnNormalizer)
        self.preprocessingBtnGroup.idClicked.connect(self.change_selected_preprocessing)
        self.radBtnVersionModelProd.setChecked(True)

        self.versionModelBtnGroup = QButtonGroup()
        self.versionModelBtnGroup.addButton(self.radBtnVersionModelTest)
        self.versionModelBtnGroup.addButton(self.radBtnVersionModelProd)
        self.versionModelBtnGroup.addButton(self.radBtnVersionModelAll)
        self.versionModelBtnGroup.idClicked.connect(self.change_selected_version_model)

    # Загрузка моделей из файла
    def load_models(self) -> None:
        frames = []
        path_list = Path('data/payload').glob('**/*.csv')
        for path in path_list:
            frames.append(pd.read_csv(str(path)))
        data = pd.concat(frames, axis=0)
        data.reindex()

        X = data.iloc[:, 2:]

        self.models = pickle.load(open('models.sav', 'rb'))
        column = 0
        for prep in self.preprocessing:
            prep.fit(X)
            models_list = self.models[column]
            row = 0
            for model_dict in models_list:
                if column == 0:
                    radBtn = QRadioButton(model_dict['models']['test']['model'].__class__.__name__)
                    self.modelsGridLayout.addWidget(radBtn, row, column)
                    self.modelsBtnGroup.addButton(radBtn)
                accuracyLabel = QLabel(model_dict['models']['print'])
                self.modelsGridLayout.addWidget(accuracyLabel, row, column + 1)
                row += 1
            column += 1

    # Выбор файла
    def select_file(self) -> None:
        dialog = QFileDialog(self)
        dialog.setNameFilter("TXT (*.txt)")
        if dialog.exec():
            file_names = dialog.selectedFiles()
            self.fileName.setText(file_names[0])
            self.filePath = Path(file_names[0])
            self.read_txt()

            time_start = self.data.iloc[0, 0]
            self.data['timestamp:'] -= time_start

            self.ecgCurveBtn.setEnabled(True)
            self.heartRateBtn.setEnabled(True)
            self.splitBtn.setEnabled(True)

    # Прочитать файл с данными
    def read_txt(self, path: str = None) -> None | pd.DataFrame:
        if path is None:
            file_path = self.filePath
        else:
            file_path = path
        with open(file_path, 'r') as f:
            data_list = []
            rows = f.read().split('\n')
            columns = rows[0].split()
            rows = rows[1:]
            for row in rows:
                if row:
                    data_row = row.split()
                    data_row[0] = float(data_row[0][:-1].replace(',', '.'))
                    data_list.append(data_row)
            if path is None:
                self.data = pd.DataFrame(data_list, columns=columns, dtype='float64')
            else:
                return pd.DataFrame(data_list, columns=columns, dtype='float64')

    # Построить график ЭКГ
    def show_ecg_graph(self) -> None:
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
        fig.set_facecolor('#292929')
        ax.set_facecolor('#575757')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')

        ax.plot(all_time, self.data['ADC'].to_numpy(dtype=int), color='#29ff42ff')
        ax.set_xlabel('Time')
        ax.set_ylabel('ADC')
        ax.grid(True)
        fig.canvas.manager.window.showMaximized()
        plt.subplots_adjust(bottom=0.059, top=0.985, left=0.037, right=0.992, hspace=0.2, wspace=0.2)
        plt.show()

    # Построить график ЧСС
    def show_heart_rate_graph(self) -> None:
        print(self.data.describe())
        mean_hr = (self.data.mean()['HeartRate4sAverage'] + self.data.mean()['HeartRate30sAverage']) / 2
        percentiles = self.data.quantile(.5)
        average_percentile = (percentiles['HeartRate4sAverage'] + percentiles['HeartRate30sAverage']) / 2
        min_values = self.data.min()
        min_hr = min([min_values['HeartRate4sAverage'], min_values['HeartRate30sAverage']])
        max_values = self.data.max()
        max_hr = max([max_values['HeartRate4sAverage'], max_values['HeartRate30sAverage']])

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

        all_heart_rate_4s_filtered = signal.savgol_filter(all_heart_rate_4s, 70, 3)
        all_heart_rate_30s_filtered = signal.savgol_filter(all_heart_rate_30s, 70, 3)

        split_data, split_time = self.split_data()
        # print(split_data)
        time_periods = split_data.iloc[:, 1:2]
        X_raw = split_data.iloc[:, 2:]
        if self.preprocessing[self.selectedPreprocessingIndex].__class__.__name__ != 'StandardScaler':
            X_raw = X_raw.to_numpy()
        X = self.preprocessing[self.selectedPreprocessingIndex].transform(X_raw)
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

        # Отрисовка областей со стрессом
        x_start = [0 for i in range(len(data_pred))]
        x_end = [0 for i in range(len(data_pred))]
        stress_type_prev = [0 for i in range(len(data_pred))]
        for i in range(len(data_pred[0])):
            for j in range(len(data_pred)):
                if len(data_pred) == 2:
                    if j == 0:
                        y0_area = min_hr
                        height_area = average_percentile - y0_area
                    elif j == 1:
                        y0_area = average_percentile
                        height_area = max_hr - y0_area
                else:
                    y0_area = min_hr
                    height_area = max_hr - y0_area
                stress = data_pred[j][i]
                if stress != stress_type_prev[j] and stress_type_prev[j] != 0:
                    period_start, period_end = time_periods.loc[i, 'time'].split('-')
                    ax.add_patch(Rectangle((x_start[j], y0_area), x_end[j], height_area, facecolor=stress_colors[j],
                                           alpha=stress_type_prev[j] * 0.1))
                    stress_type_prev[j] = stress
                    if stress != 0:
                        x_start[j] = x_start[j] + x_end[j]
                        x_end[j] = float(period_end) - x_start[j]

                elif stress != 0 and stress_type_prev[j] != stress:
                    period_start, period_end = time_periods.loc[i, 'time'].split('-')
                    x_start[j] = (float(period_start) + float(period_end)) / 2
                    x_end[j] = float(period_end) - x_start[j]
                    stress_type_prev[j] = stress
                elif stress != 0 and stress == stress_type_prev[j]:
                    period_start, period_end = time_periods.loc[i, 'time'].split('-')
                    x_end[j] = float(period_end) - x_start[j]

        ax.axhline(y=average_percentile, color='r', linestyle='--')
        ax.plot(orig_time, all_heart_rate_4s_filtered, label='Filtered Heart Rate 4s Average', color="#009dffff")
        ax.plot(orig_time, all_heart_rate_30s_filtered, label='Filtered Heart Rate 30s Average', color='#29ff42ff')
        ax.set_xlabel('Time')
        ax.set_ylabel('HeartRate')
        ax.grid(True)
        ax.legend()
        fig.canvas.manager.window.showMaximized()
        plt.subplots_adjust(bottom=0.059, top=0.985, left=0.037, right=0.992, hspace=0.2, wspace=0.2)
        plt.show()

    # Разбитие выбранного файла на отрезки по 10 сек и сохранение в .csv
    def split_csv(self) -> None:
        data, _ = self.split_data()
        filename = f'data/payload/{self.filePath.stem}.csv'
        old_data = pd.read_csv(filename)
        data['stress'] = old_data['stress']
        data.to_csv(filename, index=False)

    # Разбитие всех файлов на отрезки по 10 сек и сохранение в .csv
    def split_all_csv(self) -> None:
        path_list = Path('data').glob('*.txt')
        for path in path_list:
            filename = f'data/payload/{path.stem}.csv'
            if os.path.isfile(filename):
                data, _ = self.split_data(path)
                old_data = pd.read_csv(filename)
                data['stress'] = old_data['stress']
                data.to_csv(filename, index=False)

    # Обработка данных для пригодности к обучению
    def split_data(self, path: str = None) -> (pd.DataFrame, pd.DataFrame):
        if path is None:
            data = self.data.drop_duplicates(subset=['timestamp:'], ignore_index=True)
        else:
            data = self.read_txt(path)
            data = data.drop_duplicates(subset=['timestamp:'], ignore_index=True)
            time_start = data.iloc[0, 0]
            data['timestamp:'] -= time_start
        data.reindex()
        data.drop('ADC', axis=1, inplace=True)

        data['HeartRate4sAverage'] = signal.savgol_filter(data['HeartRate4sAverage'], 70, 3)
        data['HeartRate30sAverage'] = signal.savgol_filter(data['HeartRate30sAverage'], 70, 3)

        percentiles = data.quantile(.5)
        average_percentile = (percentiles['HeartRate4sAverage'] + percentiles['HeartRate30sAverage']) / 2

        data['HeartRate4sAverage'] -= average_percentile
        data['HeartRate30sAverage'] -= average_percentile

        split_step = 320
        split_time = []
        split_data_list = []

        columns = ['stress', 'time']
        for j in range(split_step):
            columns.append(f'HR30sA{j}')
        for j in range(split_step):
            columns.append(f'HR4sA{j}')

        i = 0
        while (i + 1) * split_step < len(data):
            split_time.append(data['timestamp:'][split_step * i])
            row_data = [0,
                        f'{round(data['timestamp:'][split_step * i], 3)}-{round(data['timestamp:'][split_step * (i + 1)], 3)}']
            hr4a = []
            hr30a = []
            for ind in range(int(split_step * i), int(split_step * (i + 1))):
                hr4a.append(data['HeartRate4sAverage'][ind])
                hr30a.append(data['HeartRate30sAverage'][ind])
            row_data.extend(hr4a)
            row_data.extend(hr30a)
            split_data_list.append(row_data)
            i += 0.25

        split_data = pd.DataFrame(split_data_list, columns=columns)
        return split_data, split_time

    def change_selected_model(self, id: int) -> None:
        print(abs(id) - 2)
        self.selectedModelIndex = abs(id) - 2

    def change_selected_preprocessing(self, id: int) -> None:
        print(abs(id) - 2)
        self.selectedPreprocessingIndex = abs(id) - 2

    def change_selected_version_model(self, id: int) -> None:
        print(abs(id) - 2)
        self.selectedVersionModelIndex = abs(id) - 2

    # Обучение моделей
    def learn(self, optim: bool = False) -> None:
        print('Start learn')
        frames = []
        path_list = Path('data/payload').glob('**/*.csv')
        for path in path_list:
            frames.append(pd.read_csv(str(path)))
        data = pd.concat(frames, axis=0)
        data.reindex()

        X = data.iloc[:, 2:]
        y = data['stress']
        y = y.replace(2, 1)
        y = y.replace(3, 2)

        test_size = 0.3
        seed = 42

        # Настройка параметров оценивания алгоритма
        num_folds = 10
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
                        grid_search = GridSearchCV(model_dict['models']['test']['model'], params, cv=kfold, n_jobs=-1,
                                                   scoring='f1_macro', verbose=2)
                        grid_search.fit(X, y)
                        # grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        model_dict['models']['test']['model'] = copy.deepcopy(grid_search.best_estimator_)
                        model_dict['models']['prod']['model'] = copy.deepcopy(grid_search.best_estimator_)
                        model_dict['models']['test']['best_params'] = best_params
                        print(best_params)

                    cv_results = cross_val_score(model_dict['models']['test']['model'], X, y, cv=kfold, n_jobs=-1,
                                                 scoring='f1_macro')
                    model_dict['models']['test']['model'].fit(X_train, y_train)
                    y_pred = model_dict['models']['test']['model'].predict(X_test)
                    accuracy = model_dict['models']['test']['model'].score(X_test, y_test)
                    print(cv_results.mean(), cv_results)
                    model_dict['models']['test']['accuracy'] = accuracy
                    model_dict['models']['test']['f1_macro'] = cv_results.mean()
                    classificationReport = classification_report(y_test, y_pred)
                    print(classificationReport)
                    cm = confusion_matrix(y_test, y_pred)
                    model_dict['models']['test']['confusion_matrix'] = cm
                    recall = recall_score(y_test, y_pred, average='macro')
                    print(recall)
                    print(cm)
                    if column == 0:
                        radBtn = QRadioButton(model_name)
                        self.modelsGridLayout.addWidget(radBtn, row, column)
                        self.modelsBtnGroup.addButton(radBtn)

                    model_dict['models']['prod']['model'].fit(X, y)
                    y_pred = model_dict['models']['prod']['model'].predict(X)
                    accuracy_full = model_dict['models']['prod']['model'].score(X, y)
                    # print(accuracy_full)
                    # print(classification_report(y, y_pred))
                    cm_full = confusion_matrix(y, y_pred)
                    print(cm_full)
                    recall_full = recall_score(y, y_pred, average='macro')
                    f1_full = f1_score(y, y_pred, average='macro')
                    model_dict['models']['print'] = (f'{cv_results.mean():.3f} => '
                                                     f'{f1_full:.3f}')
                    accuracyLabel = QLabel(model_dict['models']['print'])
                    self.modelsGridLayout.addWidget(accuracyLabel, row, column + 1)

                    model_dict['models']['prod']['accuracy'] = accuracy_full
                    model_dict['models']['prod']['f1_macro'] = f1_full
                    model_dict['models']['prod']['confusion_matrix'] = cm_full
                except Exception as e:
                    print(e)
                row += 1
            column += 1

        filename = 'models.sav'
        pickle.dump(self.models, open(filename, 'wb'))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
