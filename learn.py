import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from decimal import Decimal

from scipy.interpolate import interp1d

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from PySide6.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog
from ui_main import Ui_MainWindow

from pathlib import Path

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

        self.learn()

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
        data_pred = self.forest.predict(split_data)
        # data_pred *= 100
        # data_pred += 60

        fig, ax = plt.subplots()
        x_start = 0
        x_end = 0
        startStress = False;
        for i in range(len(data_pred)):
            stress = data_pred[i]
            if stress == 1 and not startStress:
                x_start = split_time[i]
                startStress = True
            elif stress == 0 and startStress:
                x_end = split_time[i] - x_start
                startStress = False
                ax.add_patch(Rectangle((x_start,40),x_end,110, facecolor='r', alpha=0.2))

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

    def learn(self):
        frames = []
        pathlist = Path('data/payload').glob('**/*.csv')
        for path in pathlist:
            frames.append(pd.read_csv(str(path)))
        data = pd.concat(frames, axis=0)
        data.reindex()

        X = data.iloc[:, 2:]
        y = data['stress']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=5, stratify=y)

        self.forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        self.forest.fit(X_train, y_train)

        y_pred_forest = self.forest.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_forest)
        self.accuracyModelLb.setText(str(accuracy))
        print("Accuracy:", accuracy_score(y_test, y_pred_forest))
        print(classification_report(y_test, y_pred_forest))

        self.forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        self.forest.fit(X, y)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()