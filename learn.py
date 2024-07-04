# Подкласс QMainWindow для настройки главного окна приложения
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        self.graphsBtn.clicked.connect(self.showGraphs)

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
            self.graphsBtn.setEnabled(True)

    def readCsv(self):
        self.data = pd.read_csv(self.filePath)

    def readTxt(self):
        with open(self.filePath, 'r') as f:
            rows = f.read().split('\n')
            self.data = pd.DataFrame(columns=rows[0].split())
            rows = rows[1:]
            for row in rows:
                if row:
                    data_row = row.split()
                    data_row[0] = float(data_row[0][:-1].replace(',', '.'))
                    self.data.loc[len(self.data)] = data_row
            self.data.to_csv(self.filePath.stem + '.csv', index=False)
    def showGraphs(self):
        print(self.data)
        time_start = self.data.iloc[0, 0]
        print(time_start)
        self.data['timestamp:'] -= time_start

        all_time = np.array([])
        time_0 = self.data['timestamp:'][0]

        i = 1
        for index in range(len(self.data['timestamp:'])):
            time = self.data['timestamp:'][index]
            if time != time_0 or index == len(self.data['timestamp:']) - 1:
                all_time = np.hstack([all_time, np.linspace(time_0, time, i)])
                time_0 = time
                i = 1
            else:
                i += 1

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(all_time, self.data['HeartRate4sAverage'].to_numpy(dtype=int), label='Heart Rate 4s Average')
        ax[0].plot(all_time, self.data['HeartRate30sAverage'].to_numpy(dtype=int), label='Heart Rate 30s Average')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('HeartRate')
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(all_time, self.data['ADC'].to_numpy(dtype=int))
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('ADC')
        ax[1].grid(True)

        plt.show()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()