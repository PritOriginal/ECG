# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QLayout, QMainWindow, QMenuBar, QPushButton,
    QRadioButton, QSizePolicy, QSpacerItem, QStatusBar,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(1134, 267)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet(u"QMainWindow,\n"
"QWidget {\n"
"	background-color: rgb(41, 41, 41);\n"
"}\n"
"QLabel,\n"
"QRadioButton,\n"
"QPushButton {\n"
"	color: #fff\n"
"}\n"
"QPushButton {\n"
"	background-color: rgb(87, 87, 87);\n"
"}\n"
"QPushButton:disabled {\n"
"	\n"
"	background-color: rgb(62, 62, 62);\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy1)
        self.label.setMaximumSize(QSize(40, 16777215))
        self.label.setBaseSize(QSize(0, 0))

        self.horizontalLayout.addWidget(self.label)

        self.fileName = QLabel(self.centralwidget)
        self.fileName.setObjectName(u"fileName")

        self.horizontalLayout.addWidget(self.fileName)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.selectFileBtn = QPushButton(self.centralwidget)
        self.selectFileBtn.setObjectName(u"selectFileBtn")

        self.verticalLayout.addWidget(self.selectFileBtn)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.heartRateBtn = QPushButton(self.centralwidget)
        self.heartRateBtn.setObjectName(u"heartRateBtn")
        self.heartRateBtn.setEnabled(False)

        self.horizontalLayout_2.addWidget(self.heartRateBtn)

        self.ecgCurveBtn = QPushButton(self.centralwidget)
        self.ecgCurveBtn.setObjectName(u"ecgCurveBtn")
        self.ecgCurveBtn.setEnabled(False)

        self.horizontalLayout_2.addWidget(self.ecgCurveBtn)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.learnBtn = QPushButton(self.centralwidget)
        self.learnBtn.setObjectName(u"learnBtn")

        self.horizontalLayout_6.addWidget(self.learnBtn)

        self.splitBtn = QPushButton(self.centralwidget)
        self.splitBtn.setObjectName(u"splitBtn")
        self.splitBtn.setEnabled(False)

        self.horizontalLayout_6.addWidget(self.splitBtn)

        self.splitAllBtn = QPushButton(self.centralwidget)
        self.splitAllBtn.setObjectName(u"splitAllBtn")
        self.splitAllBtn.setEnabled(True)

        self.horizontalLayout_6.addWidget(self.splitAllBtn)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_5.addWidget(self.label_9)

        self.radBtnVersionModelTest = QRadioButton(self.centralwidget)
        self.radBtnVersionModelTest.setObjectName(u"radBtnVersionModelTest")

        self.horizontalLayout_5.addWidget(self.radBtnVersionModelTest)

        self.radBtnVersionModelProd = QRadioButton(self.centralwidget)
        self.radBtnVersionModelProd.setObjectName(u"radBtnVersionModelProd")
        self.radBtnVersionModelProd.setChecked(True)

        self.horizontalLayout_5.addWidget(self.radBtnVersionModelProd)

        self.radBtnVersionModelAll = QRadioButton(self.centralwidget)
        self.radBtnVersionModelAll.setObjectName(u"radBtnVersionModelAll")

        self.horizontalLayout_5.addWidget(self.radBtnVersionModelAll)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_4.addWidget(self.label_3)

        self.prepRadBtnNone = QRadioButton(self.centralwidget)
        self.prepRadBtnNone.setObjectName(u"prepRadBtnNone")
        self.prepRadBtnNone.setChecked(True)

        self.horizontalLayout_4.addWidget(self.prepRadBtnNone)

        self.prepRadBtnStandardScaler = QRadioButton(self.centralwidget)
        self.prepRadBtnStandardScaler.setObjectName(u"prepRadBtnStandardScaler")

        self.horizontalLayout_4.addWidget(self.prepRadBtnStandardScaler)

        self.prepRadBtnRobustScaler = QRadioButton(self.centralwidget)
        self.prepRadBtnRobustScaler.setObjectName(u"prepRadBtnRobustScaler")

        self.horizontalLayout_4.addWidget(self.prepRadBtnRobustScaler)

        self.prepRadBtnMinMaxScaler = QRadioButton(self.centralwidget)
        self.prepRadBtnMinMaxScaler.setObjectName(u"prepRadBtnMinMaxScaler")

        self.horizontalLayout_4.addWidget(self.prepRadBtnMinMaxScaler)

        self.prepRadBtnNormalizer = QRadioButton(self.centralwidget)
        self.prepRadBtnNormalizer.setObjectName(u"prepRadBtnNormalizer")

        self.horizontalLayout_4.addWidget(self.prepRadBtnNormalizer)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_3.addWidget(self.label_2)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)

        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_3.addWidget(self.label_5)

        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_3.addWidget(self.label_6)

        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_3.addWidget(self.label_7)

        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_3.addWidget(self.label_8)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.modelsGridLayout = QGridLayout()
        self.modelsGridLayout.setObjectName(u"modelsGridLayout")

        self.verticalLayout.addLayout(self.modelsGridLayout)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1134, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u042d\u041a\u0413", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u0424\u0430\u0439\u043b:", None))
        self.fileName.setText(QCoreApplication.translate("MainWindow", u"\u041d\u0435 \u0432\u044b\u0431\u0440\u0430\u043d", None))
        self.selectFileBtn.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0431\u0440\u0430\u0442\u044c", None))
        self.heartRateBtn.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a \u0427\u0421\u0421", None))
        self.ecgCurveBtn.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a \u043a\u0440\u0438\u0432\u043e\u0439", None))
        self.learnBtn.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u0443\u0447\u0438\u0442\u044c", None))
        self.splitBtn.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0438\u0442\u044c", None))
        self.splitAllBtn.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0438\u0442\u044c \u0432\u0441\u0451", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"\u0412\u0435\u0440\u0441\u0438\u044f \u043c\u043e\u0434\u0435\u043b\u0438:", None))
        self.radBtnVersionModelTest.setText(QCoreApplication.translate("MainWindow", u"Test", None))
        self.radBtnVersionModelProd.setText(QCoreApplication.translate("MainWindow", u"Prod", None))
        self.radBtnVersionModelAll.setText(QCoreApplication.translate("MainWindow", u"\u0412\u0441\u0435", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430:", None))
        self.prepRadBtnNone.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.prepRadBtnStandardScaler.setText(QCoreApplication.translate("MainWindow", u"StandardScaler", None))
        self.prepRadBtnRobustScaler.setText(QCoreApplication.translate("MainWindow", u"RobustScaler", None))
        self.prepRadBtnMinMaxScaler.setText(QCoreApplication.translate("MainWindow", u"MinMaxScaler", None))
        self.prepRadBtnNormalizer.setText(QCoreApplication.translate("MainWindow", u"Normalizer", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u041c\u043e\u0434\u0435\u043b\u0438:", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"test => prod (f1_macro)", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"test => prod (f1_macro)", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"test => prod (f1_macro)", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"test => prod (f1_macro)", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"test => prod (f1_macro)", None))
    # retranslateUi

