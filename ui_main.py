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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLayout,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QSpacerItem, QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(516, 235)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
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

        self.learnBtn = QPushButton(self.centralwidget)
        self.learnBtn.setObjectName(u"learnBtn")

        self.verticalLayout.addWidget(self.learnBtn)

        self.splitBtn = QPushButton(self.centralwidget)
        self.splitBtn.setObjectName(u"splitBtn")
        self.splitBtn.setEnabled(False)

        self.verticalLayout.addWidget(self.splitBtn)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_3.addWidget(self.label_2)

        self.accuracyModelLb = QLabel(self.centralwidget)
        self.accuracyModelLb.setObjectName(u"accuracyModelLb")

        self.horizontalLayout_3.addWidget(self.accuracyModelLb)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 516, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u0424\u0430\u0439\u043b:", None))
        self.fileName.setText(QCoreApplication.translate("MainWindow", u"\u041d\u0435 \u0432\u044b\u0431\u0440\u0430\u043d", None))
        self.selectFileBtn.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0431\u0440\u0430\u0442\u044c", None))
        self.heartRateBtn.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a \u0427\u0421\u0421", None))
        self.ecgCurveBtn.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a \u043a\u0440\u0438\u0432\u043e\u0439", None))
        self.learnBtn.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u0443\u0447\u0438\u0442\u044c", None))
        self.splitBtn.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0438\u0442\u044c", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u0422\u043e\u0447\u043d\u043e\u0441\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u0438:", None))
        self.accuracyModelLb.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
    # retranslateUi

