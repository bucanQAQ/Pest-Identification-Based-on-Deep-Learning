# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Image_classify.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(544, 616)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        Dialog.setFont(font)
        self.selectImage_Btn = QtWidgets.QPushButton(Dialog)
        self.selectImage_Btn.setGeometry(QtCore.QRect(90, 350, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.selectImage_Btn.setFont(font)
        self.selectImage_Btn.setObjectName("selectImage_Btn")
        self.run_Btn = QtWidgets.QPushButton(Dialog)
        self.run_Btn.setGeometry(QtCore.QRect(360, 350, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.run_Btn.setFont(font)
        self.run_Btn.setObjectName("run_Btn")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(90, 260, 91, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(90, 300, 121, 31))
        self.label_2.setObjectName("label_2")
        self.display_result = QtWidgets.QLabel(Dialog)
        self.display_result.setGeometry(QtCore.QRect(190, 270, 221, 16))
        self.display_result.setText("")
        self.display_result.setObjectName("display_result")
        self.disply_acc = QtWidgets.QLabel(Dialog)
        self.disply_acc.setGeometry(QtCore.QRect(190, 310, 191, 16))
        self.disply_acc.setText("")
        self.disply_acc.setObjectName("disply_acc")
        self.gridLayoutWidget = QtWidgets.QWidget(Dialog)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(89, 10, 391, 251))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_image = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_image.setText("")
        self.label_image.setObjectName("label_image")
        self.gridLayout.addWidget(self.label_image, 0, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(360, 440, 131, 51))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox_2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_2.setGeometry(QtCore.QRect(90, 440, 131, 51))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")

        self.retranslateUi(Dialog)
        self.selectImage_Btn.clicked.connect(Dialog.accept)
        self.run_Btn.clicked.connect(Dialog.accept)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.selectImage_Btn.setText(_translate("Dialog", "Select image"))
        self.run_Btn.setText(_translate("Dialog", "Run"))
        self.label.setText(_translate("Dialog", "Result:"))
        self.label_2.setText(_translate("Dialog", "Accuracy:"))
        self.comboBox.setItemText(0, _translate("Dialog", "Resnet18"))
        self.comboBox.setItemText(1, _translate("Dialog", "Resnet34"))
        self.comboBox.setItemText(2, _translate("Dialog", "Resnet50"))
        self.comboBox_2.setItemText(0, _translate("Dialog", "No remove background"))
        self.comboBox_2.setItemText(1, _translate("Dialog", "Remove background"))
