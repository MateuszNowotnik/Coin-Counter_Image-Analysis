from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import QSize, Qt, QEvent
from PySide2.QtGui import QColor, QFont, QIcon, QPixmap, QImage
from PySide2.QtWidgets import *

import sys, os
from ui_main import Ui_MainWindow
from ui_functions import *
from coin_extraction import CoinExtraction
from coin_recognition_test import CoinRecognition


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.model_path = "coins.model"
        self.image_path = ""
        self.coins_sum = 0.0
        self.values = []
        self.cropped_coins = []

        # Remove the standard title bar
        UIFunctions.set_title_bar(self)

        # Set window title
        self.ui.label_title_bar_top.setText('Coin Counter - Mateusz Nowotnik')

        # Window default size
        start_size = QSize(1200, 720)
        self.resize(start_size)
        self.setMinimumSize(start_size)

        # Move, maximize, restore the window
        def move_window(event):
            # Change window to normal if it's maximized
            if UIFunctions.return_status(self) == 1:
                UIFunctions.maximize_restore(self)

            # Move window
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        # Widget to move
        self.ui.frame_label_top_btns.mouseMoveEvent = move_window

        # Load definitions
        UIFunctions.ui_definitions(self)

        # QTableWidget parameters
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        # Show main window
        self.show()

    # Initialize mouse drag position
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
