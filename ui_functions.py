from main import *
import numpy as np

# Global variables
GLOBAL_STATE = 0
GLOBAL_TITLE_BAR = True


class UIFunctions(MainWindow):
    def get_image_file(self):
        # Get the path to an image
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r"coins.jpg", "Image files (*.jpg *.jpeg *.png *.gif)")
        # Add the image to the label
        self.ui.label_image.setPixmap(QPixmap(self.image_path))
        # Resize the image
        self.ui.label_image.setScaledContents(True)

    def extract_coins(self):
        if self.image_path:
            coins_ext = CoinExtraction(self.image_path)
            circles = coins_ext.hough_transform()
            circles_img, self.cropped_coins = coins_ext.crop_hough(circles)
            # Convert numpy array to Pixmap
            circles_img = QtGui.QImage(circles_img.data, circles_img.shape[1], circles_img.shape[0],
                                      QtGui.QImage.Format_RGB888).rgbSwapped()
            # Add the image to the label
            self.ui.label_image.setPixmap(QtGui.QPixmap.fromImage(circles_img))
            self.ui.label_image.setScaledContents(True)

    def recognize_coins(self):
        if self.image_path:
            coins_rec = CoinRecognition(self.model_path)
            self.values, self.coins_sum = coins_rec.recognize(self.cropped_coins)
            self.ui.label_sum.setText("The sum is {} zł".format(round(self.coins_sum, 2)))

    def populate_table(self):
        # Clear previous records
        self.ui.tableWidget.setRowCount(0)
        if self.coins_sum > 0.0:
            for value, cropped_coin in zip(self.values, self.cropped_coins):
                # Creating a new row
                row_position = self.ui.tableWidget.rowCount()
                self.ui.tableWidget.insertRow(row_position)
                # Populate the table with coins
                self.ui.tableWidget.setCellWidget(row_position, 0, UIFunctions.get_image_widget(self, cropped_coin))
                # Check if the value is numeric
                if isinstance(self.values, float):
                    self.ui.tableWidget.setItem(row_position, 1,
                                                QtWidgets.QTableWidgetItem(str(value) + " zł"))
                else:
                    self.ui.tableWidget.setItem(row_position, 1, QtWidgets.QTableWidgetItem(str(value)))
                self.ui.tableWidget.resizeRowsToContents()

    def get_image_widget(self, coin):
        temp_label = QtWidgets.QLabel(self.ui.centralwidget)
        # Transparent label
        temp_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        # Make the array uint8 type
        coin = coin.astype(np.uint8)
        # Convert numpy array to Pixmap
        height, width, channel = coin.shape
        bytesPerLine = 3 * width
        coin_img = QtGui.QImage(coin.data, width, height, bytesPerLine,
                                   QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(coin_img).scaled(90, 90, QtCore.Qt.KeepAspectRatio)
        temp_label.setPixmap(pixmap)
        temp_label.setAlignment(QtCore.Qt.AlignCenter)
        return temp_label

    # Maximize, restore the window
    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE
        if status == 0:
            self.showMaximized()
            GLOBAL_STATE = 1
            self.ui.horizontalLayout.setContentsMargins(0, 0, 0, 0)
            self.ui.frame_top_btns.setStyleSheet("background-color: rgb(27, 29, 35)")
            self.ui.frame_size_grip.hide()
        else:
            GLOBAL_STATE = 0
            self.showNormal()
            self.resize(self.width()+1, self.height()+1)
            self.ui.horizontalLayout.setContentsMargins(10, 10, 10, 10)
            self.ui.frame_top_btns.setStyleSheet("background-color: rgba(27, 29, 35, 200)")
            self.ui.frame_size_grip.show()

    # Return window status
    def return_status(self):
        return GLOBAL_STATE

    # Set title bar
    def set_title_bar(status):
        global GLOBAL_TITLE_BAR
        GLOBAL_TITLE_BAR = status

    # UI definitions
    def ui_definitions(self):
        def dbl_click_maximize_restore(event):
            # Change status when double clicked
            if event.type() == QtCore.QEvent.MouseButtonDblClick:
                QtCore.QTimer.singleShot(250, lambda: UIFunctions.maximize_restore(self))

        # Remove standard title bar
        if GLOBAL_TITLE_BAR:
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.ui.frame_label_top_btns.mouseDoubleClickEvent = dbl_click_maximize_restore
        else:
            self.ui.horizontalLayout.setContentsMargins(0, 0, 0, 0)
            self.ui.frame_label_top_btns.setContentsMargins(8, 0, 0, 5)
            self.ui.frame_label_top_btns.setMinimumHeight(42)
            self.ui.frame_icon_top_bar.hide()
            self.ui.frame_btns_right.hide()
            self.ui.frame_size_grip.hide()

        # Drop shadow
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(17)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.ui.frame_main.setGraphicsEffect(self.shadow)

        # Resize the window
        self.sizegrip = QSizeGrip(self.ui.frame_size_grip)
        self.sizegrip.setStyleSheet("width: 20px; height: 20px; margin 0px; padding: 0px;")

        # Resize table headers
        self.ui.tableWidget.horizontalHeader().setStretchLastSection(True)

        # Minimize
        self.ui.btn_minimize.clicked.connect(lambda: self.showMinimized())

        # Maximize/restore
        self.ui.btn_maximize_restore.clicked.connect(lambda: UIFunctions.maximize_restore(self))

        # Close application
        self.ui.btn_close.clicked.connect(lambda: self.close())

        # Upload image
        self.ui.btn_upload.clicked.connect(lambda: UIFunctions.get_image_file(self))

        # Show the image with circles marked on the input image
        self.ui.btn_run.clicked.connect(lambda: UIFunctions.extract_coins(self))

        # Sum up the coins
        self.ui.btn_run.clicked.connect(lambda: UIFunctions.recognize_coins(self))

        # Put values of each coin in the table
        self.ui.btn_run.clicked.connect(lambda: UIFunctions.populate_table(self))
