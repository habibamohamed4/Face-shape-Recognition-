

from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
import sys
from PyQt5.QtWidgets import QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
import pyqtgraph
from pyqtgraph import *
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
from GUI import Ui_MainWindow
import pyqtgraph.exporters
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from math import sqrt
from PIL import Image
import pyqtgraph.exporters
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import tensorflow as tf
matplotlib.use('QT5Agg')


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        fig.tight_layout()


class mainApp(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(mainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionExit.triggered.connect(lambda: self.exit())
        self.ui.actionOpen.triggered.connect(lambda: self.browseAnImg())
        self.ui.actionModel_1.triggered.connect(lambda: self.model1())
        self.ui.actionModel_2.triggered.connect(lambda: self.Model2())
        self.ui.actionLandMarks_Model.triggered.connect(
            lambda: self.LandMarksModelRun())

    # Declaration of any global variables
        # A list created in order to log every action on the gui from its start till its closed by the user
        self.logHistory = []

    # Start of functions

    # saving function (to be used later is we want to save the resulting images or sth)
    def saveLocation(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        return filePath

    # Save image function
    def saveImag(self, whichScreenText):
        self.logging("The save image function was called")
        path = self.saveLocation()

    # The Exit function

    def exit(self):
        self.logging("Exit function was called")
        sys.exit()

    def model1(self):
        new_model = load_model('face-shape-recognizer (2).h5')
       # new_model.summary()
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            brightness_range=(0.8, 1.2),
            horizontal_flip=True,
        )
        test_set = test_datagen.flow_from_directory(
            'C:/Users/seif/Downloads/archive/dataset/test',
            target_size=(250, 190),
            batch_size=64,
            color_mode='grayscale',
            shuffle=True,
            class_mode='categorical'
        )
        #scoreSeg = new_model.evaluate(test_set)
        #print("Accuracy = ",scoreSeg[1])
        X_test, y_test = next(test_set)

        labels = list(test_set.class_indices)
        path1 = self.browseAnImg()
        # print(path1)
       # img2="{img}"
        path = os.path.normpath(path1).split(os.path.sep)
        # print(path)
        img = tf.keras.utils.load_img(
            path1,
            #  grayscale=True,
            color_mode='grayscale',
            target_size=(250, 190),
            interpolation='nearest',
            keep_aspect_ratio=False
        )
        img1 = image.img_to_array(img)
        img1 = img1/255.
        img1 = np.expand_dims(img1, axis=0)
        images = np.vstack([img1])
# print(images)
        x = new_model.predict(img1, verbose=0)
        # print(x)
        y_pred = np.argmax(new_model.predict(img1, verbose=0), axis=1)[0]
        # print(y_pred)

        # print(img1.shape)
        plt.imshow(img)
        # plt.close()
        plt.axis('off')
        plt.title(f" Y pred  ({labels[y_pred]})")
        # plt.show()
        plt.savefig('gray.jpg', bbox_inches='tight')
        # plt.show()

        pixmap = QPixmap('gray.jpg').scaled(450, 550)
        self.ui.tab1Model1FaceShapeImageLabel.setPixmap(pixmap)
        pass
    # Model I run

    def Model2(self):
        model2 = load_model('face11-shape-recognizer.h5')
        class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        class_names_label = {class_name: i for i,
                             class_name in enumerate(class_names)}
        IMAGE_SIZE = (150, 150)
        path2 = self.browseAnImg()
        # print(path2)
        image2 = cv2.imread(path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image3 = cv2.resize(image2, IMAGE_SIZE)
        image3 = np.reshape(image3, [1, 150, 150, 3])

        image3 = image3 / 255.0
        # print(image3.shape)
        predictions = model2.predict(image3)     # Vector of probabilities
        # We take the highest probability
        pred_labels = np.argmax(predictions, axis=1)[0]
# print(predictions)
        print(pred_labels)
        plt.imshow(image2)
        plt.axis('off')
        plt.title(f" Y pred ({class_names[pred_labels]})")
        plt.savefig('img2.jpg', bbox_inches='tight')
        # plt.show()

        pixmap = QPixmap('img2.jpg').scaled(450, 550)
        self.ui.tab1Model2FaceShapeImageLabel.setPixmap(pixmap)

        # pass

    # # Model II run
    # def Model2Run(self):
    #     # connect code here
    #     print("model 2 running")
    #     pass

    # Land Marks Model Run
    def LandMarksModelRun(self):
        model3 = load_model('landmark.h5')
        IMAGE_SIZE = (96, 96)
        path3 = self.browseAnImg()
        print(path3)
        image4 = cv2.imread(path3)
        image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
        image5 = cv2.resize(image4, IMAGE_SIZE)
        image5 = image5.astype('float32')
        image5 = np.reshape(image5, [1, 96, 96, 1])
        image5 = image5 / 255.0
        plt.imshow(image5[0, :, :, 0], cmap='gray')
        predictions = model3.predict(image5)
        p = predictions[:]
        plt.scatter(p[:, 0::2], p[:, 1::2], c='r')
        plt.savefig('img4.jpg')
        pixmap1 = QPixmap('img4.jpg').scaled(450, 550)
        self.ui.tab2LandMarksResultLabel.setPixmap(pixmap1)
        # pass

    # The logging function

    def logging(self, text):
        f = open("logHistory.txt", "w+")
        self.logHistory.append(text)
        for i in self.logHistory:
            f.write("=> %s\r\n" % (i))
        f.close()

    def browseAnImg(self):
        self.logging("browseAnImg function was called")
        image = QFileDialog.getOpenFileName()
        self.logging("Image path was chosen from the dialog box")
        self.imagePath = image[0]
        print(self.imagePath)
        self.logging("image path is set to "+self.imagePath)
        return self.imagePath


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())
