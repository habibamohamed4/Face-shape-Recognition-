





from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
import sys
from PyQt5.QtWidgets import QFileDialog , QLabel
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
from matplotlib import pyplot as plt
from math import sqrt
from PIL import Image
import pyqtgraph.exporters
import numpy as np
import cv2




matplotlib.use('QT5Agg')

class MatplotlibCanvas(FigureCanvasQTAgg):
	def __init__(self,parent=None, dpi = 120):
		fig = Figure(dpi = dpi)
		self.axes = fig.add_subplot(111)
		super(MatplotlibCanvas,self).__init__(fig)
		fig.tight_layout()


class mainApp(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(mainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionExit.triggered.connect(lambda: self.exit() )
        self.ui.actionOpen.triggered.connect(lambda: self.browseAnImg())
        self.ui.actionModel_1.triggered.connect(lambda: self.Model1Run())
        self.ui.actionModel_2.triggered.connect(lambda: self.Model2Run())
        self.ui.actionLandMarks_Model.triggered.connect(lambda: self.LandMarksModelRun())
      
    

    #Declaration of any global variables 
        self.logHistory=[] #A list created in order to log every action on the gui from its start till its closed by the user 
        
    #Start of functions 

    #saving function (to be used later is we want to save the resulting images or sth)
    def saveLocation(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
						"PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        return filePath

    #Save image function
    def saveImag(self, whichScreenText):
        self.logging("The save image function was called")
        path= self.saveLocation()
 

    #The Exit function 
    def exit(self):
        self.logging("Exit function was called")
        sys.exit()

    #Model I run
    def Model1Run(self):
        ###### connect code here 
        print("model 1 running")
        pass

    #Model II run
    def Model2Run(self):
        ###### connect code here 
        print("model 2 running")
        pass

    #Land Marks Model Run
    def LandMarksModelRun(self):
        ###connect code here
        print("Land marks running")
        pass


    #The logging function 
    def logging(self, text):
        f=open("logHistory.txt","w+")
        self.logHistory.append(text)
        for i in self.logHistory:
            f.write("=> %s\r\n" %(i))
        f.close()

    def browseAnImg(self):
        self.logging("browseAnImg function was called")
        image=QFileDialog.getOpenFileName()
        self.logging("Image path was chosen from the dialog box")
        self.imagePath = image[0]
        print(self.imagePath)
        self.logging("image path is set to "+self.imagePath)






if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())