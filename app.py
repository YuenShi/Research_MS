import sys
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton, QLabel, QLineEdit, QGridLayout
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon, QFont


class MainWindow(QWidget):
    
    def __init__(self):
        super().__init__()
        # 主窗口标题
        self.title = 'DQN Trader'
        # 主窗口宽度
        self.width = 640
        # 主窗口高度
        self.height = 480
        # 初始化UI界面
        self.initUI()


    def initUI(self):

        # 设置主窗口高度和宽度
        self.resize(self.width, self.height)
        # 设置主窗口位置居中
        self.center()
        # self.setGeometry(200, 200, 500, 500)
        # 设置窗口标题
        self.setWindowTitle(self.title)
        # 设置窗口的图标
        self.setWindowIcon(QIcon('icon.png'))

        # 设置主菜单
        # mainMenu = self.menuBar() 
        # dataMenu = mainMenu.addMenu('Data')
        # trainMenu = mainMenu.addMenu('Train')
        # testMenu = mainMenu.addMenu('Test')
        # searchMenu = mainMenu.addMenu('Predict')

        # 用户输入需要获取数据的股票代码
        symbol = QLabel('Stock Symbol (example: AAPL):')
        symbolEdit = QLineEdit()
        # 用户输入数据分割比例
        split_ratio = QLabel('Split Ratio (Split Train & Test Dataset):')
        splitRatioEdit = QLineEdit()
        # 用户输入训练的轮次数
        epochs = QLabel('Train Epochs (recommend value: 20):')
        epochsEdit = QLineEdit()
        # 开始训练
        start_button = QPushButton('Start Training')
        # 终止训练
        stop_button = QPushButton('Stop Training')

        # 盒布局
        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(symbol, 1, 0)
        grid.addWidget(symbolEdit, 1, 1)

        grid.addWidget(split_ratio, 2, 0)
        grid.addWidget(splitRatioEdit, 2, 1)

        grid.addWidget(epochs, 3, 0)
        grid.addWidget(epochsEdit, 3, 1)

        grid.addWidget(start_button, 4, 0)
        grid.addWidget(stop_button, 4, 1)

        self.setLayout(grid) 





        # 显示窗口      
        self.show()

    # 窗口居中
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())