import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as dates
import matplotlib.dates as mdates  # 處理日期
from Base.Strategy_base import Np_Order_Info
from matplotlib import cbook  # 引入cbook以做為距離計算

# 初始化
plt.style.use('seaborn-deep')

class Picture_maker():
    def __init__(self, Order_info: Np_Order_Info) -> None:
        self.pf = Order_info
        self.annotations = []
        
        print("參數檢查:")
        print(self.pf.order)
        print(self.pf.ClosedPostionprofit_array)
        print('*'*120)
        self.get_Mdd_UI(self.pf.order.index.to_numpy(
        ), self.pf.ClosedPostionprofit_array, self.pf.drawdown, self.pf.drawdown_per)
        

    # 定義事件響應函數
    def on_motion(self,event):
        if event.inaxes == self.ax2:
            # remove old annotations first
            for annotation in self.annotations:
                b = annotation.remove()
            self.annotations = []
            plt.draw()
                        
            for bar in self.bar_plot2:
                if bar.contains(event)[0]:
                    height = bar.get_height()
                    print(height)
                    annotation = self.ax2.annotate(f'Value: {height}',
                                            xy=(bar.get_x() + bar.get_width() / 2, height),
                                            xytext=(0, 3),  # 3 points vertical offset
                                            textcoords="offset points",
                                            ha='center', va='bottom')
                    self.annotations.append(annotation)
                    plt.draw()
                    break  # Found the bar, no need to keep checking
    
        elif event.inaxes == self.ax1:
            # 找到最接近滑鼠位置的數據點
            xdata = self.line_plot1.get_xdata()
            ydata = self.line_plot1.get_ydata()
            ind = np.argmin(np.abs(xdata - event.xdata))
            info = f'x = {xdata[ind]:.2f}, y = {ydata[ind]:.2f}'

            # 更新信息框的內容
            self.info_box.set_text(info)
            self.ax1.figure.canvas.draw_idle()
            
    def get_Mdd_UI(self, x1_data: np.ndarray, y1_data, dd_data: np.ndarray, dd_perdata: np.ndarray):
        """_summary_

        Args:
            x1_data (np.ndarray): [.....'2023-07-11 22:30:00' '2023-07-13 21:45:00' '2023-07-22 03:30:00'
                                '2023-08-12 08:00:00' '2023-08-12 13:45:00' '2023-08-14 04:00:00'
                                '2023-08-14 06:45:00' '2023-08-15 22:45:00' '2023-08-16 01:00:00'
                                '2023-08-27 22:15:00' '2023-08-28 13:45:00' '2023-08-28 20:30:00'
                                '2023-08-31 20:15:00']
            y1_data (_type_):  [.......135687.95089969 135354.97448902 135354.97448902 134834.18362074
                                134834.18362074 132094.04379792 132094.04379792 132094.04379792
                                131279.76868906 130849.70183877 130849.70183877 129237.62458353
                                129237.62458353 129237.62458353 128287.40137183 127181.94742665
                                127181.94742665 127181.94742665 126831.13012369 125575.67980744
                                125575.67980744 125575.67980744 125257.90992691 125257.90992691
                                126910.10193655]
            dd_data (np.ndarray): _description_
            dd_perdata (np.ndarray): _description_
        """
        if x1_data.dtype == 'datetime64[ns]':
            x1_data = [str(date).split('T')[0] for date in x1_data]
        else:            
            x1_data = np.array([date.split(' ')[0] for date in x1_data])

        fig, (self.ax1, self.ax2, ax3) = plt.subplots(
            3, 1, figsize=(80, 10), height_ratios=(2, 1, 1),dpi=80)

        # X 軸相關設置
        out_list = []
        out_lables = []
        i = 0
        for _i in range(8):
            out_list.append(i)
            out_lables.append(x1_data[i])
            i = i + divmod(x1_data.shape[0], 8)[0]

        # 坐標軸的位置
        self.ax1.set_xticks(out_list)
        # 坐標軸的內容
        self.ax1.set_xticklabels(out_lables)
        
        self.line_plot1, = self.ax1.plot(y1_data)

        # 創新高時畫上綠點
        for i in range(dd_perdata.shape[0]):
            if dd_perdata[i] == 0:
                self.ax1.plot(i, y1_data[i], marker="8", color='#008000')

        self.ax1.grid(True)
        self.ax1.set_title("ClosedPostionprofit")

        # 添加一個用於顯示信息的文本框
        self.info_box = self.ax1.text(0.05, 0.95, '', transform=self.ax1.transAxes, fontsize=12,
                            verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
        # 坐標軸的位置
        self.ax2.set_xticks(out_list)
        # 坐標軸的內容
        self.ax2.set_xticklabels(out_lables)
        self.bar_plot2 = self.ax2.bar(range(len(x1_data)), dd_data, width=1, color='#F08080')
        self.ax2.grid(True)
        self.ax2.set_title("MDD")

        # 坐標軸的位置
        ax3.set_xticks(out_list)
        # 坐標軸的內容
        ax3.set_xticklabels(out_lables)
        ax3.bar(range(len(x1_data)), dd_perdata, width=1, color='#22C32E')
        ax3.grid(True)
        ax3.set_title("MDDPer")
        
        # 連接事件
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        plt.show()
