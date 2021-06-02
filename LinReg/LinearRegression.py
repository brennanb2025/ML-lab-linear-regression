# For drawing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
#from scipy import stats
#from sklearn.metrics import r2_score
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate, tolerance, x_label, y_label, x_train, y_train):
        self.learning_rate = learning_rate
        self.w0 = 0 #randomize
        self.w1 = 20 #randomize
        self.tolerance = tolerance
        self.x = x_train
        self.y = y_train
        self.x_total = np.size(x_train)
        self.fig, ax = plt.subplots()
        plt.title("Graph of points")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax.set_xlim(np.amin(x_train), np.amax(x_train))
        ax.set_ylim(np.amin(y_train), np.amax(y_train))
        self.line, = ax.plot(0, 0)

    #def fit(self, x, y):
    def fit(self, i):

        #for i in range(self.num_epochs):
        fit_line = np.add(np.multiply(self.w1,self.x),self.w0)
        self.line.set_data(self.x, fit_line)
        #self.err(fit_line, x, y)
        
        error_w0 = self.learning_rate*self.errorW0(fit_line)
        error_w1 = self.learning_rate*self.errorW1(fit_line)

        #tolerance should be based on sum of difference in steps of all variables
        #print(abs(error_w0+error_w1))
        if self.tolerance > abs(error_w0+error_w1):
            self.anim.event_source.stop()
            print(w1,"x + ",w2)
            return

        self.w1-=error_w1
        self.w0-=error_w0
        print(error_w1, error_w0)
        plt.scatter(self.x, self.y, marker='o', color = 'red')
        """self.plotRegressionLine(x, y, fit_line)

    def plotRegressionLine(self, x, y, fit_line):
        # plotting the regression line
        fig.plot(x, fit_line, color = "g")"""

    """def estimateLine(self, x,y):
        # number of observations/points
        n = np.size(x)

        m_x = np.mean(x)
        m_y = np.mean(y)
        
        #cross deviation and deviation about x
        SS_xy = np.sum(y*x) - n*m_y*m_x
        SS_xx = np.sum(x*x) - n*m_x*m_x

        # calculating regression coefficients
        slope = SS_xy / SS_xx
        y_int = m_y - slope*m_x

        return (y_int, slope)"""
    
    """def error(self, line, x, y):
        diffs = line-y
        diff_total = 0
        for d in diffs:
            diff_total+=Math.pow(d, 2)
        diff_total/=np.size(x) #divide by number of observations/points
        print(diff_total)"""
    
    def errorW0(self, fit_line):
        diff_total = np.sum(np.subtract(self.y,fit_line))
        diff_total = np.divide(diff_total,self.x_total) #divide by number of observations/points
        diff_total = np.multiply(diff_total,-2)
        # and multiply by 2 for the partial derivative squared, by -1 to make the change go towards the minimum instead of the maximum
        return diff_total
    
    def errorW1(self, fit_line):
        diff_total = np.sum(np.multiply(self.x,np.subtract(self.y,fit_line)))
        diff_total = np.divide(diff_total,self.x_total) #divide by number of observations/points
        diff_total = np.multiply(diff_total,-2)
        # and multiply by 2 for the partial derivative squared, by -1 to make the change go towards the minimum instead of the maximum
        return diff_total

    """def init_func(self):
        self.line.set_data([], []) 
        return line, """

    def animate(self):
        # calling the animation function      
        self.anim = animation.FuncAnimation(self.fig, func=self.fit, interval = 17)
        plt.show()