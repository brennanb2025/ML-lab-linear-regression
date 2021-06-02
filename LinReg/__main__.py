import pandas as pd
import numpy as np
from numpy import random as rng

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

from .LinearRegression import LinearRegression

if __name__ == '__main__':

    # Read in the sklearn Boston dataset
    boston_train = datasets.load_boston()
    boston_train_df = pd.DataFrame(boston_train.data, columns=boston_train.feature_names)

    #stuff I should read from boston_train dataframe
    pos_feature = 'RAD' #positively correlated
    pos_target = 'TAX'

    neg_feature = 'NOX' #negatively correlated
    neg_target = 'DIS'

    # Convert from Pandas to NumPy arrays and reshape to 1D
    x = boston_train_df[pos_feature].to_numpy()
    y = boston_train_df[pos_target].to_numpy()
    x = x.reshape(-1,1) #make numpy array n rows x 1 col
    y = y.reshape(-1,1)

    # Use sklearn to create random test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=0)
    
    # Create a Linear Regression model and fit a line
    regressor = LinearRegression(0.001, 0.006, pos_feature, pos_target, x_train, y_train)
    #0.01 takes __ epochs
    #0.001 takes __ epochs
    #0.0001 takes __ epochs
    #0.00001 takes __ epochs
    regressor.animate()

    # Plot the training data and the prediction line
    """plt.scatter(x_train,y_train, color='red')
    plt.plot(x_train, regressor.predict(x_train), color='blue')
    plt.show()"""
    
