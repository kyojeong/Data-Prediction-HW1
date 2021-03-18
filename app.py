# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 02:17:45 2020

@author: user
"""
import argparse
import pandas as pd
#from keras.models import Model
import numpy as np
from matplotlib import pyplot
import csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed. 

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    
    df_training = pd.read_csv(args.training,squeeze=True) 
    #writer=csv.writer(args.output)
    X=df_training['data'].values
    size=int(len(X)*0.95)
    train2,test=X[0:size],X[size:len(X)]
    train=[x for x in train2]
    predictions=list()
    #df_training['data'].plot()
    #pyplot.show()
    for t in range(len(test)):
    	model = ARIMA(train, order=(5,1,0))
    	model_fit = model.fit()
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	obs = test[t]
    	train.append(obs)
    	print('predicted=%f, expected=%f' % (yhat, obs))
    
    
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    
    date=np.zeros(7)
    data=np.zeros(7)
    k=int(20210322)
    for t in range(7):
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        k=k+1
        train.append(yhat)
        print('%d, %d' % (k, yhat))
        #writer.writerrow([k,yhat])
        date[t]=int(k)
        data[t]=int(yhat)
        
    frame={
        'date':date,
        'operation_reserve(MW)':data
    }
    
    output=pd.DataFrame(frame)
    output.to_csv(args.output)