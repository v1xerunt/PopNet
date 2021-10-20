import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, ceil
import math
import time

def geo_distance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    return distance

def MAPE(y_true, y_pred):
    true_idx = (y_true > 1e-5)
    y_true = y_true[true_idx]
    y_pred = y_pred[true_idx]
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def eval(model, x_r, x_u, y, static, hidden1, hidden2, interval, normalizer, norm='True'):
    '''
    Evaluation function
    '''
    
    model.eval()
    y_pred = []

    y_pred, _, hidden1, hidden2, kl = model(x_r, x_u, static, hidden1, hidden2, interval)
    y_pred = y_pred.cpu().detach().numpy()
    
    actual_pred = np.array(y_pred.squeeze())
    actual_true = np.array(y.cpu().numpy()).squeeze()

    if norm:
        for i in range(actual_pred.shape[0]):
            actual_pred[i] = actual_pred[i] * normalizer[i][1][1] + normalizer[i][1][0]
            actual_true[i] = actual_true[i] * normalizer[i][1][1] + normalizer[i][1][0]
    
    return actual_pred, actual_true