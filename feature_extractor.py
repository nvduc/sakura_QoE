#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import cv2 as cv
from feature_measure import *
import os
from os import listdir
from os.path import isfile, join

import time
import pandas as pd
import sys


def avg_min_max_std(arr):
    return np.mean(arr), np.min(arr), np.max(arr), np.std(arr)

# Extract feature of all videos stored in pah
#
def feature_extractor_all(path):

	# get list of videos
	file_list = [f for f in listdir(path) if isfile(join(path, f))]
	
	file_list.sort()

	# print(file_list)

	# extract video features
	feature_all = []
	for file in file_list:
		feature = feature_extractor_single(path, file)
		feature_all.append(feature)
	return file_list, np.array(feature_all)


def feature_extractor_single(path, video_name):

	res_enc = {(720,1280): 1, (1280, 720):2}

	# extract QP and bitrate information
	cmd='ffmpeg_debug_qp_parser {}{} ./tmp/tmp.csv --force -a -of csv'.format(path, video_name)
	# print(cmd)

	os.system(cmd)

	df = pd.read_csv('./tmp/tmp.csv')
	size = np.sum(df.frame_size.values)/10e5
	qp_avg,qp_min,qp_max,qp_std = avg_min_max_std(df.qp_avg.values)

	if qp_avg == 0:
		qp_avg,qp_min,qp_max,qp_std = 15, 15, 15, 15

	########### Extract SI/TI information ################
	cmd = 'siti -i {}{} 2> ./tmp/log_siti_tmp.csv'.format(path, video_name)
	os.system(cmd)

	df = pd.read_csv('./tmp/log_siti_tmp.csv')
	si_avg, si_min, si_max, si_std = avg_min_max_std(df.SI.values)
	ti_avg, ti_min, ti_max, ti_std = avg_min_max_std(df.TI.values[:-1])


	# print(si_avg,si_min,si_max,si_std)
	# print(ti_avg,ti_min,ti_max,ti_std)

	cap = cv.VideoCapture('{}{}'.format(path, video_name))

	cnt = 0
	blur1, blur2, blur3 = [], [], []
	while(cap.isOpened()):
	    ret, frame = cap.read()
	    if ret == False:
	        break

	    if cnt % 15 == 0:
	        img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)[:,:,0]

	        cv.imwrite("./tmp/frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
	        img = cv.imread("./tmp/frame.jpg", 0)

	        blur1.append(blur_measure_1(img, 8.0)) # --> blur
	        blur2.append(blur_measure_1(img, 6.0))# --> blur2

	        x,y = blur_detect(img, 35) # --> blur3
	        blur3.append(y)
	       	# blur4,blur4_ext = blur_detect(img, 25)
	        
	        # print("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f"%(cnt, blur1, blur2, blur3, blur3_ext, blur4, blur4_ext))
	    cnt += 1
	# blur1_avg, blur1_min, blur1_max, blur1_std = avg_min_max_std(blur1)
	blur2_avg, blur2_min, blur2_max, blur2_std = avg_min_max_std(blur2)
	blur3_avg, blur3_min, blur3_max, blur3_std = avg_min_max_std(blur3)

	res = res_enc[img.shape]
	# return [qp_avg,size,qp_std, qp_max, qp_min, ti_min, ti_avg, ti_std,ti_max, blur2_avg,blur2_min, res, blur3_avg, blur3_max, blur3_min, blur3_std]

	return [size, res, qp_avg,qp_min,qp_max,qp_std, si_avg, si_min, si_max, si_std, ti_avg,ti_min,ti_max,ti_std,blur2_avg, blur2_min, blur2_max, blur2_std, blur3_avg, blur3_min, blur3_max, blur3_std]

	



# df_test = pd.read_csv('test.csv')

# features = []
# path='ICME_challenge/video/'


# file_list = [f for f in listdir(path) if isfile(join(path, f))]
# print(file_list, len(file_list))



# print(time.time() - begin)
# #

def RMSE(y_pred, y):
    return np.sqrt(np.mean((y_pred - y)*(y_pred - y)))
def PCC(y_pred, y):
    y_pred_mean = np.mean(y_pred)
    y_mean = np.mean(y)
    a = np.dot(y - y_mean, y_pred - y_pred_mean)
    b = np.sqrt(np.sum((y-y_mean)*(y-y_mean))) * np.sqrt(np.sum((y_pred-y_pred_mean)*(y_pred-y_pred_mean)))
    return a*1.0/b
def pcc_rmse(y_pred, y):
    return PCC(y_pred, y), RMSE(y_pred, y)

# Helper function to normalize data
def normalize(X):
    return (X - X.min())/(X.max() - X.min())
def normalize2(X):
    return 2 * (X - X.min())/(X.max() - X.min()) - 1

# Method to make predictions
def predict(X, b0, b1):
    return np.array([1 / (1 + exp(-1*b0 + -1*b1*x)) for x in X])

# Method to train the model
def logistic_regression(X, Y, L, epochs):

    X = normalize(X)

    # Initializing variables
#     b0 = 0
#     b1 = 0
    b0, b1 = 2 * np.random.rand() - 1,2 * np.random.rand() - 1
#     L = 0.001
#     epochs = 300

    for epoch in range(epochs):
        y_pred = predict(X, b0, b1)
        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0
        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
        # Update b0 and b1
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1
#         print(epoch, b0, b1)
    
    return b0, b1


def main():
	
	print('Extracting video features...')
	file_list, feature = feature_extractor_all('dataset/train/')
	feature_names = ['size', 'res', 'qp_avg','qp_min','qp_max','qp_std', 'si_avg', 'si_min', 'si_max', 'si_std', 'ti_avg','ti_min','ti_max','ti_std','blur2_avg', 'blur2_min', 'blur2_max', 'blur2_std', 'blur3_avg', 'blur3_min', 'blur3_max', 'blur3_std'];
	df = pd.DataFrame(feature, columns=feature_names)
	df['video'] = file_list
	df.to_csv('dataset/train_feature.csv', index=None)
	print('Train video features have been saved to dataset/train_feature.csv')

	file_list, feature = feature_extractor_all('dataset/test/')
	df = pd.DataFrame(feature, columns=feature_names)
	df['video'] = file_list
	df.to_csv('dataset/test_feature.csv', index=None)
	print('Test videos features have been saved to dataset/test_feature.csv')

if __name__ == "__main__":
	main()


