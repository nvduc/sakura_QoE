from feature_extractor import *
import sys
import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import exp
from sklearn.feature_selection import SelectPercentile, f_classif

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

	# load video feature of tthe train/test sets
	df_train_feature = pd.read_csv('dataset/train_feature.csv')
	df_test_feature = pd.read_csv('dataset/test_feature.csv')

	df_train_label = pd.read_csv('dataset/train_label.csv')
	df_test_label = pd.read_csv('dataset/test_label.csv')

	df_train = df_train_feature.merge(df_train_label, left_on='video', right_on='video')
	df_test = df_test_feature.merge(df_test_label, left_on='video', right_on='video')

	print(df_train.shape, df_test.shape)

	y_train = df_train['mos'].values
	y_test = df_test['mos'].values

	feature_names = ['size', 'res', 'qp_avg','qp_min','qp_max','qp_std', 'si_avg', 'si_min', 'si_max', 'si_std', 'ti_avg','ti_min','ti_max','ti_std','blur2_avg', 'blur2_min', 'blur2_max', 'blur2_std', 'blur3_avg', 'blur3_min', 'blur3_max', 'blur3_std'];
	
	X_train = df_train[feature_names].values
	X_test = df_test[feature_names].values

	clf = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=100, max_features=0.5)
	clf.fit(X_train, y_train)
	f_importance = clf.feature_importances_
	for i in range(len(feature_names)):
		print('{}: {:.4f}'.format(feature_names[i], f_importance[i]))

	# calculate performance metrics
	y_train_pred = clf.predict(X_train)
	y_test_pred = clf.predict(X_test)

	pcc_train, rmse_train = pcc_rmse(y_train, y_train_pred)
	pcc_test, rmse_test = pcc_rmse(y_test, y_test_pred)
	print('pcc_train:', pcc_train, ',rmse_train:', rmse_train, '\npcc_test:', pcc_test, ',rmse_train:', rmse_test)

	# saving predictions results
	df_train['mos_pred'] = y_train_pred
	df_test['mos_pred'] = y_test_pred

	df_train[['video', 'mos', 'mos_pred']].to_csv('dataset/train_pred.csv', index=None)
	df_test[['video', 'mos', 'mos_pred']].to_csv('dataset/test_pred.csv', index=None)

	# saving model
	pickle.dump(clf, open('pretrained/model.pkl', 'wb'))



	# # load pretrained model and calculate results
	# print('Predicting MOS values...')
	# model = pickle.load(open('pretrain.pkl', 'rb'))
	# pred = model.predict(feature)
	
	# df_result = pd.DataFrame(file_list, columns=['video'])
	# df_result['predicted_mos'] = pred

	# df_result.to_csv('test_result.txt', index=None)
	# print('Prediction results has been saved to file test_result.txt!!!')

if __name__ == "__main__":
	main()