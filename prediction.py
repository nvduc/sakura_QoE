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

def main():
	path = sys.argv[1]

	# extract video features
	print('Extracting video features...')
	file_list, feature = feature_extractor_all(path)

	# load pretrained model and calculate results
	print('Predicting MOS values...')
	model = pickle.load(open('pretrained/model.pkl', 'rb'))
	pred = model.predict(feature)
	
	df_result = pd.DataFrame(file_list, columns=['video'])
	df_result['predicted_mos'] = pred

	df_result.to_csv('test_result.txt', index=None)
	print('Prediction results has been saved to file test_result.txt!!!')

if __name__ == "__main__":
	main()