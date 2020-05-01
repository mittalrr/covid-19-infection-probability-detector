import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data, ratio):
		np.random.seed(42)
		shuffled = np.random.permutation(len(data))
		test_set_size = int(len(data) * ratio)
		test_indices = shuffled[:test_set_size]
		train_indices = shuffled[test_set_size:]
		return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
	df = pd.read_csv('datacsv.csv')
	train, test = data_split(df, 0.2)
	
	x_train = train[['fever', 'bodypain', 'age', 'runnynose', 'diffbreath']].to_numpy()
	x_test = test[['fever', 'bodypain', 'age', 'runnynose', 'diffbreath']].to_numpy()
	
	y_train = train[['infectionprob']].to_numpy().reshape(2060,)
	y_test = test[['infectionprob']].to_numpy().reshape(515,)
	
	clf = LogisticRegression()
	clf.fit(x_train, y_train)

	clf = LogisticRegression()
	clf.fit(x_train, y_train)

	filename = 'model.pkl'
	file = open(filename,'wb')

	pickle.dump(clf,file)
	file.close()

	inputfeatures = [100,1,2,-1,1]
	infprob = clf.predict_proba([inputfeatures])[0][1]