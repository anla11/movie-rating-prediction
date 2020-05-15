from ReadData import read_data, write_data
import pandas as pd
import numpy as np
import csv

def     read_data(path, delimiter = ','):
    tmp = []     
    with open(path) as f:
        lines = f.readlines()       
        for line in lines:
            line = line[:-1] #remove last charactor
            tmp.append(map(int, line.split(delimiter)))
    data = np.vstack(tmp)
    return data

def     write_data(path, data, delimiter = ',', format = "%d"):
    np.savetxt(path, data, delimiter = delimiter,  fmt=format)


def 	scale(train, val, test):
    mean = np.mean(train) 
    std = np.sqrt(np.sum((train - mean) ** 2))
    train = (train - mean) / std
    val  = (val - mean) / std
    test = (test - mean) / std
    tmin = np.min(train)
    tmax = np.max(train)
    train = (train - tmin) / (tmax - tmin)
    val = (val - tmin) / (tmax - tmin)
    test = (test - tmin) / (tmax - tmin)
    return train, val, test


def 	preprocess_data(feature_data, user_rating_data):
	feature = read_data(feature_data)
	"""  
	DAY|MONTH|YEAR|unknown | Action | Adventure | Animation |
		Children's | Comedy | Crime | Documentary | Drama | Fantasy |
	    Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
	    Thriller | War | Western |
	"""

	user = read_data(user_rating_data)
	""" user id | item id | rating | timestamp. """ 

	user_df = pd.DataFrame(user)
	#name for colume in data frame
	user_df.columns = ["user id", "item id", "rating", "timestamp"]

	# calculating mean of rating for each movie
	grouped = user_df.groupby("item id")
	rating = grouped["rating"].mean().as_matrix()

	#remove missing value
	print 'Number of row(s) in original data       : %d' % len(feature)
	rating = np.asarray([rating[i] for i in range(len(feature)) if feature[i][3] == 0])
	feature = np.asarray([feature[i] for i in range(len(feature)) if feature[i][3] == 0])
	print 'Number of row(s) in data removed by unknown value: %d' % len(feature)

	#after filtering by unknow column, delete it
	feature = np.delete(feature, 3, axis = 1)
	#delete year column
	feature = np.delete(feature, 2, axis = 1)

	#shuffle 
	rand_idx = np.arange(len(feature))
	np.random.shuffle(rand_idx)
	feature = feature[rand_idx]
	rating = rating[rand_idx]

	rating = rating.reshape(-1, 1)	

	movie = np.hstack((feature, rating))
	np.savetxt('movies.csv', movie, fmt='%.3f', delimiter = ',') 
	return feature, rating

def 	get_data(path, delimiter = ',', header = False):
	tmp = []     
	with open(path) as f:
		lines = f.readlines()
		if header == True:
			lines = lines[1:]
		for line in lines:
			line = line[:-1] #remove last charactor
			tmp.append(map(float, line.split(delimiter)))
	data = np.vstack(tmp)
	X = data[0:data.shape[0], 0:data.shape[1]-1]
	Y = data[0:data.shape[0], data.shape[1]-1]
	Y = Y.reshape(-1, 1)
	return X, Y
