import numpy as np
import re
import pandas as pd


def convert_cate_tocol(df, id_cols, cate_col, multi_cate, merge = True):
	def mapping(cat_list, all_categories):
		category_vec = np.zeros(len(all_categories)).astype(int)
		if multi_cate == True:
			cat_list = set(cat_list)
		else:
			cat_list = set([cat_list])
		for i in range(len(all_categories)):
			if all_categories[i] in cat_list:
				category_vec[i] = 1
		return category_vec
	
	all_categories = None
	if multi_cate:
		all_categories = np.unique(sum(df[cate_col], []))
	else:
		all_categories = np.unique(df[cate_col])
	
	res_df = pd.DataFrame([mapping(df.iloc[i][cate_col], all_categories) for i in range(len(df))],columns=all_categories)
	
	res_df.columns = ('%s_' % cate_col) + res_df.columns
	for col in id_cols:
		res_df[col] = df[col]
		
	if merge == True:
		res_df = pd.merge(res_df, df, on = id_cols)
	return res_df

def read_movie(filename):
	tmp_i = []
	tmp_n = []
	tmp_c = []
	with open(filename) as f:
		for line in f:
			line_split = re.split('::', line)
			tmp_i.append(line_split[0])
			tmp_n.append(line_split[1])
			tmp_c.append(line_split[2][:-1].split('|'))

	movie_df = pd.DataFrame({'movie_id':tmp_i, 'name':tmp_n, 'category':tmp_c})
	return movie_df

def read_user(filename):
	user_id = []
	gender = []
	age = []
	occu = []
	zip_code = []
	with open(filename) as f:
		for line in f:
			line_split = re.split('::', line)
			user_id.append(line_split[0])
			gender.append(line_split[1])
			age.append(line_split[2])
			occu.append(line_split[3])
			zip_code.append(line_split[4][:-1])
	user_df = pd.DataFrame({'user_id':user_id, 'gender':gender, 'age':age, 'occupation':occu,'zipcode':zip_code})
	return user_df

def read_rating(filename):
	user_id = []
	movie_id = []
	rating = []
	timestamp = []
	with open(filename) as f:
		for line in f:
			line_split = re.split('::', line)
			user_id.append(line_split[0])
			movie_id.append(line_split[1])
			rating.append(line_split[2])
			timestamp.append(line_split[3][:-1])
	rating_df = pd.DataFrame({'user_id':user_id, 'movie_id':movie_id, 'rating': rating, 'timestamp':timestamp})
	return rating_df


movie_df = read_movie('../../data/ml-1m/movies.dat')
user_df = read_user('../../data/ml-1m/users.dat')
rating_df = read_rating('../../data/ml-1m/ratings.dat')

movie_df = convert_cate_tocol(movie_df, ['movie_id', 'name'], 'category', multi_cate = True, merge = False)
rating_df['rating'] = rating_df['rating'].astype(float) 
rate_dcrb_df = rating_df.groupby(['movie_id'])['rating'].describe(percentiles = np.array(range(1, 10)) * 1.0/10) 
rate_dcrb_df['std'] = rate_dcrb_df['std'].fillna(0) 
rate_dcrb_df = rate_dcrb_df.reset_index() 
cate_rate_movie_df = pd.merge(rate_dcrb_df, movie_df, on = ['movie_id']) 

cate_rate_movie_df.to_csv('../../data/pre-processed/movie_all.csv') 
user_df.to_csv('../../data/pre-processed/user_all.csv') 
cvage_user_df = convert_cate_tocol(user_df, id_cols=['user_id'], cate_col='age', multi_cate=False, merge=False)
cvgender_user_df = convert_cate_tocol(user_df, id_cols=['user_id'], cate_col='gender', multi_cate=False, merge=False)
cvoccu_user_df = convert_cate_tocol(user_df, id_cols=['user_id'], cate_col='occupation', multi_cate=False, merge=False)
del cvage_user_df['age_56']
del cvgender_user_df['gender_F']
del cvoccu_user_df['occupation_20']
tmp1 = pd.merge(cvage_user_df, cvgender_user_df, on = ['user_id'])
cvt_user_df = pd.merge(tmp1, user_df[['user_id', 'zipcode']], on = ['user_id'])
cvt_user_df.to_csv('../../data/pre-processed/cvt_user.csv')