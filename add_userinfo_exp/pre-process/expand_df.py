import numpy as np
import re
import pandas as pd

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
movie_df = read_movie('../../data/ml-1m/movies.dat')
movie_df.head()


def expand_row(row, key_col, expand_col):
    '''
    expand_col: array
    '''
    expand = row[expand_col] 
    key = [row[key_col]] * len(row[expand_col]) 
    df = pd.DataFrame({key_col:key, expand_col: expand})
    return df

movie_df_expand = pd.DataFrame({'category': [], 'movie_id': []})
for i in range(10):
    row = movie_df.iloc[i]
    movie_df_expand = movie_df_expand.append(expand_row(row, 'movie_id', 'category'), ignore_index=True)
print movie_df_expand
pd.merge(movie_df[['movie_id', 'name']], movie_df_expand, on = ['movie_id'])