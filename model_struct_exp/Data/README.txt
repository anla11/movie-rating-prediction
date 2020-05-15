In this file, there're 2 dataset: ML100K and HetRec2011.

HetRec2011 is processed by ProcessHetRec/Preprocessing.ipynb. I copied hec_train, hec_validation and hec_test and placed it in this folder.

This data is collected from MovieLens 100K Dataset - GroupLens (http://grouplens.org/datasets/movielens/100k/).

MovieLens dataset includes many files, but we just use u.data and u.item to create our data.

1. File Structure
- u.data   The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	         user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC   

u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.

1. Convert u.data to *.csv and rename as user_rating.csv
2. Due to unknow value, we know whether or not a movies has missing value. Remove it from data.
3. Extract date, month, and year from video release date and create 3 columns to store them.
3. Remove unnecessary colums: movie title, release date, video release date, IMDb URL
4. Convert u.item to *.csv and rename as movies_feature.csv