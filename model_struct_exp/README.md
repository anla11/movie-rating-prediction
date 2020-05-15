# Movie Ratings Prediction using Artificial Neural Networks

In this project

+ Self-implement Neural Network and some advanced learning techniques: drop_out, batch_norm, weight_decay
+ Run experiments with [MovieLen](https://grouplens.org/datasets/movielens/100k/) and [HecLen](https://grouplens.org/datasets/hetrec-2011/) dataset

## I. Dependencies
   Anaconda 2.x is required.

## II. How to use the code
   Clone our ['How to use.ipynb'](https://github.com/anvy1102/movie-rating-prediction/blob/master/model_struct_exp/How%20to%20use.ipynb) to know how to use.

## III. File Structure
	Data 						Folder storing dataset
	active_function.py				Activation functions implementation
	ANN.py						Neural Network implementation
	batch_norm.py					Batch Normalization implemetation
	dropout.py					Dropout implemetation
	error_function.py				Error functions implementation
	get_data.py					Pre-process and read data from Dataset
	parm.py						Declare and init parameters of model
	plot.py						Graph Plotter
	run_ml100k.py, run_hec.py			Loading data, parameters and other functions to train and test model
	How to use.ipynb				Present sample code
	Results-ml100k.ipynb, Results_Hec.ipynb		Report our Exepiment and Result on ML 100K Data Set
	README.md					This file

1. Data folder
	Describe the original dataset and create our files to prepare for data pre-processing

2. Active_function.py
	Implement activation functions: f(x) = x, sigmoid, tanh, rectified linear and their gradidents.

3. ANN.py
	Implement neural network model.

4. batch_norm.py
	Implement functions of batch normalization.

5. dropout.py
	Implement 1 function of dropout.

6. error_function.py
	Include Mean Squared Error, Least Squares, Adjusted R-Squared Error.

7. get_data.py
	Include 2 main functions: preprocess_data and get_data.
	preprocess_data: read data from Data/movie_features.csv and Data/user_ratings.csv to create a data and save as Data/movies.csv. 
	get_data: read data from Data/movies.csv and return 2 variables: list of features of movies and list of ratings of corresponding movies.

8. parm.py
	Declare and init parameters of neural network model. You can assign new value for them. ANN.py takes value of those parameters to train and test model.

9. plot.py
	Includes codes to draw graph of a list of errors.

10. run_ml100k.py, run_hec.py
	This file loads all codes and define 2 functions: run_training and run_testing. You need to know how to use them to run this project. Details of 2 functions are describe with example code as 'Sample.ipynb'.

11. How to use.ipynb
	You need to read this file to know how to run this project.

12. Results-ml100k.ipynb, Results_Hec.ipynb
	This file reports all our exeperiment and results on ML 100K Data Set and HetRect2011: parameters of model, error in training set, validation set and testing set.

