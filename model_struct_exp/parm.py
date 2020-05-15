def 	init():
    global n_epoch, layers, mnb_size, learning_rate, momentum, lamda, stop, max_patience, run_batchnorm, run_dropout, act_func, SIGMOID, TANH, RELU, NOFUNC, alpha, run_test, start, trainfile, valfile, testfile

    n_epoch = 10000	#fix
    layers = [15] 	#fix
    mnb_size = 150 	# fix 
    #early stopping
    max_patience = 6000 #fix
    
    learning_rate = 0.005 #modify this
    momentum = 0.9	#modify this
    
    #weight decay level
    lamda = 0.00	#modify this
    run_dropout = False # use dropout - modify this

    run_batchnorm = False

    ''' activation function apply for hidden layer '''
    SIGMOID = 1
    TANH = 2
    RELU = 3 
    NOFUNC = 4
    act_func = RELU # SIGMOID, TANH, RELU - modify this

    ''' if use RELU, set value for alpha '''
    alpha = -1 #fix

    ''' if train_ok == True, test will be run.
        otherwise, just training.
    '''
    run_test = True #fix

    ''' if run_test == True, result will be written in file'''
    ''' modify this'''
    trainfile = "Result/train.csv"
    valfile = "Result/val.csv"
    testfile = "Result/test.csv"	

    ''' Error lines in graph will start at epoch = start '''
    start = 10 #fix

    ''' number of fold in cross-validation '''
    stop = n_epoch #fix