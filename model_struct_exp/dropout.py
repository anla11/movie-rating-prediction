import numpy as np
import random

def dropout(A, dropout_parm, train_mode = True):
	if dropout_parm != None:
		prb = dropout_parm[0]
		if train_mode:
			mask = dropout_parm[1]
			A = A * mask
		else:
			A *= prb
	return A