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

