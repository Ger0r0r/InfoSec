#################################
##       This file contains tools for analys of single experiments, and for whole series of experiments
#################################  


import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import time
from progress.bar import IncrementalBar
from scipy.optimize import curve_fit
import math
import pandas as pd
import os
import argparse
import sys



# plot smth for single parameters - NKL, max_steps smth else????

def plot_NKL_series(N, K, L, border, points):

	file_name_N = "./N/N-N"+str(N)+"-K"+str(K)+"-L"+str(L)+".csv"
	file_name_S = "./L/L-N"+str(N)+"-K"+str(K)+"-L"+str(L)+".csv"

	try:
		data_frame_N = pd.read_csv(file_name_N, header=None)
		data_frame_S = pd.read_csv(file_name_S, header=None)
		data_N = data_frame_N.to_numpy()
		data_S = data_frame_S.to_numpy(dtype=float)
	except FileNotFoundError:
		print("no such file")
	#---------------------------------------

	group = int(border/points)
	y = np.zeros(points)
	for i in range(points):
		temp = data_N[(i*group):((i+1)*group),0]
		y[i] = np.sum(temp)

	x = np.arange(0,border,group)
	y = y / np.sum(y)
	print("pike")
	print(x[np.where(y == max(y))])
	plt.bar(x, y, width=group)
	plt.show()
	#---------------------------------------
	y = np.zeros(points * 5)
	names = ["ab", "ae", "ag", "be", "bg"]
	whole_sums = np.zeros(5)
	y = y.reshape(5, points)
	offset = 0
	for j in range (5):
		for i in range(points):
			temp = data_N[(i*group):((i+1)*group),j]
			y[j][i] = np.sum(temp)
		
		whole_sums[j] = np.sum(data_N[:, j])
		if (whole_sums[j] == 0):
			whole_sums[j] = np.sum(y)

		x = np.arange(0,border,group)
		y = y / whole_sums[j]
		plt.plot(x, y[j], label=names[j])
		offset += group/5
	
	plt.legend()
	plt.show()

	#----------------------------------------
	x = np.arange(-L,L+1)
	data_S = data_S / np.sum(data_S)
	data_S = data_S.reshape(2 * L + 1)

	plt.bar(x, data_S, width=1)
	plt.show()
	#----------------------------------------
#----------------------------------------(plot_NKL_series)

print(sys.argv)
if (len(sys.argv) != 6):
	print("Usage: "+sys.argv[0]+" <N> <K> <L> <GRAPH_BORDER> <NUM_POINTS>")
	sys.exit()

N = int(sys.argv[1])
K = int(sys.argv[2])
L = int(sys.argv[3])
Graph_border = int(sys.argv[4])
Num_points = int(sys.argv[5])

# plot_NKL_series(2, 3, 3,1000, 100)
plot_NKL_series(N, K, L, Graph_border, Num_points)
#plot smth for different N, K, L