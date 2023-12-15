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
from threading import Thread
from concurrent.futures import ProcessPoolExecutor

# real threads
num_threads = 8
# num works which will be divided between threads
num_works = 100

# Изменяемые параметры модели N, K, L, CountOfExperiments  - задаются как аргументы командной строки при запуске
parser = argparse.ArgumentParser(description="Set parameters N, K, L for series")
parser.add_argument("N")
parser.add_argument("K")
parser.add_argument("L")
parser.add_argument("CountOfExperiments")
args = parser.parse_args()
N = int(args.N)
K = int(args.K)
L = int(args.L)
CountOfExperiments = int(args.CountOfExperiments)

num_works = 100

# flag for debug
debug = 0 
# if during this count of iter e have no changes in A & B matrices, we enable debug flag automatically
bad_s = 1000 
#for debug mode, if i > bad_i, we enable debug flag automitically
bad_i = 1000000
# if i - iter of sync lean inside of [1, border] we write syncronization success, else inc overborder var (below)
border = 1000000

#-------------------------------------------------
# replace all zeros by minus one
def transform_message(m):
	for i in range(N):
		for j in range(K):
			m[j][i] = (m[j][i] - 0.5) * 2
	return m
#------------------------------------(transform_message)

#-------------------------------------------------
# calculate inner number
def get_inter(m, s):
	return np.where(np.greater(np.sum(m * s, axis = 1), 0), 1, -1)

#------------------------------------(get_inter)

#-------------------------------------------------
# return number to range
def return_in_border(T):
	np.clip(T, -L, L, out=T)
	return T
#-------------------------------------(return_in_border)

#------------------------------------
#make calculations of result for message
def make_calc(arr, message):
	inter = get_inter(message, arr)
	result = np.prod(inter)
	return result
#------------------------------------(make_calc)

#------------------------------------
#modification mechanism for A & B
def modify(arr, message, my_result, other_result):
	new_arr = arr.copy()
	inter = get_inter(message, arr)
	for i in range (K):
		new_arr[i,:] = arr[i,:] + inter[i] * message[i,:] * (my_result==other_result and my_result==inter[i])
	return return_in_border(new_arr)
#------------------------------------(modify)

#------------------------------------
#modification mechanism for simple encryptor
def modify_encryptor(e, message, r_a, r_b, my_result):
	new_arr = e.copy()
	inter = get_inter(message, e)
	for i in range(K):
		new_arr[i,:] = e[i,:] + inter[i]*message[i,:]*(r_a==r_b and r_a==my_result and r_a==inter[i])
	return return_in_border(new_arr)
#------------------------------------(modify_encryptor)

#------------------------------------
#modification mechanism for gineous encryptor 
def modify_gineous(g, m, r_a, r_b, r_g):
	new_g = g.copy()
	i_g = get_inter(m, g)
	if (r_g==r_a and r_g==r_b):
		new_g = g + np.multiply(m, i_g.reshape(-1, 1))*(np.equal(r_g, i_g).reshape(-1, 1))
	elif (r_g==r_a and r_g!=r_b):
		index = np.argmin(np.abs(i_g))
		new_g[index,:] = g[index,:] + i_g[index]*m[index,:]
		
	return return_in_border(new_g)
#------------------------------------(modify_gineous)

#------------------------------------
#change all coeffs for one input message
def change_coefficient(a, b, m, e, g, e_flag = "disable", g_flag = "disable"):

	# results
	r_a = make_calc(a, m)
	r_b = make_calc(b, m)

	new_a = modify(a, m, r_a, r_b)
	new_b = modify(b, m, r_b, r_a)

	# debug info
	if (debug):
		i_a = get_inter(m, a)
		i_b = get_inter(m, b)
		print(str(a.tolist())+" "+str(i_a)+" "+str(r_a))
		print(str(b.tolist())+" "+str(i_b)+" "+str(r_b))
		for i in range(K):
			print("a"+str(i)+" = "+str(a[i,:])+" + "+str(i_a[i])+" * "+str(m[i,:])+" * "+str(np.heaviside(r_a*i_a[i],0))+" * "+str(np.heaviside(r_a*r_b,0))+" = "+str(new_a[i,:]))
			print("b"+str(i)+" = "+str(b[i,:])+" + "+str(i_b[i])+" * "+str(m[i,:])+" * "+str(np.heaviside(r_b*i_b[i],0))+" * "+str(np.heaviside(r_a*r_b,0))+" = "+str(new_b[i,:]))
		
		print(str(new_a.tolist()))
		print(str(new_b.tolist()))

	#make calculations with encryptors enabled
	if (e_flag == "enable"):
		r_e = make_calc(e, m)
		new_e = modify_encryptor(e, m, r_a, r_b, r_e)
	else:
		new_e = e.copy()
	if (g_flag == "enable"):
		r_g = make_calc(g, m)
		new_g = modify_gineous(g, m, r_a, r_b, r_g)
	else:
		new_g = g.copy()

	#return
	return new_a, new_b, new_e, new_g, int(r_a == r_b)
#------------------------------------(change_coefficient)

#-------------------------------------------------
#make binary array
def bin_array(num, m):
	return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)
#-------------------------------------------------(bin_array)

#-------------------------------------------------
# debug function - print possible message that change secret for case which we have no changes in A & B for long time(iterations)
def find_changed_message(a, b):
	t_a = a.copy()
	t_b = b.copy()

	#this is need for correct work of change_coefficient. Ignored
	tmp_e = a.copy()
	tmp_g = a.copy()

	r = 0
	key = 0
	for i in range(2**(N*K)):
		key = bin_array(i, N*K)
		print(key.tolist())
		key = np.array(key).reshape((K,N))
		key = transform_message(key)
		t_a, t_b, tmp_e, tmp_g, r = change_coefficient(t_a, t_b, key, tmp_e, tmp_g)
		if (r == 1):
			break
	print("not_found key")
	print(key)
	debug = 1
	t_a, t_b, tmp_e, tmp_g, r = change_coefficient(t_a, t_b, key, tmp_e, tmp_g)
	debug = 0
	return 
#-------------------------------------------------(find_changed_message)
	
#-------------------------------------------------
def debug_sit(A, B, s, i, debug, bad_s, bad_i):
	fail = 0
	if (s > bad_s or i > bad_i):
		print("problem")
		debug = 1
		decide = input()
		if (decide == "c"):
			debug = 0
			s = 0
		elif (decide == "d"):
			debug = 0
			bad_s = bad_s * 2
			s = 0
		elif (decide == "y"):
			debug = 0
			bad_i = bad_i * 2
			s = 0
		elif (decide == "a"):
			debug = 0
			find_changed_message(A, B)
		elif (decide == "s"):
			debug = 0
			i = i - 1
		elif (decide != ""):
			print("fail")
			fail = 1
	return debug, bad_s, bad_i, s, i, fail
#-------------------------------------------------(debug_sit)

#-------------------------------------------------
def make_experiment(A, B, E, G, step_sync, debug, bad_s, bad_i):
	s = 0
	i = 0
	r = 0
	while (True):
		# set random array with -1 or 1
		M = np.random.randint(0, 2, size=(K,N))*2-1

		A, B, E, G, r = change_coefficient(A, B, M, E, G, e_flag = "enable", g_flag = "enable")

		# check how long we stuck (maybe its dont_change_loop)
		if (r == 0):
			s = s + 1
		else:
			s = 0

		# print(A.tolist())
		# print(B.tolist())
		# print(M.tolist())
		debug, bad_s, bad_i, s, i, fail = debug_sit(A, B, s, i, debug, bad_s, bad_i)
		if (fail):
			break

		i = i + 1

		# check of sync
		if (not ((step_sync[0] != 0) or (not (A == B).all()))):
			step_sync[0] = i
		if (not ((step_sync[1] != 0) or (not (A == E).all()))):
			step_sync[1] = i
		if (not ((step_sync[2] != 0) or (not (B == E).all()))):
			step_sync[2] = i
		if (not ((step_sync[3] != 0) or (not (A == G).all()))):
			step_sync[3] = i
		if (not ((step_sync[4] != 0) or (not (B == G).all()))):
			step_sync[4] = i

		if (step_sync[0] != 0):
			break
	return A, step_sync
#-------------------------------------------------(make_experiment)

#-------------------------------------------------
def afterwork(over_boarder, p_syncronize_step, step_sync):

	for i in range(5):
		if (step_sync[i] > border):
			over_border[i] += 1
		elif (step_sync[i] != 0):
			p_syncronize_step[step_sync[i],i] += 1

	return over_boarder, p_syncronize_step
#-------------------------------------------------(afterwork)
	
#-------------------------------------------------
def write_p(p_syncronize_step, p_weights):
	file_name_N = "./T_N/N-N"+str(N)+"-K"+str(K)+"-L"+str(L)+".csv"
	file_name_S = "./T_L/L-N"+str(N)+"-K"+str(K)+"-L"+str(L)+".csv"
	try:
		data_frame_N = pd.read_csv(file_name_N, header=None)
		data_frame_S = pd.read_csv(file_name_S, header=None)
		data_N = data_frame_N.to_numpy()
		data_S = data_frame_S.to_numpy()
		for i in range (min(len(p_syncronize_step), len(data_N))):
			p_syncronize_step[i] = p_syncronize_step[i] + data_N[i]
		p_weights = p_weights + data_S[:,0]
	except FileNotFoundError:
		print("nothing before")

	data_frame_N = pd.DataFrame(p_syncronize_step)
	data_frame_N.to_csv(file_name_N, header=None, index=None)

	data_frame_S = pd.DataFrame(p_weights)
	data_frame_S.to_csv(file_name_S, header=None, index=None)
#-------------------------------------------------(write_p)

#-------------------------------------------------
def thread_work(thread_id):
	p_weights = np.zeros(2*L+1)
	over_border = np.zeros(5)
	p_syncronize_step = np.zeros((border, 5),dtype=int)
	bad_work = np.zeros(4)

	debug_loc = 0 
	bad_s_loc = 1000 
	bad_i_loc = 1000000 
	for _ in range(thread_id, CountOfExperiments, num_works):
		A = np.random.randint(0, 2*L + 1, size=(K,N)) - L
		B = np.random.randint(0, 2*L + 1, size=(K,N)) - L
		E = np.random.randint(0, 2*L + 1, size=(K,N)) - L
		G = np.random.randint(0, 2*L + 1, size=(K,N)) - L

		step_sync = np.zeros(5, dtype = int)
		#make experiment
		A, step_sync = make_experiment(A, B, E, G, step_sync, debug_loc, bad_s_loc, bad_i_loc)

		#afterwork of experiment done
		if (any(step_sync[1:] < step_sync[0])):
			minimum = min(step_sync[1:])
			for i in range(1,5):
				if (step_sync[i] == minimum and step_sync[i] != 0):
					bad_work[i - 1] += 1
		
		over_border, p_syncronize_step = afterwork(over_border, p_syncronize_step, step_sync)
		for t in range(N):
			for j in range(K):
				p_weights[int(A[j][t])+L] += 1

	return p_weights, over_border, p_syncronize_step, bad_work
#-------------------------------------------------(thread_work)

##################################################
## 						MAIN					##
##################################################

over_border = np.zeros(5)
p_syncronize_step = np.zeros((border, 5),dtype=int)
print(np.shape(p_syncronize_step))
p_weights = np.zeros(2*L+1)

steps = np.arange(0,CountOfExperiments)
bad_work = np.zeros(4)

bar = IncrementalBar('Done', max = num_works)
with ProcessPoolExecutor(num_threads) as exe:
	futures = []
	for i in range(num_works):
		futures.append(exe.submit(thread_work, i))
	for i in range(num_works):
		p_weights_loc, over_border_loc, p_syncronize_step_loc, bad_work_loc = futures[i].result()
		p_weights += p_weights_loc
		over_border += over_border_loc
		p_syncronize_step += p_syncronize_step_loc
		bad_work += bad_work_loc
		bar.next()

bar.finish()

#-------------------------------------------------
print(str(CountOfExperiments) + " experiments with (N, K, L): (" + str(N) + ", " + str(K) + ", " + str(L) + ") made!")
print("Over border: "+str(over_border)+" ("+str(over_border*100/border)+"%)")
print("Bad sync "+str(bad_work))
print(p_weights)

#-------------------------------------------------
#wright p to files
write_p(p_syncronize_step, p_weights)

#-------------------------------------------------

file_name_R = "./T_Results.csv"
not_found = 0

try:
	data_frame_R = pd.read_csv(file_name_R, header=None)
	data_R = data_frame_R.to_numpy(dtype=float)
except FileNotFoundError:
	not_found = 1
	print("nothing before")

if (not_found):
	file = open(file_name_R, "w")
	file.write("0," * 12 +"0" )
	file.close()

	data_frame_R = pd.read_csv(file_name_R, header=None)
	data_R = data_frame_R.to_numpy(dtype=float)

if( data_frame_R[(data_frame_R[0].isin([N])) & (data_frame_R[1].isin([K])) & (data_frame_R[2].isin([L]))].empty):
	tmp = [N, K, L,CountOfExperiments,bad_work[0], bad_work[1],bad_work[2],bad_work[3],over_border[0],over_border[1],over_border[2],over_border[3],over_border[4]]
	data_frame_R.loc[len(data_frame_R.index)] = tmp
else:
	idx = data_frame_R[(data_frame_R[0] == N) & (data_frame_R[1] == K) & (data_frame_R[2] == L)].index[0]
	data_R[idx][3] = data_R[idx][3] + CountOfExperiments
	data_R[idx][4:8] = data_R[idx][4:8] + bad_work[:]
	data_R[idx][8:] = data_R[idx][8:] + over_border[:]
	data_frame_R = pd.DataFrame(data_R)


data_frame_R = data_frame_R.sort_values(by=[0, 1, 2])
data_frame_R.to_csv(file_name_R, header=None, index=None)
