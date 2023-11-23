import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import time
from progress.bar import IncrementalBar
from scipy.optimize import curve_fit
import math
import pandas as pd

##################################################

def f1(x, a, b, c):
    return a * np.exp(-(x-b)**2/(2*c**2))

##################################################

L=3
K=3
N=5

# print("N = ", end="")
# N = int(input())
# print("K = ", end="")
# K = int(input())
# print("L = ", end="")
# L = int(input())

debug = 0
bad_s = 1000
bad_i = 10000
border = 10000
max_steps = 100

##################################################

def gen_message ():
	mes = np.zeros((K,N))
	for i in range(N):
		for j in range(K):
			mes[j][i] = random.randint(0,1)
	return mes

#-------------------------------------------------

def transform_message(m):
	for i in range(N):
		for j in range(K):
			m[j][i] = (m[j][i] - 0.5) * 2
	return m

#-------------------------------------------------

def get_inter(m, s):
	inter = np.zeros(K)
	temp = m*s
	# print(str(temp.tolist()))
	for i in range(K):
		inter[i] = np.sum(temp[i])
		if (inter[i] > 0):
			inter[i] = 1
		else:
			inter[i] = -1
		# print(inter)
	return inter

#-------------------------------------------------

def calc_answer (m, s):
	inter = get_inter(m, s)
	# print(str(inter.tolist()))
	return np.prod(inter)

#-------------------------------------------------

def return_in_border(T):
	for i in range(N):
		for j in range(K):
			if (T[j][i] > L):
				T[j][i] = L
			if (T[j][i] < -L):
				T[j][i] = -L
	return T

#-------------------------------------------------

def change_coefficient (a, b, e, g, m):
	# copies of A, B (will be changed)
	new_a = a.copy()
	new_b = b.copy()
	new_e = e.copy()
	new_g = g.copy()

	# secret values of A, B
	i_a = get_inter(m, a)
	i_b = get_inter(m, b)
	i_e = get_inter(m, e)
	i_g = get_inter(m, g)

	# results of A, B
	r_a = np.prod(i_a)
	r_b = np.prod(i_b)
	r_e = np.prod(i_e)
	r_g = np.prod(i_g)

	if (debug):
		print(str(a.tolist())+" "+str(i_a)+" "+str(r_a))
		print(str(b.tolist())+" "+str(i_b)+" "+str(r_b))

	for i in range(K):
		# new_a[i,:] = a[i,:] + i_a[i]*m[i,:]*np.heaviside(r_a*i_a[i],0)*np.heaviside(r_a*r_b,0)
		# new_b[i,:] = b[i,:] + i_b[i]*m[i,:]*np.heaviside(r_b*i_b[i],0)*np.heaviside(r_a*r_b,0)
		new_a[i,:] = a[i,:] + i_a[i]*m[i,:]*(r_a==r_b and r_a==i_a[i])
		new_b[i,:] = b[i,:] + i_b[i]*m[i,:]*(r_b==r_a and r_b==i_b[i])
		if (debug):
			print("a"+str(i)+" = "+str(a[i,:])+" + "+str(i_a[i])+" * "+str(m[i,:])+" * "+str(np.heaviside(r_a*i_a[i],0))+" * "+str(np.heaviside(r_a*r_b,0))+" = "+str(new_a[i,:]))
			print("b"+str(i)+" = "+str(b[i,:])+" + "+str(i_b[i])+" * "+str(m[i,:])+" * "+str(np.heaviside(r_b*i_b[i],0))+" * "+str(np.heaviside(r_a*r_b,0))+" = "+str(new_b[i,:]))

	for i in range(K):
		new_e[i,:] = e[i,:] + i_e[i]*m[i,:]*(r_a==r_b and r_a==r_e and r_a==i_e[i])

	if (r_g==r_a and r_g==r_b):
		for i in range(K):
			new_g[i,:] = g[i,:] + i_g[i]*m[i,:]*(r_g==i_g[i])
	elif (r_g==r_a and r_g!=r_b):
		index = np.argmin(np.abs(i_g))
		new_g[index,:] = g[index,:] + i_g[index]*m[index,:]


	new_a = return_in_border(new_a)
	new_b = return_in_border(new_b)
	new_e = return_in_border(new_e)
	new_g = return_in_border(new_g)

	if (debug):
		print(str(new_a.tolist()))
		print(str(new_b.tolist()))

	return new_a, new_b, new_e, new_g, int(r_a == r_b)

#-------------------------------------------------

def bin_array(num, m):
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

#-------------------------------------------------

def find_changed_message(a, b):
	t_a = a.copy()
	t_b = b.copy()
	r = 0
	key = 0
	for i in range(2**(N*K)):
		key = bin_array(i, N*K)
		print(key.tolist())
		key = np.array(key).reshape((K,N))
		key = transform_message(key)
		t_a, t_b, r = change_coefficient(t_a, t_b, key)
		if (r == 1):
			break
	print("found key")
	print(key)
	debug = 1
	t_a, t_b, r = change_coefficient(t_a, t_b, key)
	debug = 0
	return 

##################################################

over_border = 0
p_i = np.zeros(border)
p_l = np.zeros(2*L+1)

steps = np.arange(0,max_steps)
bar = IncrementalBar('Done', max = max_steps)

#-------------------------------------------------

for k in steps:
	debug = 0
	# print(str(k)+" ", end="")
	# print("Initialize A and B")

	A = np.zeros((K,N))
	B = np.zeros((K,N))
	E = np.zeros((K,N))
	G = np.zeros((K,N))
	for i in range(N):
		for j in range(K):
			A[j][i] = random.randint(-L,L)
			B[j][i] = random.randint(-L,L)
			E[j][i] = random.randint(-L,L)
			G[j][i] = random.randint(-L,L)

	# print(find_changed_message(A, B))

	if (debug):
		print(str(A.tolist()))
		print(str(B.tolist()))
		print(str(E.tolist()))
		print(str(G.tolist()))

	i = 0
	s = 0
	r = 0

#-------------------------------------------------

	while (True):
		if (debug):
			print("")
			print(str(i))

		M = gen_message()
		M = transform_message(M)

		if (debug):
			print(str(M.tolist()))

		
		A, B, E, G, r = change_coefficient(A, B, E, G, M)

		# check how long we stuck (maybe its dont_change_loop)
		if (r == 0):
			s = s + 1
		else:
			s = 0

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
				break

		i = i + 1

		

		if ((A == B).all() and (A == E).all() and (A == G).all()):
			# print("done")
			break
		# if (i > 10000):
		# 	print("problem")
		# 	debug = 1
		# 	cont = int(input())
		# 	if (cont == 1):
		# 		print("fail")
		# 		break

#-------------------------------------------------

	if (i >= border):
		over_border = over_border + 1
	else:
		p_i[i] = p_i[i] + 1
	for t in range(N):
		for j in range(K):
			p_l[int(A[j][t])+L] = p_l[int(A[j][t])+L] + 1
	bar.next()

bar.finish()

#-------------------------------------------------

print("Over border: "+str(over_border)+" ("+str(over_border*100/border)+"%)")
print(p_l)

file_name_N = "./N/N-N"+str(N)+"-K"+str(K)+"-L"+str(L)+".csv"
file_name_S = "./S/S-N"+str(N)+"-K"+str(K)+"-L"+str(L)+".csv"
try:
	data_frame_N = pd.read_csv(file_name_N, header=None)
	data_frame_S = pd.read_csv(file_name_S, header=None)
	data_N = data_frame_N.to_numpy()
	data_S = data_frame_S.to_numpy()
	print(np.shape(data_N[:,0]))
	
	p_i = p_i + data_N[:,0]
	p_l = p_l + data_S[:,0]
except FileNotFoundError:
	print("nothing before")

points = 50
group = int(border/points)
y = np.zeros(points)

for i in range(points):
	temp = p_i[(i*group):((i+1)*group)]
	y[i] = np.sum(temp)

x = np.arange(0,border,group)
data_frame_N = pd.DataFrame(p_i)
data_frame_N.to_csv(file_name_N, header=None, index=None)

data_frame_S = pd.DataFrame(p_l)
data_frame_S.to_csv(file_name_S, header=None, index=None)

print(str(y)+" "+str(np.sum(y))+" series")
y = y / np.sum(y)

plt.bar(x, y, width=group)
plt.show()

x = np.arange(-L,L+1)

print(str(p_l)+" "+str(np.sum(p_l))+" = "+str(np.sum(p_l) / (L*N))+" series")
p_l = p_l / np.sum(p_l)

plt.bar(x, p_l, width=1)
plt.show()