from sympy import *
import numpy
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import os

# ========= Declarations ========

x = symbols('x')

alpha = 1
beta = 1

m = 15
f = (x**2)-x-2

n_pts = 1000

dir_name = "M" + str(m) + "/"

# ===============================

def op_a(alpha, beta, f1, f2):
	# Operador a(. , .)
	
	f1_x = diff(f1, x)
	f2_x = diff(f2, x)
	
	first_arg_expr = Mul(f1_x, f2_x)
	second_arg_expr = Mul(f1, f2)
	
	first_arg_func = lambdify(x, first_arg_expr, "numpy")
	second_arg_func = lambdify(x, second_arg_expr, "numpy")
	
	left_side = alpha * integrate.quad(first_arg_func, 0.0, 1.0)[0]
	right_side = beta * integrate.quad(second_arg_func, 0.0, 1.0)[0]
	
	return left_side + right_side
	
def op_parenthesis(f1, f2):
	# Operador (. , .)
	
	arg_expr = Mul(f1, f2)
	arg_func = lambdify(x, arg_expr, "numpy")
	
	return integrate.quad(arg_func, 0.0, 1.0)[0]
	
def generate_phis(discret):
	# Gera as funções phi no intervalo discreto
	
	phis = []
	m = len(discret)-1
	for i in range(1, m):
	
		prev_x_i = discret[i-1]
		x_i = discret[i]
		next_x_i = discret[i+1]
		
		hat_1 = (0, ((x >= 0.0) & (x < prev_x_i)))
		hat_2 = ((x - prev_x_i)/(x_i - prev_x_i), ((x >= prev_x_i) & (x < x_i)))
		hat_3 = ((next_x_i - x)/(next_x_i - x_i), ((x >= x_i) & (x < next_x_i)))
		hat_4 = (0, ((x >= next_x_i) & (x <= 1.0)))
		
		phi = Piecewise(hat_1, hat_2, hat_3, hat_4)
		phis.append(phi)
		
	return phis
	
def generate_regular_interval(n):

	interval = []
	for i in range(n+1):
		interval.append(i/n)
		
	return interval
	
def generate_irregular_interval(n):
	
	interval = [0]
	for i in range(1, n):
		interval.append(interval[i-1] + (1 - interval[i-1])/2 )
	
	interval.append(1)	
	return interval
	
def galerkin(alpha, beta, discret):

	lista_phi = generate_phis(discret)
	m = len(lista_phi)
	
	K = numpy.zeros( (m, m), dtype=float )
	for i in range(m):
		for j in range(m):
			K[i][j] = op_a(alpha, beta, lista_phi[i], lista_phi[j])
			print("K{0},{1}: {2}".format(i+1, j+1, K[i][j]))
	
	F = numpy.zeros( (m), dtype=float)
	for i in range(m):
		F[i] = op_parenthesis(f, lista_phi[i])
		print("F{0}: {1}".format(i+1, F[i]))
			
	C = numpy.linalg.solve(K, F)
	
	return K, F, C
	
def apply_solution(lista_cj, lista_phi):

	solucao = 0
	
	for j in range(len(lista_phi)):
		solucao += lista_cj[j] * lista_phi[j]
		
	return solucao
	
def plot_hats(num_lista_phi, m):

	plot_x = []
	plot_y = []

	for i in range(1, m):
		start = intervalo[i-1]
		end = intervalo[i+1]
		
		numerical_function = num_lista_phi[i-1]
		
		interval_unit = (end - start)/n_pts
		for	j in range(n_pts):
			x0 = start + j * interval_unit
			plot_x.append(x0)
					
			plot_y.append(numerical_function(x0))

	plt.scatter(plot_x, plot_y, s=1)
	
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
	plt.savefig(dir_name + "hats.png")
	plt.clf()
	
def plot_discrete(intervalo, m, n_pts, f, fig_name):
	
	total_pts = n_pts * m
	
	plot_x = list(map(lambda n: n/total_pts, range(total_pts)))	
	plot_y = list(map(f, plot_x))
	
	plt.scatter(plot_x, plot_y, s=1, c="royalblue")
	
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
	plt.savefig(dir_name + fig_name + ".png")
	plt.clf()
	
def check_result(K, F, C):
	
	return (K.dot(C)) - F
		
	
# ================== Chamadas ==========================

intervalo = generate_regular_interval(m)
print(intervalo)
#intervalo = generate_irregular_interval(m)

lista_phi = generate_phis(intervalo)
print(lista_phi)

numerical_phis = [n for n in map(lambdify, [x]*m, lista_phi)]
plot_hats(numerical_phis, m)

matriz_K, vetor_F, lista_cj = galerkin(alpha, beta, intervalo)
print(lista_cj)

literal_u_h = apply_solution(lista_cj, lista_phi)
print(literal_u_h)

numerical_u_h = lambdify(x, literal_u_h)
numerical_f = lambdify(x, f)

plot_discrete(intervalo, m, n_pts, numerical_f, "f")
plot_discrete(intervalo, m, n_pts, numerical_u_h, "u_h")

print(check_result(matriz_K, vetor_F, lista_cj))
