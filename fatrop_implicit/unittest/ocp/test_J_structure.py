from casadi import *
import numpy as np

def get_bycicle_model():
	xk = MX.sym('xk', 4) 
	uk = MX.sym('uk', 2)
    
	f = Function('f', [xk, uk], [vertcat(
		xk[2]*cos(xk[3]), 
		xk[2]*sin(xk[3]),
		uk[0],
		xk[2]*tan(uk[0]))])
	
	return f

def get_double_pendulum_model():
	xk = MX.sym('xk', 6) 
	uk = MX.sym('uk', 2)
	
	m1 = 1; m2 = 1; l1 = 1; l2 = 1
	c1 = cos(xk[0]); c2 = cos(xk[1])
	s1 = sin(xk[0]); s2 = sin(xk[1])
	g = 9.81

	M = MX(2, 2)
	M[0, 0] = (m1+m2)*l1**2 + m2*l2**2 + 2*m2*l1*l2*c2
	M[0, 1] = m2*l2**2 + m2*l1*l2*c2
	M[1, 0] = m2*l2**2 + m2*l1*l2*c2
	M[1, 1] = m2*l2**2

	C = MX(2, 2)
	C[0, 0] = 0
	C[0, 1] = -m2*l1*l2*s2*(2*xk[2]+xk[3])
	C[1, 0] = 0.5*m2*l1*l2*s2*(2*xk[2]+xk[3])
	C[1, 1] = -0.5*m2*l1*l2*s2*xk[2]

	rhs = MX(2, 1)
	rhs[0] = uk[0] - g*((m1+m2)*l1*s1 + m2*l2*sin(xk[0]+xk[1]))
	rhs[1] = uk[1] - g*m2*l2*sin(xk[0]+xk[1])

	f = Function('f', [xk, uk], [vertcat(
		xk[0:4],
		mtimes(inv(M), rhs - mtimes(C, vertcat(xk[2], xk[3]))))])
	
	return f

def get_backward_euler(f):
	xk = MX.sym('xk', f.size_in(0)[0])
	uk = MX.sym('uk', f.size_in(1)[0])
	xk_next = MX.sym('xk_next', f.size_out(0)[0])

	f_be = Function('f_be', [xk, uk, xk_next], 
		[xk + 0.1*f(xk_next, uk) - xk_next])
	
	return f_be

def get_J(f_d):
	xk = MX.sym('xk', f_d.size_in(0)[0])
	uk = MX.sym('uk', f_d.size_in(1)[0])
	xk_next = MX.sym('xk_next', f_d.size_in(2)[0])

	J = Function('J', [xk, uk, xk_next], [jacobian(f_d(xk, uk, xk_next), xk_next)])
	J_inv = Function('J_inv', [xk, uk, xk_next], [inv(J(xk,uk, xk_next))])
	
	return J, J_inv

def print_sparsities(*args):
	for _, arg in enumerate(args):
		sp = arg.sparsity_out(0)

		for i in range(sp.size(1)):
			row = ""
			for j in range(sp.size(2)):
				if sp.has_nz(i,j):
					row += "X "
				else:
					row += ". "
			print(row)
		print("\n")

def evaluate_randomly(*args):
	mtxs = []
	xk = np.random.rand(args[0].size_in(0)[0], 1)
	uk = np.random.rand(args[0].size_in(1)[0], 1)
	xk_next = np.random.rand(args[0].size_in(2)[0], 1)

	for _, arg in enumerate(args):
		# xk = np.random.rand(arg.size_in(0)[0], 1)
		# uk = np.random.rand(arg.size_in(1)[0], 1)
		# xk_next = np.random.rand(arg.size_in(2)[0], 1)

		result = arg(xk, uk, xk_next)
		mtxs.append(result)
		print(f"{result}\n")

	print(mtimes(mtxs[0], mtxs[1]))

J, J_inv = get_J(get_backward_euler(get_double_pendulum_model()))
# J, J_inv = get_J(get_backward_euler(get_bycicle_model()))
print_sparsities(J, J_inv)
evaluate_randomly(J, J_inv)