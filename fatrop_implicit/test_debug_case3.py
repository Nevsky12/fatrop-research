import numpy as np

K = 2
nu = [1, 0]
nx = [1, 1]
ng_ineq = [0, 0]
ng_eq = [1, 1]
r = [0]
modified_K = 2
modified_nu = [1, 1]
modified_nx = [1, 0]
modified_ng_ineq = [0, 0]
modified_ng_eq = [2, 1]
RSQrqt = [
np.array([
	[0.236276, 0.102924 ],
	[0.102924, 0.0775578 ],
	[0, 1 ],
	])
,
np.array([
	[0.228164 ],
	[2 ],
	])
]
RSQrqt_original = [
np.array([
	[0.236276, 0.102924 ],
	[0.102924, 0.0775578 ],
	[0, 1 ],
	])
,
np.array([
	[0.228164 ],
	[2 ],
	])
]
FuFx = [
np.array([
	[],
	[],
	])
]
FuFx_original = [
np.array([
	[0.0392785 ],
	[0.0836079 ],
	])
]
GuGx = [
np.array([
	[0.0392785 ],
	[0.0836079 ],
	])
]
GuGx_original = [
np.array([
	[],
	[],
	])
]
Gg_eqt = [
np.array([
	[0.857946, 0.592845 ],
	[0.847252, 0.844266 ],
	[0, 2 ],
	])
,
np.array([
	[0.623564 ],
	[1 ],
	])
]
Gg_eqt_original = [
np.array([
	[0.857946 ],
	[0.847252 ],
	[0 ],
	])
,
np.array([
	[0.623564 ],
	[1 ],
	])
]
Gg_ineqt = [
np.array([
	[],
	[],
	[],
	])
,
np.array([
	[],
	[],
	])
]
Gg_ineqt_original = [
np.array([
	[],
	[],
	[],
	])
,
np.array([
	[],
	[],
	])
]
BAbt = [
np.array([
	[],
	[],
	[],
	])
]
BAbt_original = [
np.array([
	[0.592845 ],
	[0.844266 ],
	[2 ],
	])
]
Jt = [
np.array([
	[0 ],
	])
]
L = [
np.array([[1],
])
]
U = [
np.array([[0],
])
]
Pl = [
np.array([
	[1 ],
	])
]
Pr = [
np.array([
	[1 ],
	])
]

R1 = RSQrqt[1][:modified_nu[1], :modified_nu[1]]
G0u = GuGx[0][:modified_nu[0], :]
G0x = GuGx[0][modified_nu[0]:modified_nu[0]+modified_nx[0], :]
H1u = Gg_eqt[1][:modified_nu[1], :].T
R0 = RSQrqt[0][:modified_nu[0], :modified_nu[0]]
S0 = RSQrqt[0][:modified_nu[0], modified_nu[0]:modified_nu[0]+modified_nx[0]].T
Q0 = RSQrqt[0][modified_nu[0]:modified_nu[0]+modified_nx[0], modified_nu[0]:modified_nu[0]+modified_nx[0]]
H0u = Gg_eqt[0][:modified_nu[0],:].T
H0x = Gg_eqt[0][modified_nu[0]:modified_nu[0]+modified_nx[0],:].T

r1 = RSQrqt[1][modified_nu[1]+modified_nx[1]:, :modified_nu[1]].T
h1 = Gg_eqt[1][modified_nu[1]+modified_nx[1]:, :].T
r0 = RSQrqt[0][modified_nu[0]+modified_nx[0]:, :modified_nu[0]].T
q0 = RSQrqt[0][modified_nu[0]+modified_nx[0]:, modified_nu[0]:modified_nu[0]+modified_nx[0]].T
h0 = Gg_eqt[0][modified_nu[0]+modified_nx[0]:, :].T

### Original problem ###
KKT = np.block([
	[R1, H1u.T, G0u.T, G0x.T, np.zeros((modified_nu[1], modified_ng_eq[0]))],
    [H1u, np.zeros((modified_ng_eq[1], modified_ng_eq[1] + modified_nu[0] + modified_nx[0] + modified_ng_eq[0]))],
	[G0u, np.zeros((modified_nu[1], modified_ng_eq[1])), R0, S0.T, H0u.T],
	[G0x, np.zeros((modified_nu[1], modified_ng_eq[1])), S0, Q0, H0x.T],
    [np.zeros((modified_ng_eq[0], modified_nu[1]+modified_ng_eq[1])), H0u, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1], [h1], [r0], [q0], [h0]])

solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u1 = solution[ptr:ptr+modified_nu[1]]; ptr += modified_nu[1]
lmbd1 = solution[ptr:ptr+modified_ng_eq[1]]; ptr += modified_ng_eq[1]
u0 = solution[ptr:ptr+modified_nu[0]]; ptr += modified_nu[0]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]

print(f"original solution:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")
### Factorization of H1u ###
Tl = np.array([[-H1u[0,0]]])
Tr = np.array([[1]])

Tli = np.linalg.inv(Tl)
Tri = np.linalg.inv(Tr)

R1_tilde = Tri.T @ R1 @ Tri
r1_tilde = Tri.T @ r1
G0x_tilde = G0x @ Tri
G0u_tilde = G0u @ Tri
h1_tilde = Tli @ h1

KKT = np.block([
	[R1_tilde, -1, G0u_tilde.T, G0x_tilde.T, np.zeros((modified_nu[1], modified_ng_eq[0]))],
    [-1, np.zeros((modified_ng_eq[1], modified_ng_eq[1] + modified_nu[0] + modified_nx[0] + modified_ng_eq[0]))],
	[G0u_tilde, np.zeros((modified_nu[1], modified_ng_eq[1])), R0, S0.T, H0u.T],
	[G0x_tilde, np.zeros((modified_nu[1], modified_ng_eq[1])), S0, Q0, H0x.T],
    [np.zeros((modified_ng_eq[0], modified_nu[1]+modified_ng_eq[1])), H0u, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1_tilde], [h1_tilde], [r0], [q0], [h0]])

solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u1_tilde = solution[ptr:ptr+modified_nu[1]]; ptr += modified_nu[1]
lmbd1_tilde = solution[ptr:ptr+modified_ng_eq[1]]; ptr += modified_ng_eq[1]
u0 = solution[ptr:ptr+modified_nu[0]]; ptr += modified_nu[0]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]

u1 = Tri @ u1_tilde
lmbd1 = Tli.T @ lmbd1_tilde

print(f"solution after factorization of H1u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### elimination of u1 ###
r0_prime = r0 + G0u_tilde @ h1_tilde
q0_prime = q0 + G0x_tilde @ h1_tilde
print(f"q0_prime = {q0_prime} + {G0x_tilde} @ {h1_tilde}")

KKT = np.block([
    [R0, S0.T, H0u.T],
    [S0, Q0, H0x.T],
    [H0u, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r0_prime], [q0_prime], [h0]])

solution = np.linalg.solve(KKT, -rhs)
u0 = solution[:modified_nu[0]]
x0 = solution[modified_nu[0]:modified_nu[0]+modified_nx[0]]
lmbd0 = solution[modified_nu[0]+modified_nx[0]:]
u1_tilde = h1_tilde
lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
u1 = Tri @ u1_tilde
lmbd1 = Tli.T @ lmbd1_tilde

print(f"solution after elimination of u1:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### Factorization of H0u ###
L1 = 1
L2 = 0.691005028288494
U1 = 0.857946
L = np.array([[L1, 0], [L2, 1]])
U = np.array([[U1], [0]])

Tl0 = L @ np.array([[-U1, 0], [0, 1]])
Tr0 = np.eye(1)
Tl0i = np.linalg.inv(Tl0)
Tr0i = np.linalg.inv(Tr0)
assert(np.linalg.norm(H0u - Tl0 @ np.array([[-1], [0]]) @ Tr0) < 1e-8)

R0_tilde = Tr0i.T @ R0 @ Tr0i
r0_tilde = Tr0i.T @ r0_prime
S0_tilde = S0 @ Tr0i
H0x_tilde = Tl0i @ H0x
h0_tilde = Tl0i @ h0

KKT = np.block([
    [R0_tilde, S0_tilde.T, np.array([[-1, 0]])],
    [S0_tilde, Q0, H0x_tilde.T], 
    [np.array([[-1, 0]]).T, H0x_tilde, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))]
])
rhs = np.block([[r0_tilde], [q0_prime], [h0_tilde]])

solution = np.linalg.solve(KKT, -rhs)
u0_tilde = solution[:modified_nu[0]]
x0 = solution[modified_nu[0]:modified_nu[0]+modified_nx[0]]
lmbd0_tilde = solution[modified_nu[0]+modified_nx[0]:]
print(f"lmbd0_tilde: {lmbd0_tilde}")

u0 = Tr0i @ u0_tilde
lmbd0 = Tl0i.T @ lmbd0_tilde
u1_tilde = h1_tilde
lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
u1 = Tri @ u1_tilde
lmbd1 = Tli.T @ lmbd1_tilde

print(f"solution after factorization of H0u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")
print(f"q0_prime should be:\n{q0_prime}\n")
H0xa = H0x_tilde[:1,:]
h0xa = h0_tilde[:1,:]
P0 = Q0 + S0_tilde @ H0xa + H0xa.T @ S0_tilde.T + H0xa.T @ R0_tilde @ H0xa
p0 = q0_prime + S0_tilde @ h0xa + H0xa.T @ R0_tilde @ h0xa + H0xa.T @ r0_tilde

H0_hat = H0x_tilde[1:,:]
h0_hat = h0_tilde[1:,:]

KKT = np.block([
    [P0, H0_hat.T],
    [H0_hat, np.zeros((modified_ng_eq[0]-1, modified_ng_eq[0]-1))],
])
rhs = np.block([[p0], [h0_hat]])
solution = np.linalg.solve(KKT, -rhs)
x0 = solution[0]
v0 = solution[1:]
u0_tilde = h0_tilde[:1,:] + H0x_tilde[:1,:] @ x0
u0 = Tr0i @ u0_tilde
lmbd0_tilde = R0_tilde * u0_tilde + S0_tilde.T @ x0 + r0_tilde
lmbd0 = Tl0i.T @ np.array([[lmbd0_tilde[0,0]],[v0[0,0]]])

u1_tilde = h1_tilde
lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
u1 = Tri @ u1_tilde
lmbd1 = Tli.T @ lmbd1_tilde

print(f"solution after factorization of H0u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

print(f"initial stage should be:\n{KKT}\n{rhs}")

# ### Factorization of R0 ###
# L0 = np.array([[np.sqrt(R0[0,0])]])
# L0i = np.linalg.inv(L0)
# assert(np.linalg.norm(R0 - L0 @ L0.T) < 1e-8)

# r0_tilde = L0i @ r0_prime
# S0_tilde = S0 @ L0i.T
# H0u_tilde = H0u @ L0i.T

# KKT = np.block([
#     [np.eye(modified_nu[0]), S0_tilde.T, H0u_tilde.T],
# 	[S0_tilde, Q0, H0x.T],
# 	[H0u_tilde, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
# ])
# rhs = np.block([[r0_tilde], [q0_prime], [h0]])

# solution = np.linalg.solve(KKT, -rhs)
# u0_tilde = solution[:modified_nu[0]]
# x0 = solution[modified_nu[0]:modified_nu[0]+modified_nx[0]]
# lmbd0 = solution[modified_nu[0]+modified_nx[0]:]
# u0 = L0i.T @ u0_tilde

# u1_tilde = h1_tilde
# lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
# u1 = Tri @ u1_tilde
# lmbd1 = Tli.T @ lmbd1_tilde

# print(f"solution after factorization of R0:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
# print(f"=============================================================")

# ### elimination of u0 ###
# P0 = Q0 - S0_tilde @ S0_tilde.T
# p0 = q0_prime - S0_tilde @ r0_tilde
# H0x_tilde = H0x - H0u_tilde @ S0_tilde.T
# h0_tilde = h0 - H0u_tilde @ r0_tilde

# KKT = np.block([
#     [P0, H0x_tilde.T],
# 	[H0x_tilde, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
# ])
# print(KKT)
# rhs = np.block([[p0], [h0_tilde]])

# solution = np.linalg.solve(KKT, -rhs)
# x0 = solution[:modified_nx[0]]
# lmbd0 = solution[modified_nx[0]:]

# u0_tilde = -(r0_tilde + S0_tilde.T@x0 + H0u_tilde.T @ lmbd0)
# u0 = L0i.T @ u0_tilde

# u1_tilde = h1_tilde
# lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
# u1 = Tri @ u1_tilde
# lmbd1 = Tli.T @ lmbd1_tilde

# print(f"solution after elimination of u0:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
# print(f"=============================================================")