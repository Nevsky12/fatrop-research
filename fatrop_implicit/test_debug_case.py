import numpy as np

K = 2
nu = [0, 0]
nx = [1, 2]
ng_ineq = [0, 0]
ng_eq = [0, 0]
r = [1]
modified_K = 2
modified_nu = [0, 1]
modified_nx = [1, 1]
modified_ng_ineq = [0, 0]
modified_ng_eq = [1, 0]
RSQrqt = [
np.array([
	[0.736071 ],
	[0.000000 ],
	])
,
np.array([
	[0.236276, 0.511200 ],
	[0.511200, 1.106667 ],
	[2.000000, 1.000000 ],
	])
]
RSQrqt_original = [
np.array([
	[0.736071 ],
	[0.000000 ],
	])
,
np.array([
	[1.106667, 0.511200 ],
	[0.511200, 0.236276 ],
	[1.000000, 2.000000 ],
	])
]
FuFx = [
np.array([
	[0.000000 ],
	])
]
FuFx_original = [
np.array([
	[0.000000, 0.406084 ],
	])
]
GuGx = [
np.array([
	[0.406084 ],
	])
]
GuGx_original = [
np.array([
	[],
	])
]
Gg_eqt = [
np.array([
	[0.844266 ],
	[1.000000 ],
	])
,
np.array([
	[],
	[],
	[],
	])
]
Gg_eqt_original = [
np.array([
	[],
	[],
	])
,
np.array([
	[],
	[],
	[],
	])
]
Gg_ineqt = [
np.array([
	[],
	[],
	])
,
np.array([
	[],
	[],
	[],
	])
]
Gg_ineqt_original = [
np.array([
	[],
	[],
	])
,
np.array([
	[],
	[],
	[],
	])
]
BAbt = [
np.array([
	[0.592845 ],
	[0.000000 ],
	])
]
BAbt_original = [
np.array([
	[0.592845, 0.844266 ],
	[0.000000, 1.000000 ],
	])
]
Jt = [
np.array([
	[-1.000000, 0.000000 ],
	[0.000000, 0.000000 ],
	])
]
L = [
np.array([[1.000000, 0.000000],
[-0.000000, 1.000000],
])
]
U = [
np.array([[-1.000000, 0.000000],
[0.000000, 0.000000],
])
]
Pl = [
np.array([
	[1.000000, 0.000000 ],
	[0.000000, 1.000000 ],
	])
]
Pr = [
np.array([
	[1.000000, 0.000000 ],
	[0.000000, 1.000000 ],
	])
]



R1 = RSQrqt[1][:modified_nu[1], :modified_nu[1]]
S1 = RSQrqt[1][modified_nu[1]:modified_nu[1]+modified_nx[1], :modified_nu[1]]
Q1 = RSQrqt[1][modified_nu[1]:modified_nu[1]+modified_nx[1], modified_nu[1]:modified_nu[1]+modified_nx[1]]
G0x = GuGx[0]
F0x = FuFx[0]
Q0 = RSQrqt[0][modified_nu[0]:modified_nu[0]+modified_nx[0], modified_nu[0]:modified_nu[0]+modified_nx[0]]
H0u = Gg_eqt[0][:modified_nu[0]+modified_nx[0],:].T
A0 = BAbt[0][modified_nu[0]:modified_nu[0]+modified_nx[0], :].T

r1 = RSQrqt[1][modified_nu[1]+modified_nx[1]:, :modified_nu[1]].T
q1 = RSQrqt[1][modified_nu[1]+modified_nx[1]:, modified_nu[1]:modified_nu[1]+modified_nx[1]].T
b1 = BAbt[0][modified_nu[0]+modified_nx[0]:, :modified_nx[1]].T
q0 = RSQrqt[0][modified_nu[0]+modified_nx[0]:, modified_nu[0]:modified_nu[0]+modified_nx[0]].T
h0 = Gg_eqt[0][modified_nu[0]+modified_nx[0]:, :].T

### Original problem ###
KKT = np.block([
	[R1, S1.T, np.zeros((modified_nu[1], modified_nx[1])), G0x.T, np.zeros((modified_nu[1], modified_ng_eq[0]))],
    [S1, Q1, -np.eye(modified_nx[1], modified_nx[1]), F0x.T, np.zeros((modified_nx[1], modified_ng_eq[0]))],
    [np.zeros((modified_nx[1], modified_nu[1])), -np.eye(modified_nx[1], modified_nx[1]), np.zeros((modified_nx[1], modified_nx[1])), A0, np.zeros((modified_nx[1], modified_ng_eq[0]))],
	[G0x, F0x, A0.T, Q0, H0u.T],
	[np.zeros((modified_ng_eq[0], modified_nu[1]+2*modified_nx[1])), H0u, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1], [q1], [b1], [q0], [h0]])

solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u1 = solution[ptr:ptr+modified_nu[1]]; ptr += modified_nu[1]
x1 = solution[ptr:ptr+modified_nx[1]]; ptr += modified_nx[1]
pi1 = solution[ptr:ptr+modified_nx[1]]; ptr += modified_nx[1]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]

print(f"original solution:\nu1 = \n{u1}\nx1 = \n{x1}\npi1 = \n{pi1}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

lmbd = np.linalg.cholesky(R1)
assert np.linalg.norm(R1 - lmbd @ lmbd.T) < 1e-8
G0x_tilde = G0x @ np.linalg.inv(lmbd).T
r1_tilde = np.linalg.inv(lmbd) @ r1
S1_tilde = S1 @ np.linalg.inv(lmbd).T

Q1_tilde = Q1 - S1_tilde @ S1_tilde.T
q1_tilde = q1 - S1_tilde @ r1_tilde
F0x_tilde = F0x - G0x_tilde @ S1_tilde.T

Q0_prime = Q0 - G0x_tilde @ G0x_tilde.T
q0_prime = q0 - G0x_tilde @ r1_tilde

### Intermediate stage ###
KKT = np.block([
    [Q1_tilde, -np.eye(modified_nx[1], modified_nx[1]), F0x_tilde.T, np.zeros((modified_nx[1], modified_ng_eq[0]))],
    [-np.eye(modified_nx[1], modified_nx[1]), np.zeros((modified_nx[1], modified_nx[1])), A0, np.zeros((modified_nx[1], modified_ng_eq[0]))],
	[F0x_tilde, A0.T, Q0_prime, H0u.T],
	[np.zeros((modified_ng_eq[0], 2*modified_nx[1])), H0u, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[q1_tilde], [b1], [q0_prime], [h0]])
solution = np.linalg.solve(KKT, -rhs)
ptr = 0
x1 = solution[ptr:ptr+modified_nx[1]]; ptr += modified_nx[1]
pi1 = solution[ptr:ptr+modified_nx[1]]; ptr += modified_nx[1]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]
u1_tilde = -r1_tilde - S1_tilde.T @ x1 - G0x_tilde.T @ x0
u1 = np.linalg.inv(lmbd).T @ u1_tilde
print(f"intermediate solution:\nu1 = \n{u1}\nx1 = \n{x1}\npi1 = \n{pi1}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### Initial stage ###
Q0_tilde = Q0_prime + F0x_tilde.T @ A0 + A0.T @ F0x_tilde.T + A0.T @ Q1_tilde @ A0
q0_tilde = q0_prime + F0x_tilde.T @ b1 + A0.T @ q1_tilde + A0.T @ Q1_tilde @ b1

KKTI = np.block([
	[Q0_tilde, H0u.T],
	[H0u, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhsI = np.block([[q0_tilde], [h0]])
# print(f"\nsmall KKT:\n{KKTI}\n\nrhs:\n{rhsI}")
solutionI = np.linalg.solve(KKTI, -rhsI)

# print(f"\nP:\n{Q0_tilde}")
# print(f"p:\n{q0_tilde}")
# print(f"H0u:\n{H0u}")
# print(f"h0:\n{h0}")
x0 = solutionI[:modified_nx[0]]
lmbd0 = solutionI[modified_nx[0]:]
x1 = A0 @ x0 + b1
pi1 = Q1_tilde @ x1 + F0x_tilde.T @ x0 + q1_tilde
u1_tilde = -r1_tilde - S1_tilde.T @ x1 - G0x_tilde.T @ x0
u1 = np.linalg.inv(lmbd).T @ u1_tilde

print(f"initial solution:\nu1 = \n{u1}\nx1 = \n{x1}\npi1 = \n{pi1}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

print(f"P:\n{Q0_tilde}")
print(f"p:\n{q0_tilde}")
print(f"FuFx:\n{F0x_tilde}")
