import numpy as np

K = 2
nu = [0, 0]
nx = [1, 1]
ng_ineq = [0, 0]
ng_eq = [0, 1]
r = [0]
modified_K = 2
modified_nu = [0, 1]
modified_nx = [1, 0]
modified_ng_ineq = [0, 0]
modified_ng_eq = [1, 1]
RSQrqt = [
np.array([
	[0.736071 ],
	[0 ],
	])
,
np.array([
	[0.717836 ],
	[1 ],
	])
]
RSQrqt_original = [
np.array([
	[0.736071 ],
	[0 ],
	])
,
np.array([
	[0.717836 ],
	[1 ],
	])
]
FuFx = [
np.array([
	[],
	])
]
FuFx_original = [
np.array([
	[0.0384382 ],
	])
]
GuGx = [
np.array([
	[0.0384382 ],
	])
]
GuGx_original = [
np.array([
	[],
	])
]
Gg_eqt = [
np.array([
	[0.592845 ],
	[1 ],
	])
,
np.array([
	[0.844266 ],
	[0 ],
	])
]
Gg_eqt_original = [
np.array([
	[],
	[],
	])
,
np.array([
	[0.844266 ],
	[0 ],
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
	])
]
BAbt = [
np.array([
	[],
	[],
	])
]
BAbt_original = [
np.array([
	[0.592845 ],
	[1 ],
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
G0x = GuGx[0]
H1u = Gg_eqt[1][:modified_nu[1], :].T
Q0 = RSQrqt[0][modified_nu[0]:modified_nu[0]+modified_nx[0], modified_nu[0]:modified_nu[0]+modified_nx[0]]
H0x = Gg_eqt[0][modified_nu[0]:modified_nu[0]+modified_nx[0],:].T

r1 = RSQrqt[1][modified_nu[1]+modified_nx[1]:, :modified_nu[1]].T
h1 = Gg_eqt[1][modified_nu[1]+modified_nx[1]:, :].T
q0 = RSQrqt[0][modified_nu[0]+modified_nx[0]:, modified_nu[0]:modified_nu[0]+modified_nx[0]].T
h0 = Gg_eqt[0][modified_nu[0]+modified_nx[0]:, :].T

### Original problem ###
KKT = np.block([
	[R1, H1u.T, G0x.T, np.zeros((modified_nu[1], modified_ng_eq[0]))],
    [H1u, np.zeros((modified_ng_eq[1], modified_ng_eq[1] + modified_nx[0] + modified_ng_eq[0]))],
	[G0x, np.zeros((modified_nu[1], modified_ng_eq[1])), Q0, H0x.T],
	[np.zeros((modified_ng_eq[0], modified_nu[1]+modified_ng_eq[1])), H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1], [h1], [q0], [h0]])

solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u1 = solution[ptr:ptr+modified_nu[1]]; ptr += modified_nu[1]
lmbd1 = solution[ptr:ptr+modified_ng_eq[1]]; ptr += modified_ng_eq[1]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]

print(f"original solution:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### Factorization of H1u ###
Tl = np.array([[-H1u[0,0]]])
Tr = np.array([[1]])

Tli = np.linalg.inv(Tl)
Tri = np.linalg.inv(Tr)

R1_tilde = Tri.T @ R1 @ Tri
r1_tilde = Tri.T @ r1
G0x_tilde = G0x @ Tri
h1_tilde = Tli @ h1

KKT = np.block([
	[R1_tilde, -1, G0x_tilde.T, np.zeros((modified_nu[1], modified_ng_eq[0]))],
    [-1, np.zeros((modified_ng_eq[1], modified_ng_eq[1] + modified_nx[0] + modified_ng_eq[0]))],
	[G0x_tilde, np.zeros((modified_nu[1], modified_ng_eq[1])), Q0, H0x.T],
	[np.zeros((modified_ng_eq[0], modified_nu[1]+modified_ng_eq[1])), H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1_tilde], [h1_tilde], [q0], [h0]])

solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u1_tilde = solution[ptr:ptr+modified_nu[1]]; ptr += modified_nu[1]
lmbd1_tilde = solution[ptr:ptr+modified_ng_eq[1]]; ptr += modified_ng_eq[1]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]

u1 = Tri @ u1_tilde
lmbd1 = Tli.T @ lmbd1_tilde

print(f"solution after factorization of H1u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### elimination of u1 ###
P0 = Q0
p0 = q0 - G0x_tilde @ h1_tilde

KKT = np.block([
	[P0, H0x.T],
	[H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[p0], [h0]])

print(f"initial stage:")
print(f"KKT:\n{KKT}")
print(f"rhs:\n{rhs}")

solution = np.linalg.solve(KKT, -rhs)
x0 = solution[0]
lmbd0 = solution[1]
print(f"\ninitial stage solution:\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")

u1_tilde = h1_tilde
print()
print(G0x_tilde)
print()
print(x0)
print(f"lmbd1_tilde before:\n{r1_tilde + R1_tilde @ u1_tilde}")
lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0x_tilde.T @ x0
print(f"lmbd1_tilde after:\n{lmbd1_tilde}")
print(f"term added: {G0x_tilde.T @ x0}")

u1 = Tri @ u1_tilde
lmbd1 = Tli.T @ lmbd1_tilde

print(f"\nsolution after elimination of u1:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")
