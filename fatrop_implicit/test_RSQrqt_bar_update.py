import numpy as np
RSQrq_hat = np.array([
	[0.0885268, 0 ],
	[2, 0 ],
	[0, 0 ],
	])
Llt = np.array([
	[0.297535 ],
	[6.72191 ],
	[0 ],
	])
GuGx_tilde = np.array([
	[0.0477665 ],
	[0.0812169 ],
	])
#-----------------------------------
#testing RSQrqt_bar update (also copy RSQrqt_hat, Llt and GuGxtilde definition)
nx = 0
nu = 1
nunxm1 = 2
rank_k = 0
RSQrqt_underbar_before = np.array([
	[1.45391, 0.860652 ],
	[0.860652, 0.536581 ],
	[0, 1 ],
	])
RSQrqt_tilde = np.array([
	[0.0885268, 0 ],
	[2, 0 ],
	[0, 0 ],
	])
v_r_tilde = np.array([
	[6.72191, 0 ],
	])
GuGx_hat = np.array([
	[0.160541 ],
	[0.272966 ],
	])
RSQrqt_underbar_intermediate = np.array([
	[1.42813, 0.81683 ],
	[0.81683, 0.46207 ],
	[0, 1 ],
	])
temp = np.array([
	[-1.07914, -1.83485 ],
	])
RSQrqt_underbar_after = np.array([
	[1.42813, 0.81683 ],
	[0.81683, 0.46207 ],
	[-1.07914, -0.834853 ],
	])
FuFx_underbar_before = np.array([
	[0 ],
	[0 ],
	])
Llt = np.array([
	[0.297535 ],
	[6.72191 ],
	[0 ],
	])
FuFx_underbar_after = np.array([
	[0 ],
	[0 ],
	])

RSQrq_hat = RSQrq_hat[:nu+nx+1, :nu+nx]
GuGx_tilde = GuGx_tilde[:nunxm1, :nu]
GuGx_hat = GuGx_hat[:nunxm1, :nu]
FuFx_underbar_before = FuFx_underbar_before[:nunxm1, :nx]
FuFx_underbar_after = FuFx_underbar_after[:nunxm1, :nx]

lmbd = Llt[:nu, :nu]
print(f"Lambda should be \n{lmbd}")
for i in range(nu):
    for j in range(i+1,nu):
        lmbd[i,j] = 0
R_hat = RSQrq_hat[:nu, :nu]
print(f"Check cholesky decomposition: {np.linalg.norm(R_hat - lmbd @ lmbd.T)}")

print(f"GuGx_tilde[:, rank_k:] is\n{GuGx_tilde[:, rank_k:]}")
print(f"np.linalg.inv(lmbd).T is\n{np.linalg.inv(lmbd).T}")
GuGx_hat_expected = GuGx_tilde[:, rank_k:] @ np.linalg.inv(lmbd).T
print(f"GuGx_hat should be\n{GuGx_hat_expected}")
print(f"GuGx_hat after:\n{GuGx_hat}")
print(f"GuGx_hat error: {np.linalg.norm(GuGx_hat - GuGx_hat_expected)}")

RSQrqt_underbar_expected = RSQrqt_underbar_before - \
	(GuGx_tilde[:, rank_k:] @ np.linalg.inv(lmbd).T @ np.linalg.inv(lmbd) @ np.block([[GuGx_tilde[:, rank_k:].T, RSQrqt_tilde[nu+nx:nu+nx+1, rank_k:nu].T]])).T
print(f"RSQrqt error: {np.linalg.norm(RSQrqt_underbar_after - RSQrqt_underbar_expected)}")

S_hat = RSQrq_hat[:nu, nu:nu+nx].T
L = np.linalg.inv(lmbd) @ S_hat.T
print(f"S_hat:\n{S_hat}")
print(f"L:\n{L}")
FuFx_expected = FuFx_underbar_before - GuGx_hat[:, rank_k:] @ L
print(f"FuFx before:\n{FuFx_underbar_before}")
print(f"FuFx expected:\n{FuFx_expected}")
print(f"FuFx after:\n{FuFx_underbar_after}")
print(f"FuFx error: {np.linalg.norm(FuFx_underbar_after - FuFx_expected)}")