import numpy as np

GuGx_tilde = np.array([
	[0.0384382 ],
	])
FuFx_underbar_before = np.array([
	[0 ],
	])
Ggt_tilde = np.array([
	[-0, 0 ],
	[0.844266, 0 ],
	[0, 0 ],
	])
nu = 1
rank_k = 1
nunxm1 = 1
nx = 0
FuFx_underbar_after = np.array([
	[0 ],
	])
RSQrqt_underbar_before = np.array([
	[0.736071 ],
	[0 ],
	])
RSQrqt_underbar_after = np.array([
	[0.736071 ],
	[0.032452 ],
	])


Hx = Ggt_tilde[nu:nu+nx, :1].T
hx = Ggt_tilde[nu+nx:nu+nx+1, :1].T
print(f"Hx: {Hx}")
print("hx:", hx)
FuFx_underbar_expected = FuFx_underbar_before + GuGx_tilde[:, :rank_k] @ Hx
print(f"FuFx_underbar error: {np.linalg.norm(FuFx_underbar_after - FuFx_underbar_expected)}")

RSQrqt_underbar_expected = RSQrqt_underbar_before.copy()
RSQrqt_underbar_expected[nu+nx:nu+nx+1, :] = RSQrqt_underbar_expected[nu+nx:nu+nx+1, :] + GuGx_tilde[:, :rank_k] @ hx
print(f"RSQrqt_underbar error: {np.linalg.norm(RSQrqt_underbar_after - RSQrqt_underbar_expected)}")