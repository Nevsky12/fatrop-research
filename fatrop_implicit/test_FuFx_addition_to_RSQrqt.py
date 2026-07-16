import numpy as np

BAbt = np.array([
	[-3.19538, 0.731264 ],
	[-4.62425, 0.683719 ],
	[-0, 1 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	])
FuFx = np.array([
	[-0.235503, 0 ],
	[-3.31528, 0 ],
	])
FuFx_hessian = np.array([
	[0.0473608, 0.0508966, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0.0520477, 0.0358043, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	])
RSQrqt = np.array([
	[0.494564, 0 ],
	[0.800779, 1.44734 ],
	[21.6241, 34.798 ],
	])
nu = 0
nx = 2
nx_next = 1
RSQrqt_intermediate = np.array([
	[1.24709, 10.5936 ],
	[1.8898, 16.778 ],
	[21.6241, 34.798 ],
	])
RSQrqt_after = np.array([
	[1.99961, 11.6826 ],
	[12.4834, 32.1087 ],
	[21.6241, 34.798 ],
	])



RSQrqt = RSQrqt[:nu+nx+1, :nu+nx]
FuFx = FuFx[:nx+nu, :nx_next]
BAbt = BAbt[:nx+nu+1, :nx_next]

A = np.block([FuFx, BAbt[:nx+nu,:]])
B = np.block([[BAbt.T],
              [FuFx.T, np.zeros((nx_next, 1))]])

RSQrqt_intermediate_expected = RSQrqt + BAbt @ FuFx.T
print(np.linalg.norm(RSQrqt_intermediate - RSQrqt_intermediate_expected))
RSQrqt_final_expected = RSQrqt_intermediate + np.block([[FuFx @ BAbt[:nu+nx,:].T], [np.zeros((1, nu+nx))]])
print(np.linalg.norm(RSQrqt_after - RSQrqt_final_expected))
print(np.linalg.norm(RSQrqt_after - (RSQrqt + (A @ B).T)))
