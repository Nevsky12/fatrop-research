import numpy as np

GuGx = np.array([
	[0.0477665, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0.0812169, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	])
GuGx_hat = np.array([
	[0.160541 ],
	[0.272966 ],
	])
Llt = np.array([
	[0.297535 ],
	[6.72191 ],
	[0 ],
	])
nunxm1 = 2
nu = 1
rho_k = 0
ukxk = np.transpose(np.array([[0.968733, -0.680246 ]]))
uk_before = np.transpose(np.array([[-22.592 ]]))
GuGx_hat_intermediate = np.array([
	[0.539571 ],
	[0.917427 ],
	])
uk_after = np.transpose(np.array([[-22.592 ]]))

# assuming Tr is identity
GuGx = GuGx[:nunxm1, :nu]
lmbd = Llt[:nu-rho_k, :nu-rho_k]
GuGx_hat_expected = GuGx @ np.linalg.inv(lmbd).T
print(f"GuGx_hat error: {np.linalg.norm(GuGx_hat - GuGx_hat_expected)}")
# uk_expected = uk_before - np.linalg.inv(lmbd).T @ GuGx_hat[:, rho_k:nu].T @ ukxk
GuGx_hat_intermediate_expected = GuGx_hat[:, rho_k:nu] @ np.linalg.inv(lmbd)
print(f"GuGx_hat intermediate error: {np.linalg.norm(GuGx_hat_intermediate - GuGx_hat_intermediate_expected)}")
print(f"GuGx_hat intermediate:\n{GuGx_hat_intermediate}\nGuGx_hat intermediate expected:\n{GuGx_hat_intermediate_expected}")

uk_expected = uk_before - GuGx_hat_intermediate_expected.T @ ukxk
print(f"Check uk update: {np.linalg.norm(uk_after - uk_expected)}")