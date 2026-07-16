import numpy as np

GuGx_tilde = np.array([
	[0.0384382 ],
	])
ukxk = np.transpose(np.array([[-1.68678 ]]))

l1 = -1
l2_expected = -1 + GuGx_tilde[:, :1].T @ ukxk
l2 = -1.065
print(f"l2_expected: {l2_expected}")
print(f"l2 computed: {l2}")

l3 = -1.261

l3_true = -1.68678154
l2_true = 0.844266 * l3_true
print(f"l2 true: {l2_true}")
print(f"true_term_added = {l2_true - l1}")
print(f"actual added: {l2 - l1}")