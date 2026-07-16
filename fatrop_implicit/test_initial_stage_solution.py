import numpy as np

Ppt = np.array([
	[0.736071 ],
	[0.032452 ],
	])
Hh = np.array([
	[0.592845, 1 ],
	])



KKT = np.block([[Ppt[:-1,:].T, Hh[:, :-1].T],
                [Hh[:, :-1], np.zeros((Hh.shape[0], Hh.shape[0]))]])
rhs = np.array([Ppt[-1,:].T, Hh[:,-1].T])

print(f"KKT:\n{KKT}")
print(f"rhs:\n{rhs}")
solution = np.linalg.solve(KKT, -rhs)
print(f"solution:\n{solution}")