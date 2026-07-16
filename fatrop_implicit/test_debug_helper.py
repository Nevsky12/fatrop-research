import numpy as np

def GetBlockMatrices(K, modified_nu, modified_nx, modified_ng_eq, RSQrqt, GuGx, FuFx, Gg_eqt, BAbt):
    R = []
    S = []
    Q = []
    Gu = []
    Gx = []
    Fu = []
    Fx = []
    Hu = []
    Hx = []
    B = []
    A = []
    r = []
    q = []
    h = []
    b = []
    for k in range(K):
        R.append(RSQrqt[k][:modified_nu[k], :modified_nu[k]])
        S.append(RSQrqt[k][:modified_nu[k], modified_nu[k]:modified_nu[k]+modified_nx[k]].T)
        Q.append(RSQrqt[k][modified_nu[k]:modified_nu[k]+modified_nx[k], modified_nu[k]:modified_nu[k]+modified_nx[k]])
        if k < K-1:
            Gu.append(GuGx[k][:modified_nu[k], :])
            Gx.append(GuGx[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :])
            Fu.append(FuFx[k][:modified_nu[k], :])
            Fx.append(FuFx[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :])
            B.append(BAbt[k][:modified_nu[k], :modified_nx[k+1]].T)
            A.append(BAbt[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :modified_nx[k+1]].T)
            b.append(BAbt[k][modified_nu[k]+modified_nx[k]:, :modified_nx[k+1]].T)

            if b[k].shape[0] == 0 or b[k].shape[1] == 0:
                b[k] = np.zeros((0,1))
        Hu.append(Gg_eqt[k][:modified_nu[k], :modified_ng_eq[k]].T)
        Hx.append(Gg_eqt[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :modified_ng_eq[k]].T)
        
        r.append(RSQrqt[k][modified_nu[k]+modified_nx[k]:modified_nu[k]+modified_nx[k]+1, :modified_nu[k]].T)
        q.append(RSQrqt[k][modified_nu[k]+modified_nx[k]:modified_nu[k]+modified_nx[k]+1, modified_nu[k]:modified_nu[k]+modified_nx[k]].T)
        h.append(Gg_eqt[k][modified_nu[k]+modified_nx[k]:modified_nu[k]+modified_nx[k]+1, :modified_ng_eq[k]].T)

        if r[k].shape[0] == 0 or r[k].shape[1] == 0:
            r[k] = np.zeros((0,1))
        if q[k].shape[0] == 0 or q[k].shape[1] == 0:
            q[k] = np.zeros((0,1))
        if h[k].shape[0] == 0 or h[k].shape[1] == 0:
            h[k] = np.zeros((0,1))

    blocks = {
        "R": R, "S": S, "Q": Q, "Gu": Gu, "Gx": Gx, "Fu": Fu, "Fx": Fx,
        "Hu": Hu, "Hx": Hx, "B": B, "A": A, "r": r, "q": q, "h": h, "b": b
    }
    return blocks

def GetKKT(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b, Jt=None):
    number_of_primal_vars = sum(nu) + sum(nx)
    number_of_eqs = sum(ng_eq) + sum(nx[1:])
    KKT = np.zeros((number_of_primal_vars + number_of_eqs, number_of_primal_vars + number_of_eqs))
    rhs = np.zeros((number_of_primal_vars + number_of_eqs, 1))
    ptr = 0
    for k in range(K-1, -1, -1):
        KKT[ptr:ptr+nu[k]+nx[k], ptr:ptr+nu[k]+nx[k]] = np.block([[R[k], S[k].T], [S[k], Q[k]]])
        rhs[ptr:ptr+nu[k]+nx[k]] = np.block([[r[k]], [q[k]]])
        
        Gg_eqt = np.block([[Hu[k], Hx[k]]]).T
        KKT[ptr:ptr+nu[k]+nx[k], ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]] = Gg_eqt[:nu[k]+nx[k], :]
        KKT[ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k], ptr:ptr+nu[k]+nx[k]] = Gg_eqt[:nu[k]+nx[k], :].T
        rhs[ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]] = h[k]
        
        if k > 0:
            # BAbt
            BAbt = np.block([[B[k-1], A[k-1]]]).T
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], 
                ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = BAbt[:nu[k-1]+nx[k-1]]
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], 
                ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = BAbt[:nu[k-1]+nx[k-1]].T
            rhs[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = b[k-1]

            # J
            if Jt is None:
                KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k],
                    ptr+nu[k]:ptr+nu[k]+nx[k]] = -np.eye(nx[k])
                KKT[ptr+nu[k]:ptr+nu[k]+nx[k],
                    ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = -np.eye(nx[k])
            else:
                KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k],
                    ptr+nu[k]:ptr+nu[k]+nx[k]] = Jt[k-1]
                KKT[ptr+nu[k]:ptr+nu[k]+nx[k],
                    ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = Jt[k-1].T
            
            # FuFx
            FuFx = np.block([[Fu[k-1]], [Fx[k-1]]])
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], 
                ptr+nu[k]:ptr+nu[k]+nx[k]] = FuFx[:nx[k-1]+nu[k-1], :nx[k]]
            KKT[ptr+nu[k]:ptr+nu[k]+nx[k], 
                ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = FuFx[:nx[k-1]+nu[k-1], :nx[k]].T
            
            # GuGx
            GuGx = np.block([[Gu[k-1]], [Gx[k-1]]])
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], 
                ptr:ptr+nu[k]] = GuGx[:nx[k-1]+nu[k-1], :nu[k]]
            KKT[ptr:ptr+nu[k], 
                ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = GuGx[:nx[k-1]+nu[k-1], :nu[k]].T
            

            ptr += nu[k] + nx[k] + ng_eq[k] + nx[k]

    return KKT, rhs

def print_KKT(KKT, rhs):
    print(f"KKT:")
    for row in KKT:
        for elem in row:
            if elem == 0:
                print(f"{elem:10.0f} ", end="")
            else:
                print(f"{elem:10.6f} ", end="")
        print()
    print(rhs.T)

def extract_solultion(K, nu, nx, ng_eq, sol):
    extracted_solution = {"u": [], "x": [], "lambda": [], "pi": []}

    ptr = 0
    for k in range(K-1, -1, -1):
        extracted_solution["u"].append(sol[ptr:ptr+nu[k], :])
        extracted_solution["x"].append(sol[ptr+nu[k]:ptr+nu[k]+nx[k], :])
        extracted_solution["lambda"].append(sol[ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k], :])
        extracted_solution["pi"].append(sol[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+(nx[k] if k > 0 else 0), :])
        ptr += nu[k] + nx[k] + ng_eq[k] + (nx[k] if k > 0 else 0)

    extracted_solution["u"] = extracted_solution["u"][::-1]
    extracted_solution["x"] = extracted_solution["x"][::-1]
    extracted_solution["lambda"] = extracted_solution["lambda"][::-1]
    extracted_solution["pi"] = extracted_solution["pi"][::-1]

    return extracted_solution

def print_solution(solution):
    print(f"u:")
    for k in range(len(solution["u"])):
        print(f"\t{solution['u'][k].T}")
    print(f"x:")
    for k in range(len(solution["x"])):
        print(f"\t{solution['x'][k].T}")
    print(f"lambda:")
    for k in range(len(solution["lambda"])):
        print(f"\t{solution['lambda'][k].T}")
    print(f"pi:")
    for k in range(len(solution["pi"])):
        print(f"\t{solution['pi'][k].T}")
    print()

def Solve(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b,
          Pl, Pr, L, U, Lmbds, rank_k_values, Hut, max_recursion_depth=-1, **kwargs):
    if max_recursion_depth == 0:
        print("Maximum recursion depth reached. Solving linear system.")
        KKT, rhs = GetKKT(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b)
        solution_vector = np.linalg.solve(KKT, -rhs)
        solution = extract_solultion(K, nu, nx, ng_eq, solution_vector)
        if kwargs.get("store_linear_systems", False):
            if "linear_systems" not in kwargs:
                kwargs["linear_systems"] = []
            kwargs["linear_systems"].append({
                "KKT": KKT,
                "rhs": rhs,
            })
            return solution, kwargs["linear_systems"]
        else:
            return solution

    # figure out what step to perform
    nu_c = nu.copy(); nx_c = nx.copy(); ng_eq_c = ng_eq.copy(); 
    R_c = R.copy(); S_c = S.copy(); Q_c = Q.copy(); 
    Gu_c = Gu.copy(); Gx_c = Gx.copy(); Fu_c = Fu.copy(); 
    Fx_c = Fx.copy(); Hu_c = Hu.copy(); Hx_c = Hx.copy(); 
    B_c = B.copy(); A_c = A.copy(); r_c = r.copy(); 
    q_c = q.copy(); h_c = h.copy(); b_c = b.copy()

    if nu[K-1] != 0:
        print(f"\n\nStage {K-1}: eliminating controls")
        # print(f"checking Hu dimension")
        # for k in range(K):
        #     assert Hu[k].shape == Hut[k].T.shape, f"Hu[{k}] shape {Hu[k].shape} does not match Hut[{k}].T shape {Hut[k].T.shape}"
        # decompose Hku and eliminate constraints partially
        rank = rank_k_values[K-1]; m = ng_eq[K-1]; n = nu[K-1]
        U1 = U[K-1][:rank, :rank]
        U2 = U[K-1][:rank, rank:]

        Tl = Pl[K-1] @ L[K-1] @ np.block([[-U1, np.zeros((rank, m-rank))], [np.zeros((m-rank, rank)), np.eye(m-rank)]])
        Tr = np.block([[np.eye(rank), np.linalg.inv(U1) @ U2], [np.zeros((n-rank, rank)), np.eye(n-rank)]]) @ Pr[K-1].T
        norm = np.linalg.norm(Hu[K-1] - Tl @ np.block([[-np.eye(rank), np.zeros((rank, n-rank))], [np.zeros((m-rank, n))]]) @ Tr)
        print(norm)
        print(np.linalg.norm(Hu[K-1] - Hut[K-1].T))
        assert(norm < 1e-5)

        Tli = np.linalg.inv(Tl)
        Tri = np.linalg.inv(Tr)

        # scale
        R_tilde = Tri.T @ R[K-1] @ Tri
        r_tilde = Tri.T @ r[K-1]
        S_tilde = S[K-1] @ Tri
        Hx_tilde = Tli @ Hx[K-1]
        h0_tilde = Tli @ h[K-1]

        Hu_c[K-1] = np.block([[-np.eye(rank), np.zeros((rank, n-rank))], [np.zeros((m-rank, n))]])
        R_c[K-1] = R_tilde
        S_c[K-1] = S_tilde
        Q_c[K-1] = Q[K-1]
        Hx_c[K-1] = Hx_tilde
        r_c[K-1] = r_tilde
        h_c[K-1] = h0_tilde
        if K > 1:
            Gu_c[K-2] = Gu[K-2] @ Tri
            Gx_c[K-2] = Gx[K-2] @ Tri

        ### intermediate check ###
        # Hu_c[K-1] = np.block([[-np.eye(rank), np.zeros((rank, n-rank))], [np.zeros((m-rank, n))]])
        # KKT, rhs = GetKKT(K, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c, Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c)
        # solution_vector = np.linalg.solve(KKT, -rhs)
        # solution = extract_solultion(K, nu_c, nx_c, ng_eq_c, solution_vector)
        # u_tilde = solution["u"][K-1]
        # u = Tri @ u_tilde
        # lmbd_tilde = solution["lambda"][K-1]
        # lmbd = Tli.T @ lmbd_tilde
        # solution["u"][K-1] = u
        # solution["lambda"][K-1] = lmbd
        # return solution
        ######

        # eliminate u1a and lambda
        Sa = S_tilde[:, :rank]; Sb = S_tilde[:, rank:]
        Ra = R_tilde[:rank, :rank]; Rb = R_tilde[rank:, rank:]
        ra = r_tilde[:rank,:]; rb = r_tilde[rank:,:]
        Ha = Hx_tilde[:rank, :]; Hb = Hx_tilde[rank:, :]
        ha = h0_tilde[:rank,:]; hb = h0_tilde[rank:,:]

        Q_tilde = Q[K-1] + Ha.T @ Sa.T + Sa @ Ha + Ha.T @ Ra @ Ha
        q_tilde = q[K-1] + Ha.T @ ra + Sa @ ha + Ha.T @ Ra @ ha

        if K > 1:
            Gua = Gu_c[K-2][:, :rank]; Gub = Gu_c[K-2][:, rank:]
            Gxa = Gx_c[K-2][:, :rank]; Gxb = Gx_c[K-2][:, rank:]
            Fu_c[K-2] = Fu[K-2] + Gua @ Ha
            Fx_c[K-2] = Fx[K-2] + Gxa @ Ha
            r_c[K-2] = r[K-2] + Gua @ ha
            q_c[K-2] = q[K-2] + Gxa @ ha

            Gu_c[K-2] = Gub
            Gx_c[K-2] = Gxb
        
        Sb_prime = Sb + Ha.T @ R_tilde[:rank, rank:]
        rb_prime = rb + R_tilde[rank:, :rank] @ ha
        
        R_c[K-1] = Rb
        S_c[K-1] = Sb_prime
        Q_c[K-1] = Q_tilde
        Hu_c[K-1] = np.zeros((m-rank, n))
        Hx_c[K-1] = Hb
        r_c[K-1] = rb_prime
        q_c[K-1] = q_tilde
        h_c[K-1] = hb
        nu_c[K-1] = n - rank
        ng_eq_c[K-1] = m - rank
        if ng_eq_c[K-1] < 0:
            print(f"Error: negative number of equality constraints at stage {K-1}")
            print(f"m: {m}, rank: {rank}")
            raise ValueError("Negative number of equality constraints")

        ### intermediate check ###
        # KKT, rhs = GetKKT(K, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c, Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c)
        # solution_vector = np.linalg.solve(KKT, -rhs)
        # solution = extract_solultion(K, nu_c, nx_c, ng_eq_c, solution_vector)
        # # recover solution
        # ub_tilde = solution["u"][K-1]
        # ua_tilde = Ha @ solution["x"][K-1] + ha
        # u_tilde = np.block([[ua_tilde], [ub_tilde]])
        # u = Tri @ u_tilde
        # lmbd_tilde = (R_tilde @ u_tilde + S_tilde.T @ solution["x"][K-1] + r_tilde)[:rank]
        # if K > 1:
        #     lmbd_tilde += Gua.T @ solution["u"][K-2] + Gxa.T @ solution["x"][K-2]
        # lmbd = Tli.T @ lmbd_tilde
        # solution["u"][K-1] = u
        # solution["lambda"][K-1] = lmbd
        # return solution
        ######

        # # eliminate remaining controls
        Lmbd = np.linalg.cholesky(Rb)
        norm = np.linalg.norm(Lmbd - Lmbds[K-1])
        print(norm)
        print(np.linalg.norm(Rb - Lmbds[K-1] @ Lmbds[K-1].T))
        # print(f"Lmbd:\n{Lmbd}")
        # print(f"Lmbds[K-1]:\n{Lmbds[K-1]}")
        # assert norm < 1e-5
        norm = np.linalg.norm(Rb - Lmbd @ Lmbd.T)
        print(norm)
        assert norm < 1e-4
        # TODO: check if this matches the Llt result

        Lmbdi = np.linalg.inv(Lmbd)
        
        S_hat = Sb_prime @ Lmbdi.T
        r_hat = Lmbdi @ rb_prime

        Q_hat = Q_tilde - S_hat @ S_hat.T
        q_hat = q_tilde - S_hat @ r_hat

        if K > 1:
            Gu_hat = Gub @ Lmbdi.T
            Gx_hat = Gxb @ Lmbdi.T
            Fu_c[K-2] = Fu_c[K-2] - Gu_hat @ S_hat.T
            Fx_c[K-2] = Fx_c[K-2] - Gx_hat @ S_hat.T
            R_c[K-2] = R_c[K-2] - Gu_hat @ Gu_hat.T
            S_c[K-2] = S_c[K-2] - Gx_hat @ Gu_hat.T
            Q_c[K-2] = Q_c[K-2] - Gx_hat @ Gx_hat.T
            r_c[K-2] = r_c[K-2] - Gu_hat @ r_hat
            q_c[K-2] = q_c[K-2] - Gx_hat @ r_hat
            Gx_c[K-2] = np.zeros((nx[K-2],0))
            Gu_c[K-2] = np.zeros((nu[K-2],0))

        nu_c[K-1] = 0

        R_c[K-1] = np.zeros((0,0))
        S_c[K-1] = np.zeros((nx[K-1],0))
        Q_c[K-1] = Q_hat
        r_c[K-1] = np.zeros((0,1))
        q_c[K-1] = q_hat

        # perform recursive call
        solution = Solve(K, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c,
              Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c, Pl, Pr, L, U,
              Lmbds, rank_k_values, Hut, max_recursion_depth=max_recursion_depth-1, **kwargs)
        if kwargs.get("store_linear_systems", True):
            solution, linear_systems = solution
            kwargs["linear_systems"] = linear_systems

        # verify solution dimensions
        for k in range(K-1):
            assert solution["u"][k].shape == (nu_c[k], 1), f"solution['u'][{k}] has shape {solution['u'][k].shape}, expected {(nu_c[k], 1)}"
            assert solution["x"][k].shape == (nx_c[k], 1), f"solution['x'][{k}] has shape {solution['x'][k].shape}, expected {(nx_c[k], 1)}"
            assert solution["lambda"][k].shape == (ng_eq_c[k], 1), f"solution['lambda'][{k}] has shape {solution['lambda'][k].shape}, expected {(ng_eq_c[k], 1)}"
        
        # recover original solution
        u_b_hat = S_hat.T @ solution["x"][K-1] + r_hat
        if K > 1:
            u_b_hat += Gu_hat.T @ solution["u"][K-2] + Gx_hat.T @ solution["x"][K-2]
        u_b_tilde = - Lmbdi.T  @ u_b_hat
        u_tilde = np.block([[Ha @ solution["x"][K-1] + ha], [u_b_tilde]])
        u = Tri @ u_tilde
        lmbd_tilde = (R_tilde @ u_tilde + S_tilde.T @ solution["x"][K-1] + r_tilde)[:rank]
        if K > 1:
            lmbd_tilde += Gua.T @ solution["u"][K-2] + Gxa.T @ solution["x"][K-2]
        lmbd_tilde = np.block([[lmbd_tilde], [solution["lambda"][K-1]]])
        lmbd = Tli.T @ lmbd_tilde

        solution["u"][K-1] = u
        solution["lambda"][K-1] = lmbd[:ng_eq[K-1], :]

    elif K > 1:
        print(f"\n\nStage {K-1}: eliminating states")
        # eliminate states
        temp = np.block([[np.block([[B[K-2], A[K-2]]]).T, np.eye(nu[K-2]+nx[K-2])]]) @ \
            np.block([[Q[K-1], np.zeros((nx[K-1], nu[K-2] + nx[K-2])), q[K-1]],
                      [np.zeros((nu[K-2], nx[K-1])), R[K-2], S[K-2].T, r[K-2]],
                      [np.zeros((nx[K-2], nx[K-1])), S[K-2], Q[K-2], q[K-2]]]) @ \
            np.block([[B[K-2], A[K-2], b[K-2]],
                      [np.eye(nu[K-2] + nx[K-2] + 1)]])
        temp = temp + \
            np.block([[Fu[K-2], B[K-2].T],
                      [Fx[K-2], A[K-2].T]]) @ \
            np.block([[B[K-2], A[K-2], b[K-2]],
                      [Fu[K-2].T, Fx[K-2].T, np.zeros((nx[K-1], 1))]])
        R_bar = temp[:nu[K-2], :nu[K-2]]
        S_bar = temp[:nu[K-2], nu[K-2]:nu[K-2]+nx[K-2]].T
        Q_bar = temp[nu[K-2]:nu[K-2]+nx[K-2], nu[K-2]:nu[K-2]+nx[K-2]]
        r_bar = temp[:nu[K-2], -1:]
        q_bar = temp[nu[K-2]:, -1:]

        temp = np.block([[Hu[K-2], Hx[K-2], h[K-2]],
                         [
                             np.block([[Hx[K-1], h[K-1]]]) @ \
                             np.block([[B[K-2], A[K-2], b[K-2]],
                                       [np.zeros((1, nu[K-2] + nx[K-2])), 1]]),
                         ]])
        Hu_bar = temp[:, :nu[K-2]]
        Hx_bar = temp[:, nu[K-2]:nu[K-2]+nx[K-2]]
        h_bar = temp[:, -1:]

        R_c[K-2] = R_bar
        S_c[K-2] = S_bar
        Q_c[K-2] = Q_bar
        r_c[K-2] = r_bar
        q_c[K-2] = q_bar
        Hu_c[K-2] = Hu_bar
        Hx_c[K-2] = Hx_bar
        h_c[K-2] = h_bar
        nx_c[K-1] = 0
        nu_c[K-1] = 0
        ng_eq_c[K-1] = 0
        ng_eq_c[K-2] = h_bar.shape[0]
        
        solution = Solve(K - 1, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c,
              Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c, Pl, Pr, L, U,
              Lmbds, rank_k_values, Hut, max_recursion_depth=max_recursion_depth-1, **kwargs)
        if kwargs.get("store_linear_systems", True):
            solution, linear_systems = solution
            kwargs["linear_systems"] = linear_systems
        # KKT, rhs = GetKKT(K-1, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c, Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c)
        # solution_vector = np.linalg.solve(KKT, -rhs)
        # solution = extract_solultion(K-1, nu_c, nx_c, ng_eq_c, solution_vector)
        for k in range(K-1):
            assert solution["u"][k].shape == (nu_c[k], 1)
            assert solution["x"][k].shape == (nx_c[k], 1)
            assert solution["lambda"][k].shape == (ng_eq_c[k], 1)
            assert solution["pi"][k].shape == (nx_c[k] if k > 0 else 0, 1)
        
        # recover full solution
        x = B[K-2] @ solution["u"][K-2] + A[K-2] @ solution["x"][K-2] + b[K-2]
        solution["x"].append(x)
        lmbd = solution["lambda"][K-2]
        lmbd_true = lmbd[:ng_eq[K-2]]
        mu = lmbd[ng_eq[K-2]:]
        solution["lambda"][K-2] = lmbd_true
        solution["lambda"].append(mu)
        pi = Q[K-1] @ x + Hx[K-1].T @ solution["lambda"][K-1] + q[K-1] + Fu[K-2].T @ solution["u"][K-2] + Fx[K-2].T @ solution["x"][K-2]
        solution["pi"].append(pi)
        solution["u"].append(np.zeros((nu[K-1], 1)))

    else:
        KKT, rhs = GetKKT(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b)
        print(f"Initial stage:")
        print_KKT(KKT, rhs)
        solution_vector = np.linalg.solve(KKT, -rhs)
        solution = extract_solultion(K, nu, nx, ng_eq, solution_vector)
        print(f"solution first stage:")
        print_solution(solution)

    print(f"\n\nStage {K-1} complete. returning")
    max_recursion_depth -= 1

    if kwargs.get("store_linear_systems", False):
        if "linear_systems" not in kwargs:
            kwargs["linear_systems"] = []
        KKT, rhs = GetKKT(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b)
        kwargs["linear_systems"].append({
            "KKT": KKT,
            "rhs": rhs,
        })
        return solution, kwargs['linear_systems']
    else:
        return solution


        
def get_expected_matrices(K, nu, nx, r, ng_eq, ng_ineq, modified_nu, modified_nx, modified_ng_eq, modified_ng_ineq, 
                          RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, Gg_ineqt_original, BAbt_original, D_x,
                          Pl, Pr, L, U, 
                          RSQrqt, GuGx, FuFx, Gg_eqt, Gg_ineqt, BAbt, Jt, **kwargs):
    BAbt_expected = [m.copy() for m in BAbt_original]
    GuGx_expected = [m.copy() for m in GuGx_original]
    FuFx_expected = [m.copy() for m in FuFx_original]
    RSQrqt_expected = [m.copy() for m in RSQrqt_original]
    Gg_eqt_expected = [m.copy() for m in Gg_eqt_original]
    Gg_ineqt_expected = [m.copy() for m in Gg_ineqt_original]
    Jt_expected = [J.copy() for J in Jt] if Jt is not None else None

    if kwargs.get("store_linear_systems", False):
        linear_systems = []
        blocks = GetBlockMatrices(K, nu, nx, ng_eq, RSQrqt_expected, GuGx_expected, FuFx_expected, Gg_eqt_expected, BAbt_expected)
        KKT, rhs = GetKKT(K, nu, nx, ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'], Jt=Jt_expected)
        linear_systems.append({
            "KKT": KKT,
            "rhs": rhs,
        })

    # add regularization to hessian
    for k in range(K):
        for i in range(nu[k] + nx[k]):
            RSQrqt_expected[k][i,i] += D_x[k][i]

    # pre-processing code
    Dl_list = []
    Dr_list = []
    Dl_inv_list = []
    Dr_inv_list = []
    for k in range(K-1):
        # construct Dl and Dr
        U1 = U[k][:r[k], :r[k]]
        U2 = U[k][:r[k], r[k]:]
        J = Jt[k].T
        if Pl[k].shape[0] == 0 or Pl[k].shape[1] == 0:
            Pl[k] = np.zeros((0,0))
        if Pr[k].shape[0] == 0 or Pr[k].shape[1] == 0:
            Pr[k] = np.zeros((0,0))
        if L[k].shape[0] == 0 or L[k].shape[1] == 0:
            L[k] = np.zeros((0,0))
        if Jt_expected[k].shape[0] == 0 or Jt_expected[k].shape[1] == 0:
            Jt_expected[k] = np.zeros((0,0))

        Dl = Pl[k] @ L[k] @ np.block([[-U1, np.zeros((r[k],nx[k+1]-r[k]))], [np.zeros((nx[k+1]-r[k],r[k])), np.eye(nx[k+1]-r[k])]])
        Dr = np.block([[np.eye(r[k]), np.linalg.inv(U1) @ U2], [np.zeros((nx[k+1]-r[k],r[k])), np.eye(nx[k+1]-r[k])]]) @ Pr[k].T
        Dl_inv = np.linalg.inv(Dl)
        Dr_inv = np.linalg.inv(Dr)
        Dl_list.append(Dl)
        Dr_list.append(Dr)
        Dl_inv_list.append(Dl_inv)
        Dr_inv_list.append(Dr_inv)
        norm = np.linalg.norm(J - Dl @ np.block([[-np.eye(r[k]), np.zeros((r[k],nx[k+1]-r[k]))], [np.zeros((nx[k+1]-r[k],nx[k+1]))]]) @ Dr)
        print(Jt_expected[k].T.shape)
        print(Dr_inv.shape)
        Jt_expected[k] = ((Dl_inv @ Jt_expected[k].T @ Dr_inv).T)[:modified_nx[k+1], :modified_nx[k+1]]
        assert norm < 1e-6, f"Decomposition error at stage {k}: {norm}"

        # construct W
        W = np.block([[np.eye(nu[k+1]), np.zeros((nu[k+1], nx[k+1]))],
                    [np.zeros((nx[k+1] - r[k], nu[k+1] + r[k])), np.eye(nx[k+1] - r[k])],
                    [np.zeros((r[k], nu[k+1])), np.eye(r[k]), np.zeros((r[k], nx[k+1] - r[k]))]])
        Wp = np.block([[W, np.zeros((W.shape[0], 1))],
                    [np.zeros((1, W.shape[1])), 1]])
            
        BAbt_expected[k] = BAbt_expected[k] @ Dl_inv.T
        
        temp = np.zeros((nu[k] + nx[k], nu[k+1] + nx[k+1]))
        temp[:, nu[k+1]:] = FuFx_expected[k] @ Dr_inv
        GuGx_expected[k] = np.block([temp[:, :nu[k+1]], temp[:, nu[k+1]+modified_nx[k+1]:nu[k+1]+nx[k+1]]])
        FuFx_expected[k] = temp[:, nu[k+1]:nu[k+1]+modified_nx[k+1]]

        RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
        RSQrqt_expected[k+1][:,nu[k+1]:] = RSQrqt_expected[k+1][:,nu[k+1]:] @ np.linalg.inv(Dr)
        RSQrqt_expected[k+1] = Wp @ RSQrqt_expected[k+1]
        RSQrqt_expected[k+1] = RSQrqt_expected[k+1] @ W.T

        Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
        Gg_eqt_expected[k+1] = Wp @ Gg_eqt_expected[k+1]
        Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
        Gg_ineqt_expected[k+1] = Wp @ Gg_ineqt_expected[k+1]

        if k < K-2:
            BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
            FuFx_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ FuFx_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1],:]

            FuFx_expected[k+1] = W @ FuFx_expected[k+1]
            BAbt_expected[k+1] = Wp @ BAbt_expected[k+1]

        Gg_eqt_expected[k] = np.block([Gg_eqt_expected[k], BAbt_expected[k][:, r[k]:]])
        BAbt_expected[k] = BAbt_expected[k][:, :r[k]]

        if kwargs.get("store_linear_systems", False):
            intermediate_nu = modified_nu[:k+2] + nu[k+2:]
            intermediate_nx = modified_nx[:k+2] + nx[k+2:]
            intermediate_ng_eq = modified_ng_eq[:k+1] + ng_eq[k+1:]
            intermediate_ng_ineq = modified_ng_ineq[:k+1] + ng_ineq[k+1:]
            blocks = GetBlockMatrices(K, intermediate_nu, intermediate_nx, intermediate_ng_eq, RSQrqt_expected, GuGx_expected, FuFx_expected, Gg_eqt_expected, BAbt_expected)
            KKT, rhs = GetKKT(K, intermediate_nu, intermediate_nx, intermediate_ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'], Jt=Jt_expected)
            linear_systems.append({
                "KKT": KKT,
                "rhs": rhs,
            })

    # check pre-processing results
    for k in range(K):
        if k < K-1:
            n = np.linalg.norm(BAbt[k] - BAbt_expected[k])
            if n > 1e-6: 
                print(f"BAbt[{k}] error: {n}")
            n = np.linalg.norm(GuGx[k] - GuGx_expected[k])
            if n > 1e-6: 
                print(f"GuGx[{k}] error: {n}")
            n = np.linalg.norm(FuFx[k] - FuFx_expected[k])
            if n > 1e-6: 
                print(f"FuFx[{k}] error: {n}")
            n = np.linalg.norm(RSQrqt[k] - RSQrqt_expected[k])
            if n > 1e-6: 
                print(f"RSQrqt[{k}] error: {n}")
                # print(f"RSQrqt original\n{RSQrqt_original[k]}")
                # print(f"RSQrqt expected\n{RSQrqt_expected[k]}")
                # print(f"RSQrqt:\n{RSQrqt[k]}")
            # print(f"GuGx original\n{GuGx_original[k]}")
            # print(f"GuGx expected\n{GuGx_expected[k]}")
            # print(f"GuGx:\n{GuGx[k]}")
            # print(f"FuFx original\n{FuFx_original[k]}")
            # print(f"FuFx expected\n{FuFx_expected[k]}")
            # print(f"FuFx:\n{FuFx[k]}")

        n = np.linalg.norm(Gg_eqt[k] - Gg_eqt_expected[k])
        if n > 1e-6:
            print(f"Gg_eqt[{k}] error: {n}")
        n = np.linalg.norm(Gg_ineqt[k] - Gg_ineqt_expected[k])
        if n > 1e-6:
            print(f"Gg_ineqt[{k}] error: {n}")

    if kwargs.get("store_linear_systems", False):
        return BAbt_expected, GuGx_expected, FuFx_expected, RSQrqt_expected, Gg_eqt_expected, Gg_ineqt_expected, Dl_list, Dr_list, Dl_inv_list, Dr_inv_list, linear_systems
    else:
        return BAbt_expected, GuGx_expected, FuFx_expected, RSQrqt_expected, Gg_eqt_expected, Gg_ineqt_expected, Dl_list, Dr_list, Dl_inv_list, Dr_inv_list

        