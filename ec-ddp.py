import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import time
import model

# initialization
mod = model.CartPendulum()
Δ = 0.01
n, m = mod.n, mod.m
N = 100
max_ddp_iters = 10
max_line_search_iters = 10
Q = np.eye(n) * 0
R = np.eye(m) * 0.01
Q_ter = np.eye(n) * 10000

if mod.name == "cart_pendulum":
    x_ter = np.array((0, cs.pi, 0, 0))
elif mod.name == "pendubot":
    x_ter = np.array((cs.pi, 0, 0, 0))
elif mod.name == "uav":
    x_ter = np.array((1, 1, 0, 0, 0, 0))


# symbolic variables
opt = cs.Opti()
X = opt.variable(n)
U = opt.variable(m)

# constraints
h_ = lambda x, u: mod.constraints(x, u)
h_dim = mod.constraints(X, U).shape[0]
h = cs.Function("h", [X, U], [mod.constraints(X, U)], {"post_expand": True})

lambdas = opt.variable(h_dim)
mu = opt.parameter()


# cost function
L_ = lambda x, u: (x_ter - x).T @ Q @ (x_ter - x) + u.T @ R @ u
L_lag_ = (
    lambda x, u, lambdas, mu: L_(x, u)
    + lambdas.T @ h(x, u)
    + (mu / 2) * cs.sumsqr(h(x, u))
)
L_ter_ = lambda x: (x_ter - x).T @ Q_ter @ (x_ter - x)

# cost functions -> symbolic functions
L = cs.Function("L", [X, U], [L_(X, U)], {"post_expand": True})
L_lag = cs.Function("L_lag", [X, U, lambdas, mu], [L_lag_(X, U, lambdas, mu)], {"post_expand": True})
L_ter = cs.Function("L_ter", [X], [L_ter_(X)], {"post_expand": True})

# derivatives of cost functions -> symbolic functions
Lx_lag = cs.Function("Lx_lag", [X, U, lambdas, mu], [cs.jacobian(L_lag(X, U, lambdas, mu), X)], {"post_expand": True})
Lu_lag = cs.Function("Lu_lag", [X, U, lambdas, mu], [cs.jacobian(L_lag(X, U, lambdas, mu), U)], {"post_expand": True})
Lxx_lag = cs.Function("Lxx_lag", [X, U, lambdas, mu], [cs.jacobian(Lx_lag(X, U, lambdas, mu), X)], {"post_expand": True})
Lux_lag = cs.Function("Lux_lag", [X, U, lambdas, mu], [cs.jacobian(Lu_lag(X, U, lambdas, mu), X)], {"post_expand": True})
Luu_lag = cs.Function("Luu", [X, U, lambdas, mu], [cs.jacobian(Lu_lag(X, U, lambdas, mu), U)], {"post_expand": True})
L_terx = cs.Function("L_terx", [X], [cs.jacobian(L_ter(X), X)], {"post_expand": True})
L_terxx = cs.Function(
    "L_terxx", [X], [cs.jacobian(L_terx(X), X)], {"post_expand": True}
)


# dynamics
f_cont = mod.f
f_ = lambda x, u: x + Δ * f_cont(x, u)
f = cs.Function("f", [X, U], [f_(X, U)], {"post_expand": True})
fx = cs.Function("fx", [X, U], [cs.jacobian(f_(X, U), X)], {"post_expand": True})
fu = cs.Function("fu", [X, U], [cs.jacobian(f_(X, U), U)], {"post_expand": True})

#eq. constraints
h = cs.Function("h", [X, U], [h_(X,U)], {"post_expand": True})
hx = cs.Function("hx", [X, U], [cs.jacobian(h(X, U), X)], {"post_expand": True})
hu = cs.Function("hu", [X, U], [cs.jacobian(h(X, U), U)], {"post_expand": True})

# initial forward pass
x = np.zeros((n, N + 1))
u = np.ones((m, N))
mu_num = 1.0
lambda_num = np.zeros(h_dim)

x[:, 0] = np.zeros(n)

cost = 0
for i in range(N):
    x[:, i + 1] = np.array(f(x[:, i], u[:, i])).flatten()
    cost += L(x[:, i], u[:, i])
cost += L_ter(x[:, N])

k = [np.zeros((m, 1))] * (N + 1)
K = [np.zeros((m, n))] * (N + 1)

V = np.zeros(N + 1)
Vx = np.zeros((n, N + 1))
Vxx = np.zeros((n, n, N + 1))

total_time = 0


for iter in range(max_ddp_iters):
    # backward pass
    backward_pass_start_time = time.time()
    V[N] = L_ter(x[:, N])
    Vx[:, N] = np.array(L_terx(x[:, N])).flatten()
    Vxx[:, :, N] = L_terxx(x[:, N])


    for i in reversed(range(N)):
        fx_eval = fx(x[:, i], u[:, i])
        fu_eval = fu(x[:, i], u[:, i])
        hx_eval = hx(x[:, i], u[:, i])
        hu_eval = hu(x[:, i], u[:, i])

        Qx = Lx_lag(x[:, i], u[:, i], lambda_num, mu_num).T + fx_eval.T @ Vx[:, i + 1] + hx_eval.T @ lambda_num
        Qu = Lu_lag(x[:, i], u[:, i], lambda_num, mu_num).T + fu_eval.T @ Vx[:, i + 1] + hu_eval.T @ lambda_num
        Qxx = Lxx_lag(x[:, i], u[:, i], lambda_num, mu_num) + fx_eval.T @ Vxx[:, :, i + 1] @ fx_eval + mu_num * hx_eval.T @ hx_eval
        Quu = Luu_lag(x[:, i], u[:, i], lambda_num, mu_num) + fu_eval.T @ Vxx[:, :, i + 1] @ fu_eval + mu_num * hu_eval.T @ hu_eval
        Qux = Lux_lag(x[:, i], u[:, i], lambda_num, mu_num) + fu_eval.T @ Vxx[:, :, i + 1] @ fx_eval + mu_num * hu_eval.T @ hx_eval

        Quu_inv = np.linalg.inv(Quu)

        k[i] = -Quu_inv @ Qu
        K[i] = -Quu_inv @ Qux

        V[i] = V[i + 1] - 0.5 * np.array(cs.evalf(k[i].T @ Quu @ k[i])).flatten()[0]
        Vx[:, i] = np.array(Qx - K[i].T @ Quu @ k[i]).flatten()
        Vxx[:, :, i] = Qxx - K[i].T @ Quu @ K[i]

    backward_pass_time = time.time() - backward_pass_start_time

    # forward pass
    forward_pass_start_time = time.time() 
    unew = np.ones((m, N))
    xnew = np.zeros((n, N + 1))
    xnew[:, 0] = x[:, 0]

    # line search
    alpha = 1.0
    for ls_iter in range(max_line_search_iters):
        new_cost = 0
        for i in range(N):
            unew[:, i] = np.array(
                u[:, i] + alpha * k[i] + K[i] @ (xnew[:, i] - x[:, i])
            ).flatten()
            xnew[:, i + 1] = np.array(f(xnew[:, i], unew[:, i])).flatten()
            new_cost = new_cost + L(xnew[:, i], unew[:, i])
        new_cost = new_cost + L_ter(xnew[:, N])

        if new_cost < cost:
            cost = new_cost
            x = xnew
            u = unew
            break
        else:
            alpha /= 2.0

    for i in range(N):
        lambda_num += mu_num * (np.array(h(x[:, i], u[:, i])).squeeze(-1))

    if np.linalg.norm(h(x[:, :-1], u)) > 1e-3:
        mu_num *= 10

    forward_pass_time = time.time() - forward_pass_start_time
    total_time += backward_pass_time + forward_pass_time
    print(
        "Iteration:",
        iter,
        "BP Time:",
        round(backward_pass_time * 1000),
        "FP Time:",
        round(forward_pass_time * 1000),
        "||h(x, u)||:",
        np.linalg.norm(h(x[:, :-1], u))
    )

print("Total time: ", total_time * 1000, " ms")

# check result
xcheck = np.zeros((n, N + 1))
xcheck[:, 0] = np.zeros(n)
for i in range(N):
    xcheck[:, i + 1] = np.array(f(xcheck[:, i], u[:, i])).flatten()

# display
mod.animate(N, xcheck, u)
