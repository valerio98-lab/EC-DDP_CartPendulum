import numpy as np
import casadi as cs


def backward_pass(x_traj, u_traj, N, n, funcs, lambda_val, mu_val):
    """
    Executes the backward pass to compute the DDP gains.

    Args:
        x_traj  : Current state trajectory, a numpy array of shape (n, N+1)
        u_traj  : Current control trajectory, a numpy array of shape (m, N)
        N       : Time horizon (number of time steps)
        funcs   : Dictionary of CasADi functions (cost, dynamics, constraints, and their derivatives)
        lambda_val, mu_val : Current multiplier and penalty parameter values
        V, Vx, Vxx : Arrays for the value function, its gradient, and Hessian at each time step

    Returns:
        k, K  : Lists of feedforward (k) and feedback (K) gains for each time step
        V, Vx, Vxx : Updated value function, its gradient, and Hessian along the trajectory
    """

    V = np.zeros(N + 1)
    Vx = np.zeros((n, N + 1))
    Vxx = np.zeros((n, n, N + 1))

    # Terminal conditions: at time step N, use the terminal cost
    x_N = x_traj[:, N]

    V[N] = funcs["L_terminal"](x_N) + (lambda_val + (0.5 * mu_val) * funcs["h"](x_N)).T @ funcs["h"](x_N)
    Vx[:, N] = funcs["L_terminal_x"](x_N) + (lambda_val + mu_val * funcs["h"](x_N)).T @ funcs["hx"](x_N)
    Vxx[:, :, N] = funcs["L_terminal_xx"](x_N) + (lambda_val + mu_val * funcs["hx"](x_N)).T @ funcs["hx"](x_N)

    m = u_traj.shape[0]
    n = x_traj.shape[0]
    k = [np.zeros((m, 1)) for _ in range(N)]
    K = [np.zeros((m, n)) for _ in range(N)]

    for i in reversed(range(N)):
        x_i = x_traj[:, i]
        u_i = u_traj[:, i]

        # Evaluate dynamics derivatives
        fx_eval = funcs["fx"](x_i, u_i)
        fu_eval = funcs["fu"](x_i, u_i)

        # Evaluate constraints and their derivatives
        h_eval = funcs["h"](x_i)
        hx_eval = funcs["hx"](x_i)
        hu_eval = funcs["hu"](x_i)

        Imu = np.zeros((4, 4))
        for j in range(4):
            Imu[j, j] = mu_val if (h_eval[j] >= 0 or lambda_val[j] != 0) else 0

        Qx = np.array(funcs["Lx"](x_i, u_i)).T + fx_eval.T @ Vx[:, i + 1] + hx_eval.T @ (lambda_val + Imu @ h_eval)

        Qu = (
            np.array(funcs["Lu"](x_i, u_i)).T + fu_eval.T @ Vx[:, i + 1] + hu_eval.T @ (lambda_val + Imu @ h_eval)
        )  # substitute * with @ so Qu returns 1x1

        Qxx = (
            funcs["Lxx"](x_i, u_i) + fx_eval.T @ Vxx[:, :, i + 1] @ fx_eval + (hx_eval.T @ Imu @ hx_eval)
        )  # +(mu_val * hx_eval.T @ hx_eval))

        Quu = funcs["Luu"](x_i, u_i) + fu_eval.T @ Vxx[:, :, i + 1] @ fu_eval + (hu_eval.T @ Imu @ hu_eval)

        Qux = funcs["Lux"](x_i, u_i) + fu_eval.T @ Vxx[:, :, i + 1] @ fx_eval + (hu_eval.T @ Imu @ hx_eval)

        q = (
            float(funcs["L"](x_i, u_i))
            + V[i + 1]
            + np.array(np.expand_dims(lambda_val, axis=1).T @ h_eval).item()
            + mu_val / 2 * float(cs.sumsqr(h_eval))
        )

        Quu_inv = np.linalg.inv(Quu)
        k[i] = -Quu_inv @ Qu
        K[i] = -Quu_inv @ Qux

        V[i] = q - 0.5 * np.array(cs.evalf(k[i].T @ Quu @ k[i])).flatten()[0]
        Vx[:, i] = np.array(Qx - (k[i].T @ Quu @ K[i]).T).flatten()
        Vxx[:, :, i] = Qxx - K[i].T @ Quu @ K[i]

    return k, K


def forward_pass(x_old, u_old, k, K, N, max_line_search_iters, funcs, mu_val, prev_cost):
    """
    Performs the forward pass with a line search to update the state and control trajectories.

    Args:
        x0              : initial state (numpy array of shape (n,))
        x_old, u_old    : current state and control trajectories (shapes (n, N+1) and (m, N))
        k, K            : feedforward and feedback gains computed in the backward pass
        N               : time horizon (number of time steps)
        max_line_search_iters: maximum iterations for line search
        funcs           : dictionary of CasADi functions
        mu_val : current multiplier and penalty parameter values
        prev_cost       : cost of the previous trajectory

    Returns:
        x_new           : updated state trajectory
        u_new           : updated control trajectory
        new_cost        : cost of the new trajectory
        alpha           : step size used in the line search
    """

    alpha = 1.0  # initial step size
    m = u_old.shape[0]
    n = x_old[:, 0].shape[0]

    unew = np.zeros((m, N))
    xnew = np.zeros((n, N + 1))
    xnew[:, 0] = x_old[:, 0].copy()
    for _ in range(max_line_search_iters):
        new_cost = 0

        for i in range(N):
            dx = xnew[:, i] - x_old[:, i]
            # Calculate the new control law: u_new = u_old + alpha*k + K*(x_new - x_old)
            unew[:, i] = np.array(u_old[:, i] + alpha * k[i].full() + K[i].full() @ dx).flatten()
            xnew[:, i + 1] = np.array(funcs["f"](xnew[:, i], unew[:, i])).flatten()
            new_cost += float(funcs["L_aug_lag"](xnew[:, i], unew[:, i], mu_val))
        new_cost += float(funcs["L_terminal"](xnew[:, N]))

        if new_cost < prev_cost:
            return xnew, unew, new_cost, alpha
        else:
            alpha /= 2.0

    return x_old, u_old, prev_cost, alpha
