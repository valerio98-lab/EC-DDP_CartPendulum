import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import time
import model
import argparse

from ec_ddp.src.initialization import initialize_system, setup_symbolic_functions
from ec_ddp.src.backward_forward import backward_pass, forward_pass


def _update_multipliers(x_traj, u_traj, funcs, lambda_val, mu_val, eta, omega, beta, k_mu, N):
    """
    Updates the multipliers (λ) and the penalty parameter (μ) based on the constraint violation
    and the gradient of the augmented Lagrangian.

    Args:
        x_traj, u_traj : current state and control trajectories
        funcs          : dictionary of CasADi functions
        lambda_val, mu_val : current multiplier and penalty parameter
        eta, omega     : current thresholds for constraint violation and gradient norm
        beta           : exponent used in threshold update
        k_mu           : factor by which μ is increased when constraints are not satisfied
        N              : time horizon (number of time steps)

    Returns:
        Updated lambda_val, mu_val, eta, omega, and the gradient norm Lgrad.
    """
    # Evaluate the gradient at the terminal time using the last control input (N-1)
    lag_matrix = []

    for index in range(u_traj.shape[1]):
        Lgrad = funcs["Lx_lag"](x_traj[:, index + 1], u_traj[:, index], lambda_val, mu_val).full()
        lag_matrix.append(Lgrad)

    lag_matrix = np.array(lag_matrix)
    lag_matrix = lag_matrix.reshape(lag_matrix.shape[0], lag_matrix.shape[2])
    infinite_norm_lag = np.linalg.norm(lag_matrix, np.inf)

    if infinite_norm_lag < omega:
        # Evaluate the constraint violation at the terminal time

        cons_inf_norm = np.linalg.norm(funcs["h"](x_traj[:, N]).full(), np.inf)

        # print("infinite_norm_cons: ", cons_inf_norm)
        if cons_inf_norm < eta:
            # Update the multipliers if constraints are sufficiently satisfied
            lambda_val = lambda_val + mu_val * np.array(funcs["h"](x_traj[:, N])).flatten()
            eta /= mu_val**beta
            omega /= mu_val
        else:

            # Increase the penalty parameter if constraints are not satisfied enough
            mu_val *= k_mu
            pass
    return lambda_val, mu_val, eta, omega, infinite_norm_lag


def _compute_cost(x_traj, u_traj, mu_val, funcs, N):
    cost = 0
    for i in range(N):
        x_traj[:, i + 1] = np.array(funcs["f"](x_traj[:, i], u_traj[:, i])).flatten()
        cost += float(funcs["L"](x_traj[:, i], u_traj[:, i]))
    cost += float(funcs["L_terminal"](x_traj[:, N]))
    return cost


def _animate_trajectory(
    mod, N, x_check, u_traj, it_history, cost_history, constraint_history, total_time, constraint_norm
):
    mod.animate("ec-ddp simulation", N, x_check, u_traj)

    _, axs = plt.subplots(1, 4, figsize=(12, 3))
    axs[0].set_title("Cost EC-DDP")
    axs[1].set_title("Constraint Satisfaction EC-DDP")

    axs[0].plot(it_history, cost_history, label="Cost")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Cost")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(it_history, constraint_history, label="Constraint norm", color="red")
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("||h(x)||")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].axis("off")
    axs[2].text(0.5, 0.5, f"Total execution time:\n{total_time*1000:.2f} ms", ha="center", va="center", fontsize=12)

    axs[3].axis("off")
    axs[3].text(0.5, 0.5, f"Constraint satisfaction:\n{constraint_norm:.2f}", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.show()


def ec_ddp_algorithm(model):
    # Initialize the system parameters and symbolic functions
    mod, dt, n, m, N, max_line_search_iters, Q, R, Q_terminal, x_target = initialize_system(model)
    funcs = setup_symbolic_functions(mod, dt, n, m, x_target, Q, R, Q_terminal)

    # Initialize state and control trajectories
    h_dim = mod.constraints(np.zeros(n)).shape[0]

    x_traj = np.zeros((n, N + 1))
    u_traj = np.ones((m, N))
    x_traj[:, 0] = np.zeros(n)  # initial state
    lambda_val = np.zeros(h_dim)
    mu_val = 1.1
    eta = 2
    omega = 200
    omega_threshold = 125 if model == "cart_pendulum" else 80  # 60 for pendubot if you constrained also h3 and h4
    beta = 0.5
    k_mu = 1
    eta_threshold = 0.5

    total_time = 0
    iteration = 0
    it_history = []
    cost_history = []
    constraint_history = []

    # Compute the initial cost along the trajectory
    cost = _compute_cost(x_traj, u_traj, mu_val, funcs, N)

    while omega > omega_threshold and eta > eta_threshold:

        ## Update plots
        constraint_norm = np.linalg.norm(np.array(funcs["h"](x_traj[:, N])), np.inf)
        constraint_history.append(constraint_norm)

        it_history.append(iteration)
        cost_history.append(cost)

        iteration += 1

        bp_start = time.time()
        k, K = backward_pass(x_traj, u_traj, N, n, funcs, lambda_val, mu_val)
        bp_time = time.time() - bp_start

        fp_start = time.time()
        x_traj, u_traj, cost, _ = forward_pass(x_traj, u_traj, k, K, N, max_line_search_iters, funcs, mu_val, cost)
        fp_time = time.time() - fp_start

        total_time += bp_time + fp_time

        lambda_val, mu_val, eta, omega, infinite_norm_lag = _update_multipliers(
            x_traj, u_traj, funcs, lambda_val, mu_val, eta, omega, beta, k_mu, N
        )

        print(
            f"Iteration: {iteration:2d} | BP: {round(bp_time*1000):4d} ms | FP: {round(fp_time*1000):4d} ms | "
            f"grad_L: {infinite_norm_lag:.4f} | ||h(x)||: {np.linalg.norm(np.array(funcs['h'](x_traj[:, N])), np.inf):.4f} | "
            f"eta: {eta:.4f} | omega: {omega:.4f} | mu: {mu_val:.4f}"
        )

    return total_time, it_history, cost_history, constraint_history, x_traj, u_traj, funcs, mod, n, N


def main(model=None):

    if model is None:
        parser = argparse.ArgumentParser(description="Run the Augmented Lagrangian Trajectory Optimization")
        parser.add_argument(
            "--model",
            type=str,
            choices=["cart_pendulum", "pendubot", "uav"],
            required=True,
            help="Model type to simulate",
        )
        args = parser.parse_args()
        model = args.model

    total_time, it_history, cost_history, constraint_history, x_traj, u_traj, funcs, mod, n, N = ec_ddp_algorithm(
        model
    )

    print(f"Total time: {total_time*1000:.2f} ms")
    cost_history = np.array(cost_history).flatten().tolist()
    print("it", it_history)
    print("cost", cost_history)

    # Verify the result by simulating the trajectory using the obtained control sequence
    x_check = np.zeros_like(x_traj)
    x_check[:, 0] = np.zeros(n)
    for i in range(N):
        x_check[:, i + 1] = np.array(funcs["f"](x_check[:, i], u_traj[:, i])).flatten()

    constraint_norm = np.linalg.norm(np.array(funcs["h"](x_check[:, N])), np.inf)

    _animate_trajectory(
        mod, N, x_check, u_traj, it_history, cost_history, constraint_history, total_time, constraint_norm
    )


if __name__ == "__main__":
    main()
