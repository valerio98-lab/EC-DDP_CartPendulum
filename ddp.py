import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import time
import model
import argparse

def initialize_system(model_name):
    """
    Initializes the system parameters and optimization settings.

    Returns:
        mod             : the system model instance
        dt              : integration time step
        n, m            : state and control dimensions
        N               : time horizon (number of time steps)
        max_line_search_iters: maximum number of line search iterations
        Q               : state cost matrix (running cost)
        R               : control cost matrix (running cost)
        Q_terminal      : terminal cost matrix
        x_target        : target terminal state (depending on the model type)
    """


    if model_name == "cart_pendulum":
        mod = model.CartPendulum()
    elif model_name == "pendubot":
        mod = model.Pendubot()
    elif model_name == "uav":
        mod = model.Uav()
    else:
        raise ValueError("Invalid model name. Choose from 'cart_pendulum', 'pendubot', or 'uav'.")
    
    dt = 0.01
    n, m = mod.n, mod.m
    N = 100
    max_line_search_iters = 10
    max_ddp_iters = 10

    # Define cost matrices
    Q = np.eye(n) * 0
    R = np.eye(m) * 0.01
    Q_terminal = np.eye(n) * 10000  # terminal cost matrix (large penalty at terminal state)

    # Define the target terminal state based on the model type
    if mod.name == "cart_pendulum":
        x_target = np.array([0, cs.pi, 0, 0])  # upright position for cart-pendulum
    elif mod.name == "pendubot":
        x_target = np.array([cs.pi, 0, 0, 0])
    elif mod.name == "uav":
        x_target = np.array([1, 1, 0, 0, 0, 0])
    else:
        raise ValueError("Unrecognized model type")

    return mod, dt, n, m, N, max_line_search_iters, Q, R, Q_terminal, x_target, max_ddp_iters   


def setup_symbolic_functions(mod, dt, n, m, x_target, Q, R, Q_terminal):
    """
    Creates CasADi symbolic functions for the cost, dynamics, and constraints.

    The cost functions include:
      - Running cost: L(x,u) = (x_target - x)^T Q (x_target - x) + u^T R u
      - Augmented Lagrangian: L_lag(x,u,λ,μ) = L(x,u) + λ^T h(x,u) + (μ/2)*||h(x,u)||^2
      - Terminal cost: L_terminal(x) = (x_target - x)^T Q_terminal (x_target - x)

    The dynamics are defined as:
      f(x,u) = x + dt * f_cont(x,u)

    And the constraints function is taken from the model.

    Returns:
        funcs: A dictionary of CasADi functions including cost, derivatives, dynamics, and constraints.
    """
    opt = cs.Opti()
    X = opt.variable(n)  # symbolic state vector
    U = opt.variable(m)  # symbolic control vector

    # ----------------------------
    # Cost functions
    # ----------------------------
    funcs = dict()

    L_expr = (x_target - X).T @ Q @ (x_target - X) + U.T @ R @ U
    funcs["L"] = cs.Function("L", [X, U], [L_expr], {"post_expand": True})
    funcs["Lx"] = cs.Function("Lx", [X, U], [cs.jacobian(L_expr, X)], {"post_expand": True})
    funcs["Lu"] = cs.Function("Lu", [X, U], [cs.jacobian(L_expr, U)], {"post_expand": True})
    funcs["Lxx"] = cs.Function("Lxx", [X, U], [cs.jacobian(cs.jacobian(L_expr, X), X)], {"post_expand": True})
    funcs["Lux"] = cs.Function("Lux", [X, U], [cs.jacobian(cs.jacobian(L_expr, U), X)], {"post_expand": True})
    funcs["Luu"] = cs.Function("Luu", [X, U], [cs.jacobian(cs.jacobian(L_expr, U), U)], {"post_expand": True})

    L_terminal_expr = (x_target - X).T @ Q_terminal @ (x_target - X)
    funcs["L_terminal"] = cs.Function("L_terminal", [X], [L_terminal_expr], {"post_expand": True})

    # Derivatives for the terminal cost
    funcs["L_terminal_x"] = cs.Function("L_terminal_x", [X], [cs.jacobian(L_terminal_expr, X)], {"post_expand": True})
    funcs["L_terminal_xx"] = cs.Function(
        "L_terminal_xx", [X], [cs.jacobian(cs.jacobian(L_terminal_expr, X), X)], {"post_expand": True}
    )

    # ----------------------------
    # Constraints and their derivatives
    # ----------------------------
    h_expr = mod.constraints(X)
    funcs["h"] = cs.Function("h", [X], [h_expr], {"post_expand": True})
    funcs["hx"] = cs.Function("hx", [X], [cs.jacobian(h_expr, X)], {"post_expand": True})
    funcs["hu"] = cs.Function("hu", [X], [cs.jacobian(h_expr, U)], {"post_expand": True})


    # ----------------------------
    # Dynamics
    # ----------------------------
    # Discrete dynamics: f(x,u) = x + dt * f_cont(x,u)
    f_expr = X + dt * mod.f(X, U)
    funcs["f"] = cs.Function("f", [X, U], [f_expr], {"post_expand": True})
    funcs["fx"] = cs.Function("fx", [X, U], [cs.jacobian(f_expr, X)], {"post_expand": True})
    funcs["fu"] = cs.Function("fu", [X, U], [cs.jacobian(f_expr, U)], {"post_expand": True})

    return funcs


def backward_pass(x_traj, u_traj, N, n, funcs):
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

    V[N] = funcs["L_terminal"](x_N)
    Vx[:, N] = funcs["L_terminal_x"](x_N)
    Vxx[:, :, N] = funcs["L_terminal_xx"](x_N)

    # lambda_val=array([0., 0., 0., 0.]), mu_val=1.1, funcs['h'](x_traj[:, N])=DM([0.185698, -3.20372, 0, 0])
    # λ=DM([-674.089, -274.249, 958.138, 272.322]), μ=1600.0, g(x[:, N])=DM([-0.365853, -0.0667061, 0.47357, 0.239947])

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

        Qx = np.array(funcs["Lx"](x_i, u_i)).T + fx_eval.T @ Vx[:, i + 1] 

        Qu = (
            np.array(funcs["Lu"](x_i, u_i)).T + fu_eval.T @ Vx[:, i + 1] 
        )  # substitute * with @ so Qu returns 1x1

        Qxx = (
            funcs["Lxx"](x_i, u_i) + fx_eval.T @ Vxx[:, :, i + 1] @ fx_eval 
        )  # +(mu_val * hx_eval.T @ hx_eval))

        Quu = funcs["Luu"](x_i, u_i) + fu_eval.T @ Vxx[:, :, i + 1] @ fu_eval 

        Qux = funcs["Lux"](x_i, u_i) + fu_eval.T @ Vxx[:, :, i + 1] @ fx_eval 


        Quu_inv = np.linalg.inv(Quu)
        k[i] = -Quu_inv @ Qu
        K[i] = -Quu_inv @ Qux

        # Update the value function and its derivatives
        V[i] = V[i+1] - 0.5 * k[i].T @ Quu @ k[i]
        Vx[:,i] = np.array(Qx - K[i].T @ Quu @ k[i]).flatten()
        Vxx[:,:,i] = Qxx - K[i].T @ Quu @ K[i]
    return k, K


def forward_pass(x_old, u_old, k, K, N, max_line_search_iters, funcs, prev_cost):
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
            # Propagate the dynamics to obtain the new state trajectory
            xnew[:, i + 1] = np.array(funcs["f"](xnew[:, i], unew[:, i])).flatten()
            new_cost += float(funcs["L"](xnew[:, i], unew[:, i]))
        new_cost += float(funcs["L_terminal"](xnew[:, N]))

        if new_cost < prev_cost:
            return xnew, unew, new_cost, alpha
        else:
            alpha /= 2.0  # reduce step size if cost did not decrease
    
    # If no reduction is found, return the last computed trajectories
    return x_old, u_old, new_cost, alpha


def main():

    parser = argparse.ArgumentParser(description="Run the Augmented Lagrangian Trajectory Optimization")
    parser.add_argument(
        "--model", type=str, choices=["cart_pendulum", "pendubot", "uav"], required=True, help="Model type to simulate"
    )
    args = parser.parse_args()
    # Initialize the system parameters and symbolic functions
    mod, dt, n, m, N, max_line_search_iters, Q, R, Q_terminal, x_target, max_ddp_iters = initialize_system(args.model)
    funcs = setup_symbolic_functions(mod, dt, n, m, x_target, Q, R, Q_terminal)


    x_traj = np.zeros((n, N + 1))
    u_traj = np.ones((m, N))
    x_traj[:, 0] = np.zeros(n)  # initial state


    # Compute the initial cost along the trajectory
    cost = 0
    for i in range(N):
        x_traj[:, i + 1] = np.array(funcs["f"](x_traj[:, i], u_traj[:, i])).flatten()
        cost += float(funcs["L"](x_traj[:, i], u_traj[:, i]))
    cost += float(funcs["L_terminal"](x_traj[:, N]))


    total_time = 0

    # Main optimization loop

    iters = 0
    it_history = []
    cost_history = []

    for iters in range(max_ddp_iters):
    # ----- Backward Pass -----
        bp_start = time.time()
        k, K = backward_pass(x_traj, u_traj, N, n, funcs)
        bp_time = time.time() - bp_start
        
        # ----- Forward Pass with Line Search -----

        fp_start = time.time()
        x_traj, u_traj, new_cost, alpha  = forward_pass(x_traj, u_traj, k, K, N, max_line_search_iters, funcs, cost)
        fp_time = time.time() - fp_start

        total_time += bp_time + fp_time
        iters += 1
        it_history.append(iters)
        cost_history.append(new_cost)

        print('Iteration:', iter, 'BP Time:', round(bp_time*1000), 'FP Time:', round(fp_time*1000))


    print('Total time: ', total_time*1000, ' ms')

    print("it", it_history)
    print("cost", cost_history)

    # Verify the result by simulating the trajectory using the obtained control sequence
    x_check = np.zeros_like(x_traj)
    x_check[:, 0] = np.zeros(n)
    for i in range(N):
        x_check[:, i + 1] = np.array(funcs["f"](x_check[:, i], u_traj[:, i])).flatten()

    # Animate and visualize the results using the model's animation function
    mod.animate(N, x_check, u_traj)

    plt.figure()
    plt.plot(it_history, cost_history, label="Cost")  # Ensure x-axis is iterations and y-axis is cost
    plt.xlabel("Iterations")  # Label the x-axis
    plt.ylabel("Cost")  # Label the y-axis
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

