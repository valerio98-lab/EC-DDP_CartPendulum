import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import time
import model 


def initialize_system():
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

    mod = model.CartPendulum()
    dt = 0.01               
    n, m = mod.n, mod.m     
    N = 100                 
    max_line_search_iters = 10  

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

    return mod, dt, n, m, N, max_line_search_iters, Q, R, Q_terminal, x_target


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
    X = opt.variable(n)   # symbolic state vector
    U = opt.variable(m)   # symbolic control vector

    # ----------------------------
    # Cost functions
    # ----------------------------

    L_expr = (x_target - X).T @ Q @ (x_target - X) + U.T @ R @ U

    # Get the constraints expression from the model 
    h_expr = mod.constraints(X, U)

    # Define symbols for the multipliers and penalty parameter (using MX so that types match h_expr)
    lam_sym = cs.MX.sym("lambda", h_expr.shape[0])
    mu_sym = cs.MX.sym("mu", 1)

    # Augmented Lagrangian cost:
    # L_lag(x,u,λ,μ) = L(x,u) + λ^T h(x,u) + (μ/2)*||h(x,u)||^2
    L_lag_expr = L_expr + cs.dot(lam_sym, h_expr) + (mu_sym[0] / 2) * cs.sumsqr(h_expr)

    L_terminal_expr = (x_target - X).T @ Q_terminal @ (x_target - X)

    funcs = dict()
    funcs["L"] = cs.Function("L", [X, U], [L_expr], {"post_expand": True})
    # Note: we pass the same lam_sym and mu_sym as inputs for consistency
    funcs["L_lag"] = cs.Function("L_lag", [X, U, lam_sym, mu_sym], [L_lag_expr], {"post_expand": True})
    funcs["L_terminal"] = cs.Function("L_terminal", [X], [L_terminal_expr], {"post_expand": True})

    # Create derivative functions for the augmented cost
    funcs["Lx_lag"] = cs.Function("Lx_lag", [X, U, lam_sym, mu_sym],
                                  [cs.jacobian(L_lag_expr, X)], {"post_expand": True})
    funcs["Lu_lag"] = cs.Function("Lu_lag", [X, U, lam_sym, mu_sym],
                                  [cs.jacobian(L_lag_expr, U)], {"post_expand": True})
    funcs["Lxx_lag"] = cs.Function("Lxx_lag", [X, U, lam_sym, mu_sym],
                                   [cs.jacobian(cs.jacobian(L_lag_expr, X), X)], {"post_expand": True})
    funcs["Lux_lag"] = cs.Function("Lux_lag", [X, U, lam_sym, mu_sym],
                                   [cs.jacobian(cs.jacobian(L_lag_expr, U), X)], {"post_expand": True})
    funcs["Luu_lag"] = cs.Function("Luu_lag", [X, U, lam_sym, mu_sym],
                                   [cs.jacobian(cs.jacobian(L_lag_expr, U), U)], {"post_expand": True})

    # Derivatives for the terminal cost
    funcs["L_terminal_x"] = cs.Function("L_terminal_x", [X],
                                        [cs.jacobian(L_terminal_expr, X)],
                                        {"post_expand": True})
    funcs["L_terminal_xx"] = cs.Function("L_terminal_xx", [X],
                                         [cs.jacobian(cs.jacobian(L_terminal_expr, X), X)],
                                         {"post_expand": True})

    # ----------------------------
    # Dynamics
    # ----------------------------
    # Discrete dynamics: f(x,u) = x + dt * f_cont(x,u)
    f_expr = X + dt * mod.f(X, U)
    funcs["f"] = cs.Function("f", [X, U], [f_expr], {"post_expand": True})
    funcs["fx"] = cs.Function("fx", [X, U], [cs.jacobian(f_expr, X)], {"post_expand": True})
    funcs["fu"] = cs.Function("fu", [X, U], [cs.jacobian(f_expr, U)], {"post_expand": True})

    # ----------------------------
    # Constraints and their derivatives
    # ----------------------------
    funcs["h"] = cs.Function("h", [X, U], [h_expr], {"post_expand": True})
    funcs["hx"] = cs.Function("hx", [X, U], [cs.jacobian(h_expr, X)], {"post_expand": True})
    funcs["hu"] = cs.Function("hu", [X, U], [cs.jacobian(h_expr, U)], {"post_expand": True})
    funcs["hxx"] = cs.Function("hxx", [X, U],
                               [cs.jacobian(cs.jacobian(h_expr, X), X)],
                               {"post_expand": True})
    funcs["huu"] = cs.Function("huu", [X, U],
                               [cs.jacobian(cs.jacobian(h_expr, U), U)],
                               {"post_expand": True})
    funcs["hux"] = cs.Function("hux", [X, U],
                               [cs.jacobian(cs.jacobian(h_expr, U), X)],
                               {"post_expand": True})

    return funcs


def backward_pass(x_traj, u_traj, N, funcs, lambda_val, mu_val, V, Vx, Vxx):
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
    # Terminal conditions: at time step N, use the terminal cost
    x_N = x_traj[:, N]
    V[N] = float(funcs["L_terminal"](x_N))
    Vx[:, N] = np.array(funcs["L_terminal_x"](x_N)).flatten()
    Vxx[:, :, N] = funcs["L_terminal_xx"](x_N)

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
        h_eval = funcs["h"](x_i, u_i)
        hx_eval = funcs["hx"](x_i, u_i)
        hu_eval = funcs["hu"](x_i, u_i)
        hxx_eval = funcs["hxx"](x_i, u_i)
        huu_eval = funcs["huu"](x_i, u_i)
        hux_eval = funcs["hux"](x_i, u_i)

        print("Type: ", type(h_eval))
        print("Funcs: ",  type(funcs["h"]))

        Imu = np.zeros((2, 2))
        for j in range(2):
            Imu[j,j] = mu_val if (h_eval[j] >= 0 or lambda_val[j] != 0) else 0

        # Compute Q_x, Q_u (jacobian of the cost-to-go)
        print("H_xx: ", hxx_eval.shape)
        Qx = (np.array(funcs["Lx_lag"](x_i, u_i, lambda_val, mu_val)).T +
              fx_eval.T @ Vx[:, i+1] +
              hx_eval.T @ (lambda_val + Imu @ h_eval))
        
        print("Size of Qx", Qx.shape)
        Qu = (np.array(funcs["Lu_lag"](x_i, u_i, lambda_val, mu_val)).T +
              fu_eval.T @ Vx[:, i+1] +
              hu_eval.T @ (lambda_val + Imu @ h_eval))      #substitute * with @ so Qu returns 1x1
        print("size of Qu", Qu.shape)
        print("Size of Lu_shape", funcs["Lu_lag"](x_i, u_i, lambda_val, mu_val).shape)

        
        # Compute Q_xx, Q_uu, and Q_ux (Hessian of the cost-to-go)
        print("allala",( fu_eval.T @ Vx[:, i+1]).shape)
        print("wlellle",(hu_eval.T @ (lambda_val + Imu @ h_eval)).shape) ##1,2  ## substitute * with @
        print(((hx_eval.T @ Imu @ hx_eval).T).shape)
        Qxx = (funcs["Lxx_lag"](x_i, u_i, lambda_val, mu_val) +
               fx_eval.T @ Vxx[:, :, i+1] @ fx_eval + (hx_eval.T @ Imu @ hx_eval)) #+(mu_val * hx_eval.T @ hx_eval))
        

        print("huu: ", huu_eval.shape)
        print("hux: ", hux_eval.shape)
        print("Size of Qxx", Qxx.shape)
        Quu = (funcs["Luu_lag"](x_i, u_i, lambda_val, mu_val) +
               fu_eval.T @ Vxx[:, :, i+1] @ fu_eval + (hu_eval.T @ Imu @ hu_eval))
        
        print("Size of Quu", Quu.shape)
        print("Size of Luu_lag", funcs["Luu_lag"](x_i, u_i, lambda_val, mu_val).shape)
        
        Qux = (funcs["Lux_lag"](x_i, u_i, lambda_val, mu_val) +
               fu_eval.T @ Vxx[:, :, i+1] @ fx_eval + (hu_eval.T @ Imu @ hx_eval))
        print("Size of Qux", Qux.shape)

        # # Compute the cost-to-go at time step i                            #following Scianca's code
        # q = (float(funcs["L_lag"](x_i, u_i, lambda_val, mu_val)) +
        #      V[i+1] +
        #      np.array(lambda_val.T @ h_eval).item() +
        #      mu_val/2 * float(cs.sumsqr(h_eval)))

        Quu_inv = np.linalg.inv(Quu)
        k[i] = -Quu_inv @ Qu
        K[i] = -Quu_inv @ Qux


        # Update the value function and its derivatives
        V[i] = V[i+1] - 0.5 * np.array(cs.evalf(k[i].T @ Quu @ k[i])).flatten()[0]   # Scianca's code q = V[i+1]
        Vx[:, i] = np.array(Qx - K[i].T @ Quu @ k[i]).flatten()
        Vxx[:, :, i] = Qxx - K[i].T @ Quu @ K[i]
    return k, K, V, Vx, Vxx

def forward_pass(x0, x_old, u_old, k, K, N, max_line_search_iters, funcs, lambda_val, mu_val, prev_cost):
    """
    Performs the forward pass with a line search to update the state and control trajectories.
    
    Args:
        x0              : initial state (numpy array of shape (n,))
        x_old, u_old    : current state and control trajectories (shapes (n, N+1) and (m, N))
        k, K            : feedforward and feedback gains computed in the backward pass
        N               : time horizon (number of time steps)
        max_line_search_iters: maximum iterations for line search
        funcs           : dictionary of CasADi functions
        lambda_val, mu_val : current multiplier and penalty parameter values
        prev_cost       : cost of the previous trajectory
        
    Returns:
        x_new           : updated state trajectory
        u_new           : updated control trajectory
        new_cost        : cost of the new trajectory
        alpha           : step size used in the line search
    """
   
    alpha = 1.0  # initial step size
    m = u_old.shape[0]
    n = x0.shape[0]

    for ls_iter in range(max_line_search_iters):
        new_cost = 0
        unew = np.zeros((m, N))
        xnew = np.zeros((n, N+1))
        xnew[:, 0] = x0.copy()

        for i in range(N):
            dx = xnew[:, i] - x_old[:, i]
            # Calculate the new control law: u_new = u_old + alpha*k + K*(x_new - x_old)
            unew[:, i] = np.array(u_old[:, i] + alpha * k[i].full().flatten() + K[i].full() @ dx).flatten()
            # Propagate the dynamics to obtain the new state trajectory
            xnew[:, i+1] = np.array(funcs["f"](xnew[:, i], unew[:, i])).flatten()
            new_cost += float(funcs["L_lag"](xnew[:, i], unew[:, i], lambda_val, mu_val))
        new_cost += float(funcs["L_terminal"](xnew[:, N]))

        if new_cost < prev_cost:
            return xnew, unew, new_cost, alpha
        else:
            alpha /= 2.0  # reduce step size if cost did not decrease

    # If no reduction is found, return the last computed trajectories
    return xnew, unew, new_cost, alpha

def update_multipliers(x_traj, u_traj, funcs, lambda_val, mu_val, eta, omega, beta, k_mu, N):
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
    Lgrad = np.linalg.norm(np.array(funcs["Lx_lag"](x_traj[:, N], u_traj[:, N-1], lambda_val, mu_val)), np.inf)
    if Lgrad < omega:
        # Evaluate the constraint violation at the terminal time
        norm_cons = np.linalg.norm(np.array(funcs["h"](x_traj[:, N], u_traj[:, N-1])), np.inf)
        if norm_cons < eta:
            # Update the multipliers if constraints are sufficiently satisfied
            lambda_val = lambda_val + mu_val * np.array(funcs["h"](x_traj[:, N], u_traj[:, N-1])).flatten()
            eta /= mu_val**beta
            omega /= mu_val
        else:
            # Increase the penalty parameter if constraints are not satisfied enough
            mu_val *= k_mu
    return lambda_val, mu_val, eta, omega, Lgrad



def main():
    # Initialize the system parameters and symbolic functions
    mod, dt, n, m, N, max_line_search_iters, Q, R, Q_terminal, x_target = initialize_system()
    funcs = setup_symbolic_functions(mod, dt, n, m, x_target, Q, R, Q_terminal)

    # Initialize state and control trajectories
    x_traj = np.zeros((n, N+1))
    u_traj = np.ones((m, N))
    x_traj[:, 0] = np.zeros(n)  # initial state

    # Initialize multipliers and penalty parameter
    h_dim = mod.constraints(np.zeros(n), np.ones(m)).shape[0]
    lambda_val = np.zeros(h_dim)
    mu_val = 1.1

    # Compute the initial cost along the trajectory
    cost = 0
    for i in range(N):
        x_traj[:, i+1] = np.array(funcs["f"](x_traj[:, i], u_traj[:, i])).flatten()
        cost += float(funcs["L_lag"](x_traj[:, i], u_traj[:, i], lambda_val, mu_val))
    cost += float(funcs["L_terminal"](x_traj[:, N]))

    # Initialize arrays for the value function, its gradient, and Hessian used in the backward pass
    V = np.zeros(N+1)
    Vx = np.zeros((n, N+1))
    Vxx = np.zeros((n, n, N+1))

    # Algorithm parameters for updating constraints and gradient tolerances
    eta = 1
    omega = 5
    beta = 0.6
    k_mu = 3
    eta_threshold = 0.6
    omega_threshold = 1

    # Lists to store the history of μ and the norm of λ
    mu_history = []
    lambda_history = []

    total_time = 0
    iteration = 0
    bp_start = time.time()

    # Main optimization loop
    while eta > eta_threshold and omega > omega_threshold:
        iteration += 1
        mu_history.append(mu_val)
        lambda_history.append(np.linalg.norm(lambda_val, np.inf))
        print(f"Iteration {iteration}")

        # ----- Backward Pass -----
        k, K, V, Vx, Vxx = backward_pass(x_traj, u_traj, N, funcs, lambda_val, mu_val, V, Vx, Vxx)
        bp_time = time.time() - bp_start

        # ----- Forward Pass with Line Search -----
        fp_start = time.time()
        x_new, u_new, new_cost, alpha = forward_pass(x_traj[:, 0], x_traj, u_traj, k, K,
                                                     N, max_line_search_iters, funcs, lambda_val, mu_val, cost)
        fp_time = time.time() - fp_start

        if new_cost < cost:
            cost = new_cost
            x_traj = x_new.copy()
            u_traj = u_new.copy()

        total_time += bp_time + fp_time

        # Update multipliers and penalty parameter based on current constraint violation and gradient norm
        lambda_val, mu_val, eta, omega, Lgrad = update_multipliers(x_traj, u_traj, funcs,
                                                                     lambda_val, mu_val, eta, omega, beta, k_mu, N)

        print(f"Iteration: {iteration:2d} | BP: {round(bp_time*1000):4d} ms | FP: {round(fp_time*1000):4d} ms | "
              f"grad_L: {Lgrad:.4f} | ||h(x)||: {np.linalg.norm(np.array(funcs['h'](x_traj[:, N], u_traj[:, N-1])), np.inf):.4f} | "
              f"eta: {eta:.4f} | omega: {omega:.4f} | mu: {mu_val:.4f}")

    print(f"Total time: {total_time*1000:.2f} ms")

    # Verify the result by simulating the trajectory using the obtained control sequence
    x_check = np.zeros_like(x_traj)
    x_check[:, 0] = np.zeros(n)
    for i in range(N):
        x_check[:, i+1] = np.array(funcs["f"](x_check[:, i], u_traj[:, i])).flatten()

    # Animate and visualize the results using the model's animation function
    mod.animate(N, x_check, u_traj)

    plt.figure()
    plt.plot(mu_history, label="mu")
    plt.plot(lambda_history, label="||lambda||")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
