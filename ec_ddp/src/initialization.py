import casadi as cs
import numpy as np
import model


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
    X = opt.variable(n)  # symbolic state vector
    U = opt.variable(m)  # symbolic control vector
    h_expr = mod.constraints(X)
    lam_sym = cs.MX.sym("lambda", h_expr.shape[0])
    mu_sym = cs.MX.sym("mu", 1)

    # Cost function
    funcs = dict()
    L_expr = (x_target - X).T @ Q @ (x_target - X) + U.T @ R @ U
    funcs["L"] = cs.Function("L", [X, U], [L_expr], {"post_expand": True})
    funcs["Lx"] = cs.Function("Lx", [X, U], [cs.jacobian(L_expr, X)], {"post_expand": True})
    funcs["Lu"] = cs.Function("Lu", [X, U], [cs.jacobian(L_expr, U)], {"post_expand": True})
    funcs["Lxx"] = cs.Function("Lxx", [X, U], [cs.jacobian(cs.jacobian(L_expr, X), X)], {"post_expand": True})
    funcs["Lux"] = cs.Function("Lux", [X, U], [cs.jacobian(cs.jacobian(L_expr, U), X)], {"post_expand": True})
    funcs["Luu"] = cs.Function("Luu", [X, U], [cs.jacobian(cs.jacobian(L_expr, U), U)], {"post_expand": True})

    # Terminal cost
    L_terminal_expr = (x_target - X).T @ Q_terminal @ (x_target - X)
    funcs["L_terminal"] = cs.Function("L_terminal", [X], [L_terminal_expr], {"post_expand": True})
    funcs["L_terminal_x"] = cs.Function("L_terminal_x", [X], [cs.jacobian(L_terminal_expr, X)], {"post_expand": True})
    funcs["L_terminal_xx"] = cs.Function(
        "L_terminal_xx", [X], [cs.jacobian(cs.jacobian(L_terminal_expr, X), X)], {"post_expand": True}
    )

    # Constraints
    funcs["h"] = cs.Function("h", [X], [h_expr], {"post_expand": True})
    funcs["hx"] = cs.Function("hx", [X], [cs.jacobian(h_expr, X)], {"post_expand": True})
    funcs["hu"] = cs.Function("hu", [X], [cs.jacobian(h_expr, U)], {"post_expand": True})

    # Augmented Lagrangian cost:
    # L_lag(x,u,λ,μ) = L(x,u) + λ^T h(x,u) + (μ/2)*||h(x,u)||^2
    L_lag_expr = L_expr + cs.dot(lam_sym, h_expr) + (mu_sym / 2) * cs.sumsqr(h_expr)
    L_aug_lag = L_expr + (mu_sym / 2) * cs.sumsqr(h_expr)

    funcs["L_aug_lag"] = cs.Function("L_aug_lag", [X, U, mu_sym], [L_aug_lag], {"post_expand": True})
    funcs["L_lag_expr"] = cs.Function("L_lag_expr", [X, U, lam_sym, mu_sym], [L_lag_expr], {"post_expand": True})

    # Create derivative functions for the augmented cost
    funcs["Lx_lag"] = cs.Function(
        "Lx_lag", [X, U, lam_sym, mu_sym], [cs.jacobian(L_lag_expr, X)], {"post_expand": True}
    )


    # Discrete dynamics: f(x,u) = x + dt * f_cont(x,u)
    f_expr = X + dt * mod.f(X, U)
    funcs["f"] = cs.Function("f", [X, U], [f_expr], {"post_expand": True})
    funcs["fx"] = cs.Function("fx", [X, U], [cs.jacobian(f_expr, X)], {"post_expand": True})
    funcs["fu"] = cs.Function("fu", [X, U], [cs.jacobian(f_expr, U)], {"post_expand": True})

    return funcs
