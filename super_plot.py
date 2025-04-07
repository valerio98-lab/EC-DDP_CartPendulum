import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def plot(model):
    # === Load model-specific data ===
    prefix_plots = f"vectors_for_plots/{model}_"
    prefix_params = f"vectors_for_pparameters/{model}_"

    cost_history_ddp = np.load(prefix_plots + "cost_history_ddp.npy")

    it_history_ddp = np.load(prefix_plots + "it_history_ddp.npy")
    constraint_history_ddp = np.load(prefix_plots + "constraint_history_ddp.npy")

    cost_history_ec_ddp = np.load(prefix_plots + "cost_history_ec_ddp.npy")
    true_cost_history_ddp = np.load(prefix_plots + "true_cost_history_ec_ddp.npy")
    it_history_ec_ddp = np.load(prefix_plots + "it_history_ec_ddp.npy")
    constraint_history_ec_ddp = np.load(prefix_plots + "constraint_history_ec_ddp.npy")

    eta_history = np.load(prefix_params + "eta_history.npy")
    infinite_norm_lag_history = np.load(prefix_params + "infinite_norm_lag_history.npy")
    mu_history = np.load(prefix_params + "mu_history.npy")
    omega_history = np.load(prefix_params + "omega_history.npy")
    lambda_history = np.load(prefix_params + "lambda_history.npy")

    # === Output folder for frames ===
    os.makedirs("frames", exist_ok=True)

    def create_gif_plot(plot_func, num_frames, gif_name):
        filenames = []
        for i in range(1, num_frames + 1):
            fname = f"frames/{model}_{gif_name}_{i:03d}.png"
            plot_func(i)
            plt.savefig(fname)
            plt.close()
            filenames.append(fname)

        gif_path = f"{model}_{gif_name}.gif"
        with imageio.get_writer(gif_path, mode="I", fps=2) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in filenames:
            os.remove(filename)

    # === 1. Cost Progression GIF ===
    def plot_cost(i):
        plt.figure(figsize=(8, 5))
        plt.plot(
            it_history_ec_ddp[:i],
            cost_history_ec_ddp[:i],
            marker="o",
            color="red",
            label="EC-DDP",
        )
        plt.plot(
            it_history_ddp[:i],
            cost_history_ddp[:i],
            marker="o",
            color="blue",
            label="DDP",
        )
        plt.plot(
            it_history_ddp[:i],
            true_cost_history_ddp[:i],
            marker="o",
            color="green",
            label="True cost ",
        )
        plt.title("EC-DDP cost vs DDP cost")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    # === 2. Lambda + mu Progression GIF ===
    def plot_lambdas(i):
        plt.figure(figsize=(8, 5))
        for j in range(lambda_history.shape[1]):
            plt.plot(
                it_history_ec_ddp[:i],
                lambda_history[:i, j],
                marker="o",
                label=r"$\lambda_" + f"{j + 1}" + r"$",
            )
        plt.plot(
            it_history_ec_ddp[:i],
            mu_history[:i],
            marker="o",
            label=r"$\mu$",
            color="orange",
        )
        plt.title(r"$\lambda$ and $\mu$ values over iterations")
        plt.xlabel("Iterations")
        plt.ylabel(r"$\lambda$ , $\mu$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    # === 3. Parameters GIF ===
    def plot_parameters(i):
        plt.figure(figsize=(8, 5))
        plt.plot(it_history_ec_ddp[:i], eta_history[:i], marker="o", label=r"$\eta$")
        plt.plot(
            it_history_ec_ddp[:i],
            infinite_norm_lag_history[:i],
            marker="o",
            label=r"$|| L(x) ||_{\infty}$",
        )
        plt.plot(it_history_ec_ddp[:i], mu_history[:i], marker="o", label=r"$\mu$")
        plt.plot(
            it_history_ec_ddp[:i], omega_history[:i], marker="o", label=r"$\omega$"
        )
        plt.title("Parameter values over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Parameter values")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    # === 4. Constraint Norm GIF ===
    def plot_constraint_norm(i):
        plt.figure(figsize=(8, 5))
        plt.plot(
            it_history_ec_ddp[:i],
            constraint_history_ec_ddp[:i],
            marker="o",
            color="red",
            label="EC-DDP",
        )
        plt.plot(
            it_history_ddp[:i],
            constraint_history_ddp[:i],
            marker="o",
            color="blue",
            label="DDP",
        )
        plt.title(r"EC-DDP $||h(x)||$ vs DDP $||h(x)||$")
        plt.xlabel("Iterations")
        plt.ylabel(r"$||h(x)||$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    # === Generate all GIFs ===
    num_frames = len(it_history_ec_ddp)
    create_gif_plot(plot_cost, num_frames, "GIF_cost_progression")
    create_gif_plot(plot_lambdas, num_frames, "GIF_lambda_mu_progression")
    create_gif_plot(plot_parameters, num_frames, "GIF_parameter_progression")
    create_gif_plot(plot_constraint_norm, num_frames, "GIF_constraint_norm")
