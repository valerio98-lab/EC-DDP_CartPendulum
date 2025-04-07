import matplotlib.pyplot as plt
import numpy as np

cost_history_ddp = np.load("vectors_for_plots/cost_history_ddp.npy")
constraint_history_ddp = np.load("vectors_for_plots/constraint_history_ddp.npy")
it_history_ddp = np.load("vectors_for_plots/it_history_ddp.npy")
cost_history_ec_ddp = np.load("vectors_for_plots/cost_history_ec_ddp.npy")
constraint_history_ec_ddp = np.load("vectors_for_plots/constraint_history_ec_ddp.npy")
it_history_ec_ddp = np.load("vectors_for_plots/it_history_ec_ddp.npy")
eta_history = np.load("vectors_for_pparameters/eta_history.npy")
infinite_norm_lag_history = np.load("vectors_for_pparameters/infinite_norm_lag_history.npy")
mu_history = np.load("vectors_for_pparameters/mu_history.npy")
omega_history = np.load("vectors_for_pparameters/omega_history.npy")
lambda_history = np.load("vectors_for_pparameters/lambda_history.npy")

print(lambda_history.shape)

print (it_history_ddp.shape)
print(it_history_ec_ddp.shape)

# Create the plot
plt.figure(figsize=(8, 5))

# Plot both functions
plt.plot(it_history_ec_ddp,lambda_history[:,0], linestyle="-", marker="o", color="blue", label="lambda_1")
plt.plot(it_history_ec_ddp, lambda_history[:,1], linestyle="-", marker="o", color="red", label="lambda_2")
plt.plot(it_history_ec_ddp, lambda_history[:,2], linestyle="-", marker="o", color="green", label="lambda_3")
plt.plot(it_history_ec_ddp, lambda_history[:,3], linestyle="-", marker="o", color="purple", label="lambda_4")
plt.plot(it_history_ec_ddp, mu_history, linestyle="-", marker="o", color="orange", label="mu")

# Labels and title
plt.xlabel("Iterations")
plt.ylabel("Lambda, mu")
plt.title("Lambda and mu values over iterations")

# Legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
