import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
from sklearn.cluster import KMeans
from mfgpflow.data_loader import StellarMassFunctions
from mfgpflow.linear_svgp import LatentMFCoregionalizationSVGP

def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Fidelity GP Model with CAMELS Data")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--output_folder", type=str, default="./output", help="Folder to save model and figures")
    parser.add_argument("--num_latents", type=int, default=5, help="Number of latent GPs")
    parser.add_argument("--num_inducing", type=int, default=100, help="Number of inducing points")
    parser.add_argument("--max_iters", type=int, default=2000, help="Number of training iterations")
    return parser.parse_args()

def generate_data(folder):
    return StellarMassFunctions(folder=folder)

def save_txt(data, filename):
    np.savetxt(filename, data, fmt='%e')

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    fig_folder = os.path.join(args.output_folder, "figures")
    model_folder = os.path.join(args.output_folder, "models")
    os.makedirs(fig_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    
    data = generate_data(args.data_folder)
    log10_mass_bins = np.array([8.15, 8.45, 8.75, 9.05, 9.35, 9.65, 9.95, 10.25, 10.55, 10.85])
    
    X_LF, Y_LF = data.X_train_norm[0], data.Y_train_norm_log10[0]
    X_HF, Y_HF = data.X_train_norm[1], data.Y_train_norm_log10[1]
    X_test_HF, Y_test_HF = data.X_test_norm[0], data.Y_test_log10[0]
    
    n_LF, n_HF = X_LF.shape[0], X_HF.shape[0]
    output_dim = Y_LF.shape[1]
    
    X_LF_aug = np.hstack([X_LF, np.zeros((n_LF, 1))])
    X_HF_aug = np.hstack([X_HF, np.ones((n_HF, 1))])
    X_train = np.vstack([X_LF_aug, X_HF_aug])
    Y_train = np.vstack([Y_LF, Y_HF])
    
    kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(6), variance=1.0)
    kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(6), variance=1.0)
    
    kmeans = KMeans(n_clusters=args.num_inducing, random_state=42).fit(X_train)
    Z_init = kmeans.cluster_centers_
    
    mf_gp = LatentMFCoregionalizationSVGP(
        X_train, Y_train, kernel_L, kernel_delta, num_outputs=output_dim, num_latents=args.num_latents, Z=Z_init
    )

    mf_gp.optimize((X_train, Y_train), max_iters=args.max_iters, initial_lr=0.1, unfix_noise_after=500)
    
    num_test_points = X_test_HF.shape[0]
    X_test_HF_aug = np.hstack([X_test_HF, np.ones((num_test_points, 1))])
    mean_pred, var_pred = mf_gp.predict_f(X_test_HF_aug)
    residuals = mean_pred.numpy() - Y_test_HF
    relative_error = np.abs(10**mean_pred / 10**Y_test_HF - 1)

    save_txt(mean_pred, os.path.join(model_folder, "predictions.txt"))    
    save_txt(var_pred.numpy(), os.path.join(model_folder, "variances.txt"))
    save_txt(10**mean_pred / 10**Y_test_HF, os.path.join(model_folder, "pred_over_exact.txt"))

    # training results
    plt.imshow(mf_gp.kernel.W.numpy(), aspect="auto")
    plt.colorbar()
    plt.title("Learned W Matrix")
    plt.ylabel("Output Dimension")
    plt.xlabel("Latent Dimension")
    plt.savefig(os.path.join(fig_folder, "W_matrix.png"))
    plt.clf()
    plt.close()
    # loss history
    plt.plot(mf_gp.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.savefig(os.path.join(fig_folder, "loss_history.png"))
    plt.clf()
    plt.close()
    # save model
    # You can save the model for later use
    # There are other ways, e.g., checkpoint manager, but this is the simplest
    mf_gp.save_model(os.path.join(model_folder, "mf_gp_model.pkl"))

    ####### Hyperparameters #######
    num_outputs = len(mf_gp.kernel.kernels)

    rho_values = []
    lengthscale_values = []
    lengthscale_delta_values = []

    for i in range(num_outputs):
        rho_values.append(mf_gp.kernel.kernels[i].rho.numpy()[0])
        lengthscale_values.append(mf_gp.kernel.kernels[i].kernel_L.lengthscales.numpy())
        lengthscale_delta_values.append(mf_gp.kernel.kernels[i].kernel_delta.lengthscales.numpy())


    # Extract the learned parameters
    param_names = [
        "Omega_m",
        "sigma_8",
        "A_SN1",
        "A_AGN1",
        "A_SN2",
        "A_AGN2",
    ]

    plt.plot(range(args.num_latents), rho_values)
    plt.xlabel("Latent Dimension")
    plt.ylabel("$\\rho$")
    plt.savefig(os.path.join(fig_folder, "rho_values.png"))
    plt.clf()
    plt.close()
    # Lengthscale values
    for i,param_name in enumerate(param_names):
        plt.plot(range(args.num_latents), np.array(lengthscale_values)[:, i], label=param_name)
    plt.xlabel(r"Latent Dimension")
    plt.legend()
    plt.ylabel(r"$\ell$")
    plt.savefig(os.path.join(fig_folder, "lengthscale_values.png"))
    plt.clf()
    plt.close()
    # Lengthscale delta values
    for i,param_name in enumerate(param_names):
        plt.plot(range(args.num_latents), np.array(lengthscale_delta_values)[:, i], label=param_name)
    plt.xlabel(r"Latent Dimension")
    plt.ylabel(r"$\ell_{\delta}$")
    plt.legend()
    plt.savefig(os.path.join(fig_folder, "lengthscale_delta_values.png"))
    plt.clf()
    plt.close()

    # Projected rho values
    projected_rho = mf_gp.kernel.W.numpy() @ np.array(rho_values)
    plt.plot(log10_mass_bins, projected_rho[:, 0])
    plt.xlabel(r"$\log_{10}M_{\star}$")
    plt.ylabel(r"$\rho$")
    plt.savefig(os.path.join(fig_folder, "rho_values_projected.png"))
    plt.clf()
    plt.close()

    # Projected lengthscale values
    for i,param_name in enumerate(param_names):
        plt.plot(log10_mass_bins, mf_gp.kernel.W.numpy() @ np.array(lengthscale_values)[:, i], label=param_name)
    plt.xlabel(r"$\log_{10}M_{\star}$")
    plt.legend()
    plt.ylabel(r"$\ell$")
    plt.savefig(os.path.join(fig_folder, "lengthscale_values_projected.png"))
    plt.clf()
    plt.close()

    for i,param_name in enumerate(param_names):
        plt.plot(log10_mass_bins, mf_gp.kernel.W.numpy() @ np.array(lengthscale_delta_values)[:, i], label=param_name)
    plt.xlabel(r"$\log_{10}M_{\star}$")
    plt.ylabel(r"$\ell_{\delta}$")
    plt.legend()
    plt.savefig(os.path.join(fig_folder, "lengthscale_delta_values_projected.png"))
    plt.clf()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.errorbar(log10_mass_bins, Y_test_HF[0], yerr=np.sqrt(var_pred.numpy()[0]), fmt='o', label="True")
    plt.plot(log10_mass_bins, mean_pred.numpy()[0], 'r--', label="Predicted")
    plt.xlabel(r"$\log_{10}M_{\star}$")
    plt.ylabel("Prediction")
    plt.legend()
    plt.title("Multi-Fidelity GP Prediction (First Test Point)")
    plt.savefig(os.path.join(fig_folder, "Predict.png"))
    plt.clf()
    plt.close()

    # 🔹 8️⃣ Residual Plot
    plt.figure(figsize=(10, 5))
    plt.plot(log10_mass_bins, residuals[0], 'bo-', label="Residuals")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(r"$\log_{10}M_{\star}$")
    plt.ylabel("Residual")
    plt.legend()
    plt.title("Prediction Residuals (First Test Point)")
    plt.savefig(os.path.join(fig_folder, "Residual.png"))
    plt.clf()
    plt.close()

    plt.plot(log10_mass_bins, (mean_pred - Y_test_HF).numpy().T);
    plt.xlabel(r"$\log_{10}M_{\star}$")
    plt.ylabel(r"$\Phi_{\mathrm{pred}}/\Phi_{\mathrm{true}}$")
    plt.savefig(os.path.join(fig_folder, "pred_exact.png"))
    plt.clf()
    plt.close()

    # absolute error
    relative_error = np.abs(10**mean_pred / 10**Y_test_HF - 1)
    plt.plot(log10_mass_bins, relative_error.mean(axis=0));
    plt.fill_between(log10_mass_bins, relative_error.min(axis=0), relative_error.max(axis=0), alpha=0.3)
    plt.xlabel(r"$\log_{10}M_{\star}$")
    plt.ylabel(r"$|\Phi_{\mathrm{pred}}-\Phi_{\mathrm{true}}|$")
    plt.savefig(os.path.join(fig_folder, "absolute_error.png"))
    plt.clf()
    plt.close()

    print(f"Model, figures, and data saved in {args.output_folder}")

if __name__ == "__main__":
    main()
