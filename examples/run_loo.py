import argparse
import os
import numpy as np
import gpflow
import tensorflow as tf
from sklearn.cluster import KMeans
from mfgpflow.data_loader import StellarMassFunctions
from mfgpflow.linear_svgp import LatentMFCoregionalizationSVGP

def parse_args():
    """Parse command-line arguments for testing emulation error."""
    parser = argparse.ArgumentParser(description="Test Emulation Error as a Function of LF and HF Counts")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--output_folder", type=str, default="./results", help="Folder to save results")
    parser.add_argument("--num_latents", type=int, default=5, help="Number of latent GPs")
    parser.add_argument("--num_inducing", type=int, default=100, help="Number of inducing points")
    parser.add_argument("--max_iters", type=int, default=2000, help="Number of training iterations")
    return parser.parse_args()

def save_txt(data, filename):
    """Save numerical data to a text file."""
    np.savetxt(filename, data, fmt='%e')

def run_experiment(data, num_LF, num_HF, holdout_idx, output_folder, num_latents, num_inducing, max_iters):
    """Run a single experiment with specified LF and HF counts, holding out one HF sample.
    
    Args:
        data (StellarMassFunctions): The dataset object containing training data.
        num_LF (int): Number of LF samples to use.
        num_HF (int): Number of HF samples to use.
        holdout_idx (int): Index of the HF sample to hold out.
        output_folder (str): Directory to save results.
        num_latents (int): Number of latent GPs.
        num_inducing (int): Number of inducing points.
        max_iters (int): Number of training iterations.
    """
    
    X_LF, Y_LF = data.X_train_norm[0][:num_LF], data.Y_train_norm_log10[0][:num_LF]
    X_HF, Y_HF = np.delete(data.X_train_norm[1], holdout_idx, axis=0), np.delete(data.Y_train_norm_log10[1], holdout_idx, axis=0)
    X_test_HF, Y_test_HF = data.X_train_norm[1][holdout_idx:holdout_idx+1], data.Y_train_norm_log10[1][holdout_idx:holdout_idx+1]
    
    X_LF_aug = np.hstack([X_LF, np.zeros((X_LF.shape[0], 1))])
    X_HF_aug = np.hstack([X_HF, np.ones((X_HF.shape[0], 1))])
    X_train = np.vstack([X_LF_aug, X_HF_aug])
    Y_train = np.vstack([Y_LF, Y_HF])
    
    kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(6), variance=1.0)
    kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(6), variance=1.0)
    
    kmeans = KMeans(n_clusters=num_inducing, random_state=42).fit(X_train)
    Z_init = kmeans.cluster_centers_
    
    mf_gp = LatentMFCoregionalizationSVGP(
        X_train, Y_train, kernel_L, kernel_delta, num_outputs=Y_LF.shape[1], num_latents=num_latents, Z=Z_init
    )
    
    mf_gp.optimize((X_train, Y_train), max_iters=max_iters, initial_lr=0.1, unfix_noise_after=500)
    
    X_test_HF_aug = np.hstack([X_test_HF, np.ones((X_test_HF.shape[0], 1))])
    mean_pred, var_pred = mf_gp.predict_f(X_test_HF_aug)
    residuals = mean_pred.numpy() - Y_test_HF
    relative_error = np.abs(10**mean_pred / 10**Y_test_HF - 1)
    
    exp_folder = os.path.join(output_folder, f"LF{num_LF}_HF{num_HF}_Holdout{holdout_idx}")
    os.makedirs(exp_folder, exist_ok=True)
    
    save_txt(10**mean_pred, os.path.join(exp_folder, "predictions.txt"))
    save_txt(var_pred.numpy(), os.path.join(exp_folder, "variances.txt"))
    save_txt(10**mean_pred / 10**Y_test_HF, os.path.join(exp_folder, "pred_over_exact.txt"))
    save_txt(residuals, os.path.join(exp_folder, "residuals.txt"))
    save_txt(relative_error, os.path.join(exp_folder, "relative_error.txt"))
    save_txt(np.array([np.mean(residuals**2)]), os.path.join(exp_folder, "mean_squared_error.txt"))

    mf_gp.save(os.path.join(exp_folder, "mf_gp_model"))
    print(f"Experiment {exp_folder} completed.")

def main():
    """Main function to run emulation error testing over various LF and HF counts."""
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    data = StellarMassFunctions(folder=args.data_folder)
    
    num_LF_list = np.arange(100, 1005, 100)
    num_HF_list = np.arange(1, 8)
    holdout_list = np.arange(8)
    
    for num_LF in num_LF_list:
        for num_HF in num_HF_list:
            for holdout_idx in holdout_list:
                if num_HF < len(data.X_train_norm[1]):
                    run_experiment(data, num_LF, num_HF, holdout_idx, args.output_folder, args.num_latents, args.num_inducing, args.max_iters)
    
if __name__ == "__main__":
    main()
