import gpflow
import tensorflow as tf
import numpy as np
import pickle
from copy import deepcopy

from sklearn.cluster import KMeans
from gpflow.kernels import SquaredExponential, Linear
from gpflow.models import SVGP
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints, SharedIndependentInducingVariables
from gpflow.kernels import LinearCoregionalization
from .linear import LinearMultiFidelityKernel  # Your existing LinearMultiFidelityKernel

def initialize_W(output_dim, num_latents, window_fraction=0.3, scale=0.1):
    """
    Initialize W with a localized diagonal structure ensuring full output coverage.

    - Each latent GP influences multiple nearby outputs.
    - Overlapping mappings provide smooth transitions.
    - Weaker prior influence allows the model to adjust.

    Parameters:
        output_dim (int): Number of output bins.
        num_latents (int): Number of latent GPs.
        window_fraction (float): Fraction of outputs each latent covers (~0.3 is a good default).
        scale (float): Scaling factor for trainability.

    Returns:
        W_init (np.ndarray): Initialized coregionalization matrix (output_dim, num_latents).
    """
    W_init = np.zeros((output_dim, num_latents))

    # Define coverage for each latent GP
    window_size = max(int(output_dim * window_fraction), 2)  # Ensure at least 2 output bins
    stride = max(output_dim // (num_latents - 1), 1)  # Spread latents evenly, ensuring full coverage

    for j in range(num_latents):
        center = min(int(j * stride), output_dim - 1)  # Ensure last latent maps fully
        for i in range(output_dim):
            distance = abs(i - center)
            if distance < window_size / 2:
                W_init[i, j] = np.exp(-0.1 * distance)  # Weaker exponential decay for flexibility

    return W_init * scale  # Scale for trainability

class LatentMFCoregionalizationSVGP(SVGP):
    """
    Multi-Fidelity Sparse Variational GP with:
    - **Inference in Latent GP Space** (L < P) for better scalability.
    - **LinearCoregionalization** for capturing multi-output correlation.
    - **Inducing Variable Selection with KMeans** for robust initialization.
    - **Stable Optimization** using better parameter initialization.
    """

    def __init__(self, X, Y, kernel_L, kernel_delta, num_latents, num_outputs, Z, window_fraction=0.4, scale=0.2):
        """
        Initializes the Multi-Fidelity SVGP model.

        Parameters:
            X (np.ndarray): Input data `(N, D)`, where `D` is the input dimension.
            Y (np.ndarray): Output data `(N, P)`, where `P` is the number of output bins.
            kernel_L (gpflow.kernels.Kernel): Kernel for low-fidelity (LF) data.
            kernel_delta (gpflow.kernels.Kernel): Kernel for high-fidelity (HF) discrepancy.
            num_latents (int): Number of latent GPs `(L)`, typically `L < P`.
            num_outputs (int): Number of output dimensions `(P)`, e.g., 49 bins.
            Z (np.ndarray): Inducing point locations `(M, D)`, where `M` is the number of inducing points.
            window_fraction (float): Fraction of total outputs each latent covers.
            scale (float): Scaling factor for initial weights.
        """
        self.num_outputs = num_outputs
        self.num_latents = num_latents

        # ✅ Multi-Fidelity Kernel
        # mf_kernel = LinearMultiFidelityKernel(kernel_L, kernel_delta, num_latents)

        # ✅ Initialize W (P × L) with structured correlations
        W_init = initialize_W(num_outputs, num_latents, window_fraction=window_fraction, scale=scale)
        W = gpflow.Parameter(W_init)  # Learnable mixing matrix

        # ✅ Use LinearCoregionalization for Multi-Output GP
        # kernel_list = [mf_kernel for _ in range(num_latents)]
        kernel_list = [LinearMultiFidelityKernel(deepcopy(kernel_L), deepcopy(kernel_delta), num_output_dims=1) for _ in range(num_latents)]
        multioutput_kernel = LinearCoregionalization(kernel_list, W=W)

        # ✅ Use KMeans to Find Good Inducing Points
        kmeans = KMeans(n_clusters=Z.shape[0], random_state=42).fit(X)
        Z_init = kmeans.cluster_centers_
        print("🔹 KMeans Inducing Points:", Z_init)
        inducing_variable = SharedIndependentInducingVariables(InducingPoints(Z_init))

        # ✅ Variational Parameters Initialization
        q_mu = np.zeros((Z.shape[0], num_latents))  # M × L
        q_sqrt = np.repeat(np.eye(Z.shape[0])[None, ...], num_latents, axis=0) * 0.1  # L × M × M, scaled down

        # ✅ Define SVGP Model
        likelihood = Gaussian()
        super().__init__(kernel=multioutput_kernel, likelihood=likelihood,
                         inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt)

    def optimize(self, data, max_iters=10000, initial_lr=0.005, unfix_noise_after=5000):
        """
        Optimizes the model using Adam with cosine decay.

        Parameters:
            data (tuple): Tuple `(X, Y)`, where:
                - `X` is the training input `(N, D)`.
                - `Y` is the training output `(N, P)`.
            max_iters (int): Maximum number of optimization iterations.
            initial_lr (float): Initial learning rate.
            unfix_noise_after (int): Iteration at which to allow noise variance to be learned.
        """
        X, Y = data
        optimizer = tf.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(initial_lr, max_iters))
        self.loss_history = []

        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                loss = -self.elbo((X, Y))  # ✅ GPflow’s ELBO computation
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss  # Track loss

        print("🔹 Optimizing...")
        for i in range(max_iters):
            loss = optimization_step()
            self.loss_history.append(loss.numpy())  # Store loss history

            if i == unfix_noise_after:
                print("🔹 Unfixing noise variance at iteration", i)
                gpflow.utilities.set_trainable(self.likelihood.variance, True)
            if i % 10 == 0:
                print(f"🔹 Iteration {i}: ELBO = {-self.elbo((X, Y)).numpy()}")

    def save_model(self, filename="latent_mf_svgp.pkl"):
        """Saves the trained SVGP model."""
        params = gpflow.utilities.parameter_dict(self)
        with open(filename, "wb") as f:
            pickle.dump(params, f)
        print(f"✅ Model saved to {filename}")

    @staticmethod
    def load_model(filename, *args):
        """Loads an SVGP model from a saved file."""
        with open(filename, "rb") as f:
            params = pickle.load(f)
        model = LatentMFCoregionalizationSVGP(*args)
        gpflow.utilities.multiple_assign(model, params)
        print(f"✅ Model loaded from {filename}")
        return model