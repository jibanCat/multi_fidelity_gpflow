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
from .graph import GraphMultiFidelityKernel  # Your existing LinearMultiFidelityKernel

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


class GraphLatentMF_SVGP(SVGP):
    def __init__(
        self,
        X,
        Y,
        num_outputs,
        latent_dims_list,
        Z,
        lf_kernel_factories,
        delta_kernel,
        window_fraction=0.4,
        scale=0.2,
    ):
        """
        Latent Graph Multi-Fidelity SVGP model:
        - f_H(x) = sum_i rho_i f_{L_i}(x) + delta(x)
        - Each f_{L_i}(x) = LinearCoregionalization(W_i @ latent GPs)
        """
        self.num_outputs = num_outputs
        self.num_LF = len(latent_dims_list)

        # 1. Construct each f_{L_i}(x) = LinearCoregionalization(W_i @ latent GPs)
        lf_multi_output_kernels = []
        total_latents = 0

        for i, L_i in enumerate(latent_dims_list):
            base_kernels_i = [lf_kernel_factories[i]() for _ in range(L_i)]
            W_i = gpflow.Parameter(
                initialize_W(num_outputs, L_i, window_fraction, scale), trainable=True
            )
            LMC_i = LinearCoregionalization(base_kernels_i, W=W_i)
            lf_multi_output_kernels.append(LMC_i)
            total_latents += L_i

        # 2. Build Graph MF Kernel
        graph_kernel = GraphMultiFidelityKernel(
            kernel_L_list=lf_multi_output_kernels,
            kernel_delta=delta_kernel,
            num_output_dims=num_outputs,
        )

        # 3. Inducing variable: shared across outputs
        inducing_variable = SharedIndependentInducingVariables(InducingPoints(Z))

        # 4. Variational parameters
        q_mu = np.zeros((Z.shape[0], total_latents))
        q_sqrt = np.repeat(
            np.eye(Z.shape[0])[None, ...], total_latents, axis=0
        ) * 0.1

        # 5. Final SVGP call
        likelihood = Gaussian()
        super().__init__(
            kernel=graph_kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
        )

    def optimize(self, data, max_iters=1000, initial_lr=0.01, unfix_noise_after=500):
        X, Y = data
        optimizer = tf.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(initial_lr, max_iters))
        self.loss_history = []

        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                loss = -self.elbo((X, Y))
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss

        print("🔹 Optimizing...")
        for i in range(max_iters):
            loss = optimization_step()
            self.loss_history.append(loss.numpy())

            if i == unfix_noise_after:
                print("🔹 Unfixing noise variance at iteration", i)
                gpflow.utilities.set_trainable(self.likelihood.variance, True)

            if i % 10 == 0:
                print(f"🔹 Iteration {i}: ELBO = {-self.elbo((X, Y)).numpy()}")

    def save_model(self, filename="graph_latent_mf_svgp.pkl"):
        import pickle
        params = gpflow.utilities.parameter_dict(self)
        with open(filename, "wb") as f:
            pickle.dump(params, f)
        print(f"✅ Model saved to {filename}")

    @staticmethod
    def load_model(filename, *args):
        import pickle
        with open(filename, "rb") as f:
            params = pickle.load(f)
        model = GraphLatentMF_SVGP(*args)
        gpflow.utilities.multiple_assign(model, params)
        print(f"✅ Model loaded from {filename}")
        return model
