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
        lf_kernel_factories,  # list of callables to make base kernels
        delta_kernel,
        window_fraction=0.4,
        scale=0.2,
    ):
        self.num_outputs = num_outputs
        self.num_LF = len(latent_dims_list)

        # 1. Create LinearCoregionalization kernel for each LF source
        lf_kernels = []
        for i, L_i in enumerate(latent_dims_list):
            base_kernels = [lf_kernel_factories[i]() for _ in range(L_i)]
            W_i = gpflow.Parameter(
                initialize_W(num_outputs, L_i, window_fraction, scale),
                trainable=True,
            )
            LMC_i = LinearCoregionalization(base_kernels, W=W_i)
            lf_kernels.append(LMC_i)

        # 2. Construct GraphMultiFidelityKernel (no outer LMC!)
        graph_kernel = GraphMultiFidelityKernel(
            kernel_L_list=lf_kernels,
            kernel_delta=delta_kernel,
            num_output_dims=num_outputs,
        )

        # 3. Shared inducing variables across L total latents
        inducing_variable = SharedIndependentInducingVariables(InducingPoints(Z))

        # 4. Variational parameters
        total_latents = sum(latent_dims_list)
        q_mu = np.zeros((Z.shape[0], total_latents))
        q_sqrt = np.repeat(np.eye(Z.shape[0])[None, ...], total_latents, axis=0) * 0.1

        # 5. Final SVGP
        likelihood = Gaussian()
        super().__init__(
            kernel=graph_kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
        )

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
        schedule = tf.keras.optimizers.schedules.CosineDecay(initial_lr, max_iters)
        optimizer = tf.optimizers.Adam(schedule)
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
                print(f"🔹 Unfixing noise variance at iteration {i}")
                gpflow.utilities.set_trainable(self.likelihood.variance, True)

            if i % 20 == 0:
                print(f"🔹 Iteration {i}: ELBO = {-self.elbo((X, Y)).numpy()}")

    def save_model(self, filename="graph_latent_mf_svgp.pkl"):
        """Saves the trained GraphLatentMF_SVGP model."""
        params = gpflow.utilities.parameter_dict(self)
        with open(filename, "wb") as f:
            pickle.dump(params, f)
        print(f"✅ Model saved to {filename}")

    @staticmethod
    def load_model(filename, constructor_args: tuple):
        """
        Loads the GraphLatentMF_SVGP model from disk.

        Parameters:
            filename (str): Path to saved model pickle.
            constructor_args (tuple): Arguments used to initialize the model, e.g.
                (X, Y, graph_kernel, num_latents, Z)

        Returns:
            GraphLatentMF_SVGP instance with loaded parameters.
        """
        with open(filename, "rb") as f:
            params = pickle.load(f)

        model = GraphLatentMF_SVGP(*constructor_args)
        gpflow.utilities.multiple_assign(model, params)
        print(f"✅ Model loaded from {filename}")
        return model