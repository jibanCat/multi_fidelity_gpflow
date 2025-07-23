import gpflow
import tensorflow as tf
import numpy as np
import pickle
from copy import deepcopy
from gpflow import Parameter
from gpflow.utilities import positive

from sklearn.cluster import KMeans
from gpflow.models import SVGP
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints, SharedIndependentInducingVariables
from gpflow.kernels import LinearCoregionalization
from .linear import LinearMultiFidelityKernel

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

    def __init__(self, X, Y, kernel_L, kernel_delta, num_latents, num_inducing, num_outputs, heterosed=False, window_fraction=0.4, scale=0.2):
        """
        Initializes the Multi-Fidelity SVGP model.
        Note: All the data (X, Y or even the paramterts in kernel_L and kernel_delta) are 
        expected to be in float64.
        Parameters:
            X (np.ndarray): Input data `(N, D)`, where `D` is the input dimension.
            Y (np.ndarray): Output data
                if heterosed==False: shape is `(N, P)`, where `P` is the number of output bins.
                if heterosed==True: shape is `(N, 2*P)`, where the first `P` columns are the observed outputs
                and the next `P` columns are the uncertainties. Loo at at the `HeteroscedasticGaussian` class for more details.
            kernel_L (gpflow.kernels.Kernel): Kernel for low-fidelity (LF) data.
            kernel_delta (gpflow.kernels.Kernel): Kernel for high-fidelity (HF) discrepancy.
            num_latents (int): Number of latent GPs `(L)`, typically `L < P`.
            num_inducing (int): Number of inducing points `(M)`.
            num_outputs (int): Number of output dimensions `(P)`, e.g., 49 bins.
            heterosed (bool): If True, uses heteroscedastic likelihood, i.e. the output for each samplehas a given
                uncertainty. If False, uses a homoscedastic likelihood.
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
        self.kernel = LinearCoregionalization(kernel_list, W=W)

        # ✅ Use KMeans to Find Good Inducing Points
        kmeans = KMeans(n_clusters=num_inducing, random_state=42).fit(X)
        Z_init = kmeans.cluster_centers_
        #print("🔹 KMeans Inducing Points:", Z_init)
        inducing_variable = SharedIndependentInducingVariables(InducingPoints(Z_init))

        # ✅ Define SVGP Model
        # one learnable parameter for the noise variance in the likelihood
        variance = np.array([1.0], dtype=np.float64)
        if heterosed:
            self.likelihood = HeteroscedasticGaussian(variance=variance)
        else:
            self.likelihood = Gaussian(variance=variance)  
        super().__init__( kernel=self.kernel,
                         likelihood=self.likelihood,
                         inducing_variable=inducing_variable,
                         num_latent_gps=num_latents,
                         num_data=X.shape[0],
                         mean_function=None
                        )

        self.loss_history = []

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


        # Warm up the TFP cache by calling elbo once outside the tf.function. Otherwise the code fails 
        # for the heteroscedastic likelihood.
        _ = self.elbo((X, Y))

        # Define a reusable tf.function that accepts X and Y as arguments.
        @tf.function
        def optimization_step(X, Y):
            with tf.GradientTape() as tape:
                loss = -self.elbo((X, Y))
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss

        # Run the optimization loop, reusing the same tf.function.
        for i in range(len(self.loss_history), max_iters):
            loss = optimization_step(X, Y)
            self.loss_history.append(loss.numpy())
            if i%100 == 0:
                print(f"🔹 Iteration {i}: ELBO = {-self.elbo((X, Y)).numpy()}", flush=True)

            # Optionally, set the likelihood's noise variance to be trainable at a given iteration.
            if i == unfix_noise_after:   
                self.likelihood.variance.trainable = True


    def save_model(self, filename="latent_mf_svgp.pkl"):
        """Saves the trained SVGP model."""
        params = gpflow.utilities.parameter_dict(self)
        with open(filename, "wb") as f:
            pickle.dump(params, f)
        #print(f"✅ Model saved to {filename}")

    @staticmethod
    def load_model(filename, *args):
        """Loads an SVGP model from a saved file."""
        with open(filename, "rb") as f:
            params = pickle.load(f)
        model = LatentMFCoregionalizationSVGP(*args)
        gpflow.utilities.multiple_assign(model, params)
        #print(f"✅ Model loaded from {filename}")
        return model
    
class HeteroscedasticGaussian(gpflow.likelihoods.Gaussian):
    """
    Gaussian likelihood that incorporates a per-data-point uncertainty for each output.
    
    Instead of passing a tuple, this implementation expects the targets to be a combined tensor:
    
         Y_combined = [Y_obs, Y_unc]
         
    concatenated along the last dimension, so that if Y_obs and Y_unc are each shape [N, P],
    then Y_combined has shape [N, 2*P]. In this likelihood, the effective noise variance is:
    
         effective_variance = self.variance + Y_unc
         
    where self.variance is a (possibly vector-valued) baseline noise parameter.
    """
    def __init__(self, variance):
        # Ensure the variance is wrapped as a trainable parameter with a positivity transform.
        variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive())
        super().__init__(variance=variance)

    def _variational_expectations(self, X, Fmu, Fvar, Y):
        # Fmu and Fvar have shape [N, P] where P is the number of outputs.
        # Y is assumed to have shape [N, 2*P], with the first P columns for Y_obs and the next P for Y_unc.
        P = Fmu.shape[-1]
        Y_obs = Y[:, :P]
        Y_unc = Y[:, P:]
        
        # Cast to correct type.
        Y_obs = tf.cast(Y_obs, Fmu.dtype)
        Y_unc = tf.cast(Y_unc, Fmu.dtype)

        # Make sure to cast but do NOT override self.variance.
        var = tf.cast(self.variance, Fmu.dtype)
        
        # Compute the effective noise variance per data point and output.
        effective_variance = var + Y_unc  # [N, P]
        
        # Standard variational expectations of a Gaussian likelihood:
        ve = -0.5 * tf.math.log(2.0 * np.float64(np.pi)) \
             - 0.5 * tf.math.log(effective_variance) \
             - 0.5 * ((Y_obs - Fmu) ** 2 + Fvar) / effective_variance
        
        # Sum over outputs to produce a [N]-shaped tensor.
        return tf.reduce_sum(ve, axis=-1)