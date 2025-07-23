from typing import Optional, Tuple

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive, set_trainable, add_likelihood_noise_cov, assert_params_false
from gpflow.logdensities import multivariate_normal
from check_shapes import check_shapes, inherit_check_shapes
from gpflow import posteriors
from gpflow.base import InputData, MeanAndVariance, RegressionData, TensorData

class LinearMultiFidelityKernel(gpflow.kernels.Kernel):
    """
    Linear Multi-Fidelity Kernel (Kennedy & O’Hagan, 2000).

    This kernel models the high-fidelity function as:

        f_H(x) = ρ * f_L(x) + δ(x)

    where:
        - f_L(x) is a Gaussian process modeling the low-fidelity function.
        - δ(x) is an independent GP modeling discrepancies.
        - ρ is a learnable scaling factor.

    The covariance matrix has a block structure:

        K =
        [  K_LL   K_LH  ]
        [  K_HL   K_HH  ]

    where:
        - K_LL = Covariance matrix for low-fidelity points.
        - K_LH = K_HL^T = Scaled cross-covariance.
        - K_HH = Scaled LF + discrepancy covariance.

    Parameters:
        - kernel_L: GP kernel for the low-fidelity function.
        - kernel_delta: GP kernel for the discrepancy.
        - num_output_dims: The number of independent outputs (Y.shape[1]).
    """

    def __init__(self, kernel_L, kernel_delta, num_output_dims):
        super().__init__()
        self.kernel_L = kernel_L  # Kernel for low-fidelity function
        self.kernel_delta = kernel_delta  # Kernel for discrepancy

        self.rho = gpflow.Parameter(
            np.ones((num_output_dims, 1)), transform=positive()
        )  # Shape (P, 1), separate for each output dim

    def K(self, X, X2=None, ith_output_dim=0):
        """
        Constructs the full covariance matrix for multi-fidelity modeling.
        """
        if X2 is None:
            X2 = X

        # Convert input to TensorFlow tensors
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        X2 = tf.convert_to_tensor(X2, dtype=tf.float64)

        # Extract fidelity indicators
        mask_L = tf.where(X[:, -1] == 0)[:, 0]
        mask_H = tf.where(X[:, -1] == 1)[:, 0]
        mask2_L = tf.where(X2[:, -1] == 0)[:, 0]
        mask2_H = tf.where(X2[:, -1] == 1)[:, 0]

        # Extract LF and HF data
        X_L = tf.gather(X[:, :-1], mask_L)
        X_H = tf.gather(X[:, :-1], mask_H)
        X2_L = tf.gather(X2[:, :-1], mask2_L)
        X2_H = tf.gather(X2[:, :-1], mask2_H)

        # Get the number of output dimensions
        output_dim = self.rho.shape[1]

        # Initialize full covariance matrix
        K_full = tf.zeros((X.shape[0], X2.shape[0]), dtype=tf.float64)

        # Extract index positions for placing the computed covariance submatrices
        indices_LL = tf.stack(tf.meshgrid(mask_L, mask2_L, indexing="ij"), axis=-1)
        indices_LH = tf.stack(tf.meshgrid(mask_L, mask2_H, indexing="ij"), axis=-1)
        indices_HL = tf.stack(tf.meshgrid(mask_H, mask2_L, indexing="ij"), axis=-1)
        indices_HH = tf.stack(tf.meshgrid(mask_H, mask2_H, indexing="ij"), axis=-1)
        # Extract rho for this output dimension
        rho_i = self.rho[ith_output_dim, :]  # Shape: (1,)

        # Compute covariance components
        K_LL = self.kernel_L.K(X_L, X2_L)  # LF covariance
        K_LH = self.kernel_L.K(X_L, X2_H) * rho_i  # LF-HF scaled covariance
        K_HL = self.kernel_L.K(X_H, X2_L) * rho_i  # Transposed scaling
        K_HH = self.kernel_L.K(X_H, X2_H) * (rho_i * rho_i) + self.kernel_delta.K(X_H, X2_H)

        # Apply tensor updates to construct the full covariance matrix
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_LL, [-1, 2]), tf.reshape(K_LL, [-1]))
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_LH, [-1, 2]), tf.reshape(K_LH, [-1]))
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_HL, [-1, 2]), tf.reshape(K_HL, [-1]))
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_HH, [-1, 2]), tf.reshape(K_HH, [-1]))

        return K_full

    def K_diag(self, X, ith_output_dim=0):
        """
        Computes the diagonal elements of the covariance matrix.
        """
        X = tf.convert_to_tensor(X, dtype=tf.float64)

        # Extract LF and HF indices
        mask_L = tf.where(X[:, -1] == 0)[:, 0]
        mask_H = tf.where(X[:, -1] == 1)[:, 0]

        X_L = tf.gather(X[:, :-1], mask_L)
        X_H = tf.gather(X[:, :-1], mask_H)

        # Compute diagonal covariance elements
        K_diag_L = self.kernel_L.K_diag(X_L)

        # Extract rho for this output dimension
        rho_i = self.rho[ith_output_dim, :]  # Shape: (1,)

        # Ensure rho is squared and has correct shape (broadcast correctly)
        rho_sq = tf.reshape(rho_i**2, [-1, 1])  # ✅ Ensure correct shape for multiplication
        K_diag_H = self.kernel_L.K_diag(X_H) * rho_sq + self.kernel_delta.K_diag(X_H)

        # Construct full diagonal vector
        K_diag_full = tf.zeros((X.shape[0],), dtype=tf.float64)

        # Place diagonal values at correct indices
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(mask_L, [-1, 1]), tf.reshape(K_diag_L, [-1]))
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(mask_H, [-1, 1]), tf.reshape(K_diag_H, [-1]))

        return K_diag_full

class MultiFidelityGPModel(gpflow.models.GPR):
    """
    Gaussian Process model for multi-fidelity learning with multiple output dimensions.

    This model ensures:
    - Each output dimension has an independent `rho[i]` parameter.
    - The kernels `kernel_L` and `kernel_delta` are shared across output dimensions.
    - Training correctly propagates per-output fidelity while using the correct `rho[i]`.
    """

    def __init__(self, X, Y, kernel_L, kernel_delta):
        num_output_dims = Y.shape[1]  # Number of independent outputs
        self.kernel = LinearMultiFidelityKernel(kernel_L, kernel_delta, num_output_dims)
        likelihood = gpflow.likelihoods.Gaussian(variance=1e-3)

        super().__init__((X, Y), kernel=self.kernel, likelihood=likelihood)
        set_trainable(self.likelihood.variance, False)

        self.num_output_dims = num_output_dims
    
    # def log_marginal_likelihood(self) -> tf.Tensor:
    #     """
    #     Computes the total log marginal likelihood across all output dimensions.
    #     Uses vectorized operations for correct backpropagation.
    #     """
    #     X, Y = self.data  # Training data

    #     # Compute kernel for **all outputs at once** by broadcasting ith_output_dim
    #     K_all = tf.map_fn(lambda i: self.kernel.K(X, X, ith_output_dim=i), tf.range(self.num_output_dims), dtype=tf.float64)

    #     # Ensure likelihood variance is correctly broadcasted
    #     noise_term = self.likelihood.variance * tf.eye(tf.shape(K_all)[0], batch_shape=[self.num_output_dims], dtype=tf.float64)

    #     # Add noise to each output dim independently
    #     K_with_noise = K_all + noise_term  # ✅ Correct broadcasting

    #     eigenvalues = tf.linalg.eigvalsh(K_with_noise)  # Compute eigenvalues
    #     print(f"🔍 Min Eigenvalue: {tf.reduce_min(eigenvalues)}, Max: {tf.reduce_max(eigenvalues)}")
    #     print(f"🔹 Any NaN in K_with_noise? {tf.reduce_any(tf.math.is_nan(K_with_noise))}")
    #     # Compute Cholesky decomposition for all output dims at once
    #     L_all = tf.map_fn(lambda K: tf.linalg.cholesky(K), K_with_noise, dtype=tf.float64)

    #     # Compute mean function across all output dimensions
    #     mean_all = self.mean_function(X)  # Shape: (N, P)

    #     # Compute log probability in vectorized way
    #     log_probs = tf.vectorized_map(
    #         lambda i: multivariate_normal(tf.expand_dims(Y[:, i], axis=-1), tf.expand_dims(mean_all[:, i], axis=-1), L_all[:, :, i]), 
    #         tf.range(self.num_output_dims)
    #     )
    #     return tf.reduce_sum(log_probs)  # ✅ Correct gradient tracking

    def optimize(self, max_iters=1000, learning_rate=0.01, use_adam=True, unfix_noise_after=500):
        """
        Optimizes the model while ensuring proper multi-output learning.

        - Handles per-output `rho[i]` updates separately.
        - Uses Adam or Scipy L-BFGS with noise-fixing for stability.
        """
        # print(f"🔹 Pre-Optimization rho: {self.kernel.rho.numpy()}")
        self.loss_history = []

        if use_adam:
            optimizer = tf.optimizers.Adam(learning_rate)

            @tf.function
            def optimization_step():
                with tf.GradientTape() as tape:
                    loss = -self.log_marginal_likelihood()  # Maximize log-marginal likelihood
                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                return loss  # Track loss

            print("Optimizing with Adam...")
            for i in range(max_iters):
                loss = optimization_step()
                self.loss_history.append(loss.numpy())  # Store loss history

                if i == unfix_noise_after:
                    print(f"🔹 Unfixing noise at iteration {i}")
                    set_trainable(self.likelihood.variance, True)

                if i % 100 == 0:
                    print(f"🔹 Iteration {i}: Loss = {-loss.numpy()}")

        else:
            print("Optimizing with L-BFGS (Scipy)...")
            
            def loss_closure():
                loss = -self.log_marginal_likelihood()
                return loss

            scipy_optimizer = gpflow.optimizers.Scipy()
            scipy_optimizer.minimize(loss_closure, self.trainable_variables, options={"maxiter": max_iters})

            set_trainable(self.likelihood.variance, True)
            scipy_optimizer.minimize(loss_closure, self.trainable_variables, options={"maxiter": max_iters})

    # @inherit_check_shapes
    # def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
    #     """
    #     Predict function for Multi-Output Multi-Fidelity GP.

    #     - Computes predictions separately for each output dimension.
    #     - Uses `ith_output_dim` to select the correct fidelity scaling.
    #     - Ensures `knn` has correct shape when `full_cov=False`.
    #     """
    #     assert_params_false(self.predict_f, full_output_cov=full_output_cov)

    #     X, Y = self.data  # Training data
    #     err = Y - self.mean_function(X)  # Compute mean-subtracted training targets

    #     f_means = []
    #     f_vars = []

    #     for i in range(self.num_output_dims):
    #         print(f"🔹 Predicting for output dimension {i}")

    #         # Compute covariance matrices per output dimension
    #         kmm = self.kernel.K(X, X, ith_output_dim=i)
    #         knn = self.kernel.K(Xnew, Xnew, ith_output_dim=i)
            
    #         # ✅ Fix shape issue: Extract diagonal when full_cov=False
    #         if not full_cov:
    #             knn = tf.linalg.diag_part(knn)  # Extract diagonal, making shape (N_test,)

    #         kmn = self.kernel.K(X, Xnew, ith_output_dim=i)

    #         print(f"   🔹 Shapes - kmm: {kmm.shape}, knn: {knn.shape}, kmn: {kmn.shape}")
    #         print(f"   🔹 Shapes - err[:, {i}:{i+1}]: {err[:, i:i+1].shape}")

    #         # Add likelihood noise to K_MM for numerical stability
    #         kmm_plus_s = add_likelihood_noise_cov(kmm, self.likelihood, X)

    #         # Compute conditional mean & variance
    #         conditional = gpflow.conditionals.base_conditional
    #         f_mean_zero, f_var = conditional(
    #             kmn, kmm_plus_s, knn, err[:, i:i+1], full_cov=full_cov, white=False
    #         )

    #         f_means.append(f_mean_zero)
    #         f_vars.append(f_var)

    #     # ✅ Ensure correct shape for multi-output GP
    #     mean_pred = tf.concat(f_means, axis=1)  # Stack results for all outputs
    #     var_pred = tf.concat(f_vars, axis=1)

    #     print(f"✅ Final prediction shapes: mean {mean_pred.shape}, var {var_pred.shape}")
    #     return mean_pred, var_pred