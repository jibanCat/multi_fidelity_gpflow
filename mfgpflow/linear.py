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

class FlatLinearMultiFidelityKernel(gpflow.kernels.Kernel):
    """
    A multifidelity kernel that:
     - interprets X[..., -2] as fidelity (0=LF,1=HF)
     - interprets X[..., -1] as dimension index dim in {0..P-1}
     - uses sub-kernels for LF and discrepancy
     - uses a separate scaling rho[d] for each dimension
     - does *not* rely on ith_output_dim
    """

    def __init__(self, kernel_L, kernel_delta, num_output_dims):
        super().__init__()
        self.kernel_L = kernel_L      # Low-fidelity kernel
        self.kernel_delta = kernel_delta
        # We'll store a separate scale factor for each dimension:
        self.rho = gpflow.Parameter(
            tf.ones([num_output_dims, 1], dtype=tf.float64),
            transform=positive()
        )

        self.num_output_dims = num_output_dims

    def _split_fidelity_dim(self, X):
        """
        Utility: separate 'fidelity' and 'dim_index' from the main features.
        X shape = (N, D+2).
          - X[:, :D]   = actual features
          - X[:, -2]   = fidelity (0 or 1)
          - X[:, -1]   = dimension index
        Returns (X_features, fidelity, dim_index).
        """
        X_features = X[..., :-2]
        fidelity   = X[..., -2]
        dim_index  = X[..., -1]
        return X_features, fidelity, dim_index

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        X = tf.convert_to_tensor(X, dtype=self.dtype)
        X2 = tf.convert_to_tensor(X2, dtype=self.dtype)

        # 1) Split out fidelity / dim for each set
        Xf, F1, D1 = self._split_fidelity_dim(X)
        X2f, F2, D2 = self._split_fidelity_dim(X2)

        # 2) We'll build a (N1, N2) covariance matrix K_full.
        N1 = tf.shape(Xf)[0]
        N2 = tf.shape(X2f)[0]
        K_full = tf.zeros((N1, N2), dtype=self.dtype)

        # 3) Identify the subsets of rows that share the same dimension
        #    If D1[i] != D2[j], we want that K[i,j] = 0 (no cross-cov between different output dims)
        #    Or we might want some correlation. For now, let's do block-diagonal structure (0 if dims differ).
        #
        #    If you'd rather have correlation across different dims, you can implement it
        #    by adding some B[dim1, dim2] factor. We'll keep it simple.
        for d in range(self.num_output_dims):
            # Indices where X dim_index = d
            mask_1 = tf.where(tf.equal(D1, d))[:, 0]
            # Indices where X2 dim_index = d
            mask_2 = tf.where(tf.equal(D2, d))[:, 0]

            # Gather those subsets
            Xf_d  = tf.gather(Xf,  mask_1)
            F1_d  = tf.gather(F1,  mask_1)
            X2f_d = tf.gather(X2f, mask_2)
            F2_d  = tf.gather(F2,  mask_2)

            # We can now form the block K for these subsets. 
            # Inside each dimension d, we do a multi-fidelity approach:
            #   - If F1_d[i]==0 and F2_d[j]==0 => both LF => use kernel_L
            #   - If one is HF => scale or add discrepancy kernel
            # Then multiply by rho[d].
            # We'll show a simplified approach:
            K_block = self._mf_block(Xf_d, F1_d, X2f_d, F2_d, d)

            # Then place K_block into the correct rows/cols of K_full
            # We use tf.tensor_scatter_nd_update to assign those sub-blocks.
            # Let's create the index grids:
            I1 = tf.reshape(mask_1, [-1, 1])  # shape (n1, 1)
            I2 = tf.reshape(mask_2, [-1, 1])  # shape (n2, 1)
            grid_1, grid_2 = tf.meshgrid(I1[:,0], I2[:,0], indexing="ij") 
            # We'll stack them as (n1*n2, 2)
            idx_pairs = tf.stack([tf.reshape(grid_1, [-1]), tf.reshape(grid_2, [-1])], axis=-1)

            # Flatten K_block for scatter
            K_block_flat = tf.reshape(K_block, [-1])
            K_full = tf.tensor_scatter_nd_update(K_full, idx_pairs, K_block_flat)

        return K_full

    def _mf_block(self, Xf1, F1, Xf2, F2, d):
        """
        Build the sub-covariance for dimension d. 
        We interpret F1=0 => LF, F1=1 => HF, similarly for F2.
        Then apply your multi-fidelity formula with self.kernel_L, self.kernel_delta,
        plus the scaling factor rho[d].
        """
        rho_d = self.rho[d, 0]  # scalar

        # Let's gather the subsets:
        #   For row-block: LF vs HF
        mask_lf_1 = tf.where(tf.equal(F1, 0))[:, 0]
        mask_hf_1 = tf.where(tf.equal(F1, 1))[:, 0]
        Xf1_lf = tf.gather(Xf1, mask_lf_1)
        Xf1_hf = tf.gather(Xf1, mask_hf_1)

        #   For col-block: LF vs HF
        mask_lf_2 = tf.where(tf.equal(F2, 0))[:, 0]
        mask_hf_2 = tf.where(tf.equal(F2, 1))[:, 0]
        Xf2_lf = tf.gather(Xf2, mask_lf_2)
        Xf2_hf = tf.gather(Xf2, mask_hf_2)

        # Cov sub-blocks
        K_LL = self.kernel_L.K(Xf1_lf, Xf2_lf)
        K_LH = self.kernel_L.K(Xf1_lf, Xf2_hf) * rho_d
        K_HL = self.kernel_L.K(Xf1_hf, Xf2_lf) * rho_d
        K_HH = self.kernel_L.K(Xf1_hf, Xf2_hf) * (rho_d**2) + self.kernel_delta.K(Xf1_hf, Xf2_hf)

        # We'll now assemble them into one block shape (N1, N2).
        N1 = tf.shape(Xf1)[0]
        N2 = tf.shape(Xf2)[0]
        block = tf.zeros((N1, N2), dtype=self.dtype)

        # Indices:
        row_LL = tf.stack(tf.meshgrid(mask_lf_1, mask_lf_2, indexing='ij'), axis=-1)
        row_LH = tf.stack(tf.meshgrid(mask_lf_1, mask_hf_2, indexing='ij'), axis=-1)
        row_HL = tf.stack(tf.meshgrid(mask_hf_1, mask_lf_2, indexing='ij'), axis=-1)
        row_HH = tf.stack(tf.meshgrid(mask_hf_1, mask_hf_2, indexing='ij'), axis=-1)

        block = tf.tensor_scatter_nd_update(block, 
                tf.reshape(row_LL, [-1, 2]), 
                tf.reshape(K_LL, [-1]))
        block = tf.tensor_scatter_nd_update(block, 
                tf.reshape(row_LH, [-1, 2]), 
                tf.reshape(K_LH, [-1]))
        block = tf.tensor_scatter_nd_update(block, 
                tf.reshape(row_HL, [-1, 2]), 
                tf.reshape(K_HL, [-1]))
        block = tf.tensor_scatter_nd_update(block, 
                tf.reshape(row_HH, [-1, 2]), 
                tf.reshape(K_HH, [-1]))

        return block

    def K_diag(self, X):
        """
        Diagonal elements. 
        We do a simpler version: if fidelity=0 => kernel_L diag,
        if fidelity=1 => (rho^2 kernel_L diag + kernel_delta diag).
        But each row's dimension => picks which rho[d].
        """
        Xf, F, D = self._split_fidelity_dim(X)
        Kd = tf.zeros((tf.shape(X)[0],), dtype=self.dtype)

        for d in range(self.num_output_dims):
            mask_d = tf.where(tf.equal(D, d))[:, 0]
            Xf_d = tf.gather(Xf, mask_d)
            F_d  = tf.gather(F,  mask_d)
            rho_d = self.rho[d, 0]

            # Sub-block:
            mask_lf = tf.where(tf.equal(F_d, 0))[:, 0]
            mask_hf = tf.where(tf.equal(F_d, 1))[:, 0]

            Xf_d_lf = tf.gather(Xf_d, mask_lf)
            Xf_d_hf = tf.gather(Xf_d, mask_hf)

            K_diag_lf = self.kernel_L.K_diag(Xf_d_lf)
            K_diag_hf = self.kernel_L.K_diag(Xf_d_hf) * (rho_d**2) + self.kernel_delta.K_diag(Xf_d_hf)

            # Place them:
            idx_lf = tf.gather(mask_d, mask_lf)
            idx_hf = tf.gather(mask_d, mask_hf)
            Kd = tf.tensor_scatter_nd_update(Kd, tf.reshape(idx_lf, [-1, 1]), tf.reshape(K_diag_lf, [-1]))
            Kd = tf.tensor_scatter_nd_update(Kd, tf.reshape(idx_hf, [-1, 1]), tf.reshape(K_diag_hf, [-1]))

        return Kd
