import gpflow
import tensorflow as tf
import numpy as np
from gpflow.utilities import positive, set_trainable, add_likelihood_noise_cov, assert_params_false
import tensorflow_probability as tfp
from gpflow.utilities import positive, set_trainable
from gpflow.models import GPR
from gpflow.likelihoods import Gaussian
from gpflow.base import InputData, MeanAndVariance
from gpflow.kernels import Kernel

class GraphMultiFidelityKernel(gpflow.kernels.Kernel):
    """
    Multi-output Linear Multi-Fidelity Kernel:
    f_H(x) = sum_i rho_i f_Li(x) + delta(x)
    where each f_Li has its own kernel, and delta(x) is an additional GP.
    """
    def __init__(self, kernel_L_list, kernel_delta, num_output_dims):
        super().__init__()
        self.kernel_L_list = kernel_L_list  # List of LF kernels
        self.kernel_delta = kernel_delta
        self.num_LF = len(kernel_L_list)
        self.rho = gpflow.Parameter(
            np.ones((self.num_LF, num_output_dims)), transform=positive()
        )

    def K(self, X, X2=None, ith_output_dim=0):
        if X2 is None:
            X2 = X

        X = tf.convert_to_tensor(X, dtype=tf.float64)
        X2 = tf.convert_to_tensor(X2, dtype=tf.float64)

        mask_L = [tf.where(X[:, -1] == i)[:, 0] for i in range(self.num_LF)]
        mask2_L = [tf.where(X2[:, -1] == i)[:, 0] for i in range(self.num_LF)]
        mask_H = tf.where(X[:, -1] == self.num_LF)[:, 0]
        mask2_H = tf.where(X2[:, -1] == self.num_LF)[:, 0]

        K_full = tf.zeros((X.shape[0], X2.shape[0]), dtype=tf.float64)
        rho_i = self.rho[:, ith_output_dim]  # shape (num_LF,)

        for i in range(self.num_LF):
            for j in range(self.num_LF):
                Xi = tf.gather(X[:, :-1], mask_L[i])
                X2j = tf.gather(X2[:, :-1], mask2_L[j])
                Kij = tf.zeros((tf.shape(Xi)[0], tf.shape(X2j)[0]), dtype=tf.float64)
                if i == j:
                    Kij = self.kernel_L_list[i].K(Xi, X2j)
                indices = tf.stack(tf.meshgrid(mask_L[i], mask2_L[j], indexing="ij"), axis=-1)
                K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices, [-1, 2]), tf.reshape(Kij, [-1]))

        for i in range(self.num_LF):
            X_L = tf.gather(X[:, :-1], mask_L[i])
            X_H2 = tf.gather(X2[:, :-1], mask2_H)
            K_LH = self.kernel_L_list[i].K(X_L, X_H2) * rho_i[i]
            indices = tf.stack(tf.meshgrid(mask_L[i], mask2_H, indexing="ij"), axis=-1)
            K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices, [-1, 2]), tf.reshape(K_LH, [-1]))

            X_H1 = tf.gather(X[:, :-1], mask_H)
            X_L2 = tf.gather(X2[:, :-1], mask2_L[i])
            K_HL = self.kernel_L_list[i].K(X_H1, X_L2) * rho_i[i]
            indices = tf.stack(tf.meshgrid(mask_H, mask2_L[i], indexing="ij"), axis=-1)
            K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices, [-1, 2]), tf.reshape(K_HL, [-1]))

        X_H = tf.gather(X[:, :-1], mask_H)
        X2_H = tf.gather(X2[:, :-1], mask2_H)
        K_HH = sum([self.kernel_L_list[i].K(X_H, X2_H) * (rho_i[i] ** 2) for i in range(self.num_LF)])
        K_HH += self.kernel_delta.K(X_H, X2_H)
        indices = tf.stack(tf.meshgrid(mask_H, mask2_H, indexing="ij"), axis=-1)
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices, [-1, 2]), tf.reshape(K_HH, [-1]))

        return K_full

    def K_diag(self, X, ith_output_dim=0):
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        mask_L = [tf.where(X[:, -1] == i)[:, 0] for i in range(self.num_LF)]
        mask_H = tf.where(X[:, -1] == self.num_LF)[:, 0]

        K_diag_full = tf.zeros((X.shape[0],), dtype=tf.float64)
        for i in range(self.num_LF):
            X_L = tf.gather(X[:, :-1], mask_L[i])
            K_diag_L = self.kernel_L_list[i].K_diag(X_L)
            K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(mask_L[i], [-1, 1]), tf.reshape(K_diag_L, [-1]))

        X_H = tf.gather(X[:, :-1], mask_H)
        rho_i = self.rho[:, ith_output_dim]
        K_diag_H = sum(self.kernel_L_list[i].K_diag(X_H) * (rho_i[i] ** 2) for i in range(self.num_LF))
        K_diag_H += self.kernel_delta.K_diag(X_H)
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(mask_H, [-1, 1]), tf.reshape(K_diag_H, [-1]))

        return K_diag_full


class MultiFidelityGPModel(GPR):
    """
    Gaussian Process model for multi-fidelity learning with multiple output dimensions using GraphMultiFidelityKernel.

    This model ensures:
    - Each output dimension has an independent `rho[i]` parameter.
    - Uses a custom kernel that models multiple LF sources and a residual HF GP.
    """

    def __init__(self, X, Y, kernel_L_list, kernel_delta):
        num_output_dims = Y.shape[1]  # Number of independent outputs
        self.kernel = GraphMultiFidelityKernel(kernel_L_list, kernel_delta, num_output_dims)
        likelihood = Gaussian(variance=1e-3)

        super().__init__((X, Y), kernel=self.kernel, likelihood=likelihood)
        set_trainable(self.likelihood.variance, False)

        self.num_output_dims = num_output_dims

    def optimize(self, max_iters=1000, learning_rate=0.01, use_adam=True, unfix_noise_after=500):
        """
        Optimizes the model while ensuring proper multi-output learning.

        - Handles per-output `rho[i]` updates separately.
        - Uses Adam or Scipy L-BFGS with noise-fixing for stability.
        """
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
