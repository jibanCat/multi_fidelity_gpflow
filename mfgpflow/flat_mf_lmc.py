import gpflow
import tensorflow as tf
import numpy as np

class FlattenedLinearCoreg(gpflow.kernels.Kernel):
    """
    A flattened Linear Coregionalization kernel that combines:
      - multiple "multi-fidelity" kernels (one per latent),
      - a learnable mixing matrix W of shape (num_outputs, num_latents),
      - the last column of X as the task index p in [0..num_outputs-1].
    
    The data X is shaped (N, D+2):
      - The last column => integer "task index" p
      - The second-last column => fidelity (ignored by this class)
      - The first D columns => continuous features, if any.
    
    Summation logic:
      K((x_i,p_i),(x_j,p_j)) = sum_{ℓ=1..L} W[p_i,ℓ]* W[p_j,ℓ]* base_kernels[ℓ].K( x_i[..], x_j[..] )
    """
    def __init__(self, base_kernels, num_outputs, W_init=None):
        """
        base_kernels: list of length L (e.g., your multi-fidelity kernels).
        num_outputs: number of tasks/outputs (P).
        W_init: optional initial shape (P, L). If None, random small init.
        """
        super().__init__()
        self.base_kernels = base_kernels
        self.num_latents = len(base_kernels)
        self.num_outputs = num_outputs

        if W_init is None:
            # small random init
            W_init = 0.1 * np.random.randn(num_outputs, self.num_latents)

        self.W = gpflow.Parameter(W_init)  # shape (P, L)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        X2 = tf.convert_to_tensor(X2, dtype=tf.float64)

        # The last column is the "task index"
        p1 = tf.cast(X[:, -1], tf.int32)   # shape (N1,)
        p2 = tf.cast(X2[:, -1], tf.int32)  # shape (N2,)

        # We'll pass only up to the second-last column to each base kernel
        # so it sees (features + fidelity), ignoring the dimension index
        Xfeat = X[:, :-1]   # shape (N1, D+1)
        X2feat = X2[:, :-1] # shape (N2, D+1)

        N1 = tf.shape(X)[0]
        N2 = tf.shape(X2)[0]
        K_full = tf.zeros((N1, N2), dtype=tf.float64)

        # sum_{ℓ} W[p1,ℓ] * W[p2,ℓ] * base_kernels[ℓ].K(Xfeat, X2feat)
        for ell, bk in enumerate(self.base_kernels):
            K_ell = bk.K(Xfeat, X2feat)  # shape (N1, N2)
            w1 = tf.gather(self.W[:, ell], p1)  # shape (N1,)
            w2 = tf.gather(self.W[:, ell], p2)  # shape (N2,)

            # Outer product => shape (N1, N2)
            W_outer = tf.matmul(w1[:, None], w2[None, :])
            K_full += K_ell * W_outer
        return K_full

    def K_diag(self, X):
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        p = tf.cast(X[:, -1], tf.int32)
        Xfeat = X[:, :-1]

        N = tf.shape(X)[0]
        Kd = tf.zeros((N,), dtype=tf.float64)

        for ell, bk in enumerate(self.base_kernels):
            kd_ell = bk.K_diag(Xfeat)
            w_ell = tf.gather(self.W[:, ell], p)  # shape (N,)
            Kd += kd_ell * tf.square(w_ell)

        return Kd