import gpflow
import tensorflow as tf
import numpy as np
from gpflow.utilities import positive

class FlattenedLinearCoreg(gpflow.kernels.Kernel):
    """
    A flattened Linear Coregionalization kernel that combines:
      - multiple base "multi-fidelity" kernels (one per latent),
      - a learnable mixing matrix W of shape (num_outputs, num_latents),
      - a final "task index"/"output" column in X to pick the appropriate row(s) of W.
    
    The data X is shaped (N, D+2):
      - The last column: integer task index p in [0..num_outputs-1].
      - The second-last column: fidelity (0=LF,1=HF) or more advanced logic if your base kernel supports it.
      - The first D columns: continuous features, if any.
    
    Summation logic: The N1*N2 kernel matrix K(X1, X2) is weighted sum of base kernels in each latent:
      K((x_i,p_i), (x_j,p_j)) = sum_{ell=1..L} [W[p_i, ell] * W[p_j, ell] * base_kernels[ell].K(x_i, x_j)].
    """
    def __init__(self, base_kernels, num_outputs, W_init=None):
        """
        base_kernels: list of length L, each a LinearMultiFidelityKernel or similar
        num_outputs: P (number of tasks)
        W_init: optional initial (P, L). If None, random small init.
        """
        super().__init__()
        self.base_kernels = base_kernels
        self.num_latents = len(base_kernels)
        self.num_outputs = num_outputs

        if W_init is None:
            # small random init
            W_init = 0.1 * np.random.randn(num_outputs, self.num_latents)
        # You can wrap W in a positive transform if you only want nonnegative, 
        # or keep it unconstrained:
        self.W = gpflow.Parameter(W_init, transform=None)  

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        X2 = tf.convert_to_tensor(X2, dtype=tf.float64)

        # parse out the last column => task index
        p1 = tf.cast(X[:, -1], tf.int32)   # shape (N1,)
        p2 = tf.cast(X2[:, -1], tf.int32)  # shape (N2,)

        N1 = tf.shape(X)[0]
        N2 = tf.shape(X2)[0]
        K_full = tf.zeros((N1, N2), dtype=tf.float64)

        # sum_{ell} W[p1, ell] * W[p2, ell] * base_kernels[ell].K(x1, x2)
        for ell, bk in enumerate(self.base_kernels):
            K_ell = bk.K(X, X2) # shape (N1, N2)
            w1 = tf.gather(self.W[:, ell], p1)  # (N1,) => gather row p1[i], col=ell
            w2 = tf.gather(self.W[:, ell], p2)  # (N2,)
            # outer product
            W_outer = tf.matmul(w1[:, None], w2[None, :])  # shape (N1, N2)
            K_full += K_ell * W_outer

        return K_full

    def K_diag(self, X):
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        p = tf.cast(X[:, -1], tf.int32)
        Xfeat = X[:, :-1]
        N = tf.shape(X)[0]
        Kd = tf.zeros((N,), dtype=tf.float64)

        # sum_{ell} W[p,ell]^2 * base_kernels[ell].K_diag(x)
        for ell, bk in enumerate(self.base_kernels):
            kd_ell = bk.K_diag(Xfeat)
            w_ell = tf.gather(self.W[:, ell], p)  # shape (N,)
            Kd += kd_ell * tf.square(w_ell)

        return Kd
