import numpy as np
import pandas as pd
from pysr import PySRRegressor
import joblib  # For saving and loading models

class MFGPDistiller:
    def __init__(self, niterations=100, complexity_penalty=0.01, population_size=100, alpha=1.0, gamma=0.01):
        """
        Initialize the distiller with PySR settings and custom loss weights.
        
        Parameters:
        - niterations: Number of iterations for PySR.
        - complexity_penalty: Regularization for equation complexity.
        - population_size: Number of symbolic equations in each generation.
        - alpha: Weight for MSE loss.
        - gamma: Weight for complexity penalty.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.trained = False
        
        self.model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=population_size,
            progress=True,
            model_selection="best",  # Trade-off between accuracy & complexity
            parsimony=self.gamma,  # Directly penalizes complex equations
            elementwise_loss=self._get_custom_loss(),  # MSE weighted by variance
            complexity_of_operators={
                "sin": 2, "cos": 2, "exp": 2, "log": 2,
                "+": 1, "-": 1, "*": 1, "/": 1, "^": 1},
            should_simplify=True,
        )
    
    def _get_custom_loss(self):
        """
        Returns the custom loss function as a Julia string for PySR.
        
        Uses MSE weighted by variance and includes complexity penalty:
        
        \[
        \mathcal{L} = \alpha \frac{(y_{\text{pred}} - y_{\text{HF}})^2}{\sigma_{\text{HF}}^2 + 1e-6} + \gamma C
        \]
        
        Where \( C \) is the complexity of the equation.
        """
        return (
            f"""
            (prediction, target, sigma_gp) -> (
                let sigma_gp_safe = max(sigma_gp, 1e-6)
                    {self.alpha} * ((prediction - target)^2 / sigma_gp_safe)
                end
            )
            """
        )
    
    def generate_training_data(self, mu_HF, sigma_HF, mu_LF, x, sample_size=300):
        """
        Generate training data from MF-GP predictions.
        
        Parameters:
        - mu_HF: High-fidelity predictions (numpy array).
        - sigma_HF: High-fidelity prediction uncertainties (numpy array).
        - mu_LF: Low-fidelity predictions (numpy array).
        - x: Additional input features (numpy array).
        - sample_size: Number of points to sample (prioritizing low uncertainty).
        
        Returns:
        - train_data: Pandas DataFrame with sampled (mu_LF, x, mu_HF, sigma_HF).
        """
        weights = 1 / (sigma_HF + 1e-6)
        weights /= weights.sum()
        # if multiple outputs, use the mean weight
        if len(sigma_HF.shape) > 1:
            weights = weights.mean(axis=1)
            weights /= weights.sum()
        selected_indices = np.random.choice(np.arange(len(mu_LF)), size=sample_size, p=weights)
        
        # multi-dimensional input
        if len(x.shape) > 1:
            data_dict = {
                'mu_LF': mu_LF[selected_indices], 
                'mu_HF': mu_HF[selected_indices],
                'sigma_HF': sigma_HF[selected_indices]
            }
            # adding xi features
            for i in range(x.shape[1]):
                data_dict[f'x{i}'] = x[selected_indices, i]
            train_data = pd.DataFrame(data_dict)
        else:
            train_data = pd.DataFrame({
                'mu_LF': mu_LF[selected_indices], 
                'x': x[selected_indices],
                'mu_HF': mu_HF[selected_indices],
                'sigma_HF': sigma_HF[selected_indices]
            })
        return train_data
    
    def train(self, train_data):
        """
        Train the symbolic regression model on MF-GP outputs using custom loss function.
        
        Parameters:
        - train_data: Pandas DataFrame with columns (mu_LF, x, mu_HF, sigma_HF).
        """
        # check if the data has multiple input features
        if 'x' in train_data.columns:
            self.model.fit(
                train_data[['mu_LF', 'x']],
                train_data['mu_HF'],
                weights=train_data['sigma_HF']
            )
        else:
            query = [f'x{i}' for i in range(train_data.shape[1] - 3)]
            self.model.fit(
                train_data[['mu_LF'] + query],
                train_data['mu_HF'],
                weights=train_data['sigma_HF']
            )

        self.trained = True
    
    def predict(self, mu_LF, x):
        """
        Use the trained model to make predictions.
        
        Parameters:
        - mu_LF: Low-fidelity predictions (numpy array).
        - x: Additional input features (numpy array).
        
        Returns:
        - mu_HF_pred: Predicted high-fidelity output.
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions.")
        
        if len(x.shape) > 1:
            data = pd.DataFrame({'mu_LF': mu_LF})
            for i in range(x.shape[1]):
                data[f'x{i}'] = x[:, i]
        else:
            data = pd.DataFrame({'mu_LF': mu_LF, 'x': x})
        return self.model.predict(data)
    
    def get_equation(self):
        """
        Retrieve the best symbolic equation.
        """
        if not self.trained:
            raise ValueError("Model must be trained before retrieving equation.")
        return self.model.get_best()
    
    def save(self, filename="mfgp_distiller.pkl"):
        """
        Save the trained symbolic regression model.
        """
        joblib.dump(self.model, filename)
    
    def load(self, filename="mfgp_distiller.pkl"):
        """
        Load a pre-trained symbolic regression model.
        """
        self.model = joblib.load(filename)
        self.trained = True