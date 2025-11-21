import torch
import torch.nn as nn


"""
    Multiplicative Filter Network to deal with oscillations
"""
class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=6.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(torch.rand(out_features) * 2 - 1)
        self.gamma = nn.Parameter(torch.ones(out_features) * alpha)
        self.beta = beta

    
    def forward(self, x):
        D = self.linear(x)
        return torch.sin(self.gamma * (D + self.mu)) * torch.exp(-self.beta * D**2)


class MultiplicativeFilterNetwork(nn.Module):
    def __init__(self, hidden_layers=[128, 128, 128, 128], alpha=6.0, beta=0.1):
        super().__init__()
        
        # Input layer
        self.input_layer = GaborLayer(2, hidden_layers[0], alpha=alpha, beta=beta)
        
        # Hidden layers
        self.hidden = nn.ModuleList([GaborLayer(hidden_layers[i], hidden_layers[i+1], alpha=alpha, beta=beta) 
                                        for i in range(len(hidden_layers) - 1)])
        
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.apply(self._initialize_weights)
    
    @staticmethod 
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    
    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        out = self.input_layer(input)
        for layer in self.hidden:
            out = layer(out)
        return self.output(out)
    
"""
    Standard feedforward NN (bad performance)
"""
class NN(nn.Module):
    def __init__(self, hidden_layers=[20,20,20,20], activation="relu"):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_layers[0])
        self.activation = self._get_activation(activation)
        self.hidden = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)])
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.apply(self._initialize_weights)
    
    @staticmethod 
    def _get_activation(activation):
        if activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "relu":
            return nn.ReLU()
        else:
            return nn.ReLU()
        
    @staticmethod 
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        output = self.input_layer(input)
        for layer in self.hidden:
            output = self.activation(layer(output))
        output = self.output(output)
        return output








"""
    Helmholtz Equation PINN
"""
class HelmholtzPINN(nn.Module):
    def __init__(self, f, k=1, hidden_layers=[40,40,40,40], activation="tanh", alpha=6.0, beta=1.0):
        super().__init__()

        self.k = k
        self.f = f
        self.model = MultiplicativeFilterNetwork(hidden_layers=hidden_layers, alpha=alpha, beta=beta)
    
    def forward(self, x, y):
        return self.model(x,y)


    def predict(self, x, y):
        self.model.eval()
        with torch.no_grad():
            return self.model(x,y)
    
    def helmholtz_residual(self, x, y):
        """
        Helmholtz redisual 
        """
        # Create tensors that require gradients
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        # Forward pass
        u = self.forward(x, y)

        # First derivatives
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0]

        # Heat equation residual
        return u_xx + u_yy + self.k**2 * u - self.f(x,y)
    