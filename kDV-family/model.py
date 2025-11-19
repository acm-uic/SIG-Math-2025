import torch
import torch.nn as nn

"""
    Standard feedforward NN
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
    kdV Equation PINN
"""
class KdVPINN(nn.Module):
    def __init__(self, hidden_layers=[40,40,40,40], activation="tanh"):
        super().__init__()

        self.model = NN(hidden_layers=hidden_layers, activation=activation)
    
    def forward(self, x, t):
        return self.model(x,t)


    def predict(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x,t)
    
    def kdv_residual(self, x, t):
        """
        kdv redisual: u_t - 6u u_x + u_xxx = 0 
        """
        # Create tensors that require gradients
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        # Forward pass
        u = self.forward(x, t)

        # First derivatives
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xxx = torch.autograd.grad(
            u_xx, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        return u_t - 6*u*u_x + u_xxx
    