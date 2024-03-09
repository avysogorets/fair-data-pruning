import torch


class MLP(torch.nn.Module):
    def __init__(self,
                in_shape,
                hidden_size,
                depth,
                num_classes,
                dropout,
                device=torch.device('cpu')):
        super().__init__()
        self.input_dim = in_shape
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device
        modules = []
        curr_dim = in_shape
        for layer in range(depth):
            if layer>0:
                curr_dim = hidden_size
            modules += self.block(input_dim=curr_dim)
        modules += [torch.nn.Linear(curr_dim, num_classes, bias=True)]
        self.mlp = torch.nn.Sequential(*modules)
        self.to(self.device)

    def block(self, input_dim):
        block_modules = [
            torch.nn.Linear(input_dim, self.hidden_size, bias=False),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout)]
        return block_modules

    def forward(self, X):
        return self.mlp(X)


class MCDropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self._inference_mask = None
        self.enable_inference_dropout = False
    def resample_inference_mask(self):
        self._inference_mask = None
    def forward(self, x, **kwargs):
        if self.training:
            return x*torch.bernoulli(torch.full_like(x, 1-self.p))/(1-self.p)
        else:
            if self._inference_mask is None:
                self._inference_mask = torch.bernoulli(torch.full_like(x[0], 1-self.p))/(1-self.p)
            if self.enable_inference_dropout:
                return x*self._inference_mask
            else:
                return x


def configure_dropout_layers(model):
    dropout_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            mc_dropout = MCDropout(m.p)
            dropout_layers.append(mc_dropout)
            m.forward = mc_dropout.forward
    return dropout_layers