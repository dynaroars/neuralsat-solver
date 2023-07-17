from torch.nn import functional as F
import torch.nn as nn
import torch


class ExampleNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        linear1 = nn.Linear(2, 2)
        linear1.weight.data = torch.Tensor([[-0.5, 0.5], [1, 1]]).float()
        linear1.bias.data = torch.Tensor([1, -1]).float()
        
        linear2 = nn.Linear(2, 1)
        linear2.weight.data = torch.Tensor([[-1, 1]]).float()
        linear2.bias.data = torch.Tensor([-1]).float()
        
        self.layers = nn.Sequential(
            linear1,
            nn.ReLU(),
            linear2
        )
        
    @torch.no_grad()
    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            print(idx, x)
        return x
    
    
if __name__ == '__main__':

    model = ExampleNet()
    # model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4), nn.Linear(4, 5))
    model.eval()
    print(model)
    
    x = torch.tensor([[-1.0, 1.0]])
    y = model(x)
    
    torch.onnx.export(
        model,
        x,
        "example/paper_example.onnx",
        opset_version=12,
        verbose=True
    )
    