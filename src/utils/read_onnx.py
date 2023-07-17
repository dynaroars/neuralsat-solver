import onnxruntime as ort
import torch.nn as nn
import collections
import numpy as np
import torch
import math
import onnx

import onnx2pytorch
import onnx2pytorch_old


def get_activation_shape(name, result):
    def hook(model, input, output):
        result[name] = output.shape
    return hook


class PyTorchModelWrapper(nn.Module):

    def __init__(self, layers):
        super().__init__()

        if isinstance(layers, list):
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = layers

        self.layers_mapping = None
        self.input_shape = None

        self.n_input = None
        self.n_output = None
        
        
        self.activation = collections.OrderedDict()
        for name, layer in self.layers.named_modules():
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(get_activation_shape(name, self.activation))
        

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)


    def forward_grad(self, x):
        return self.layers(x)



    @torch.no_grad()
    def get_assignment(self, x):
        idx = 0
        implication = {}
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                s = torch.zeros_like(x, dtype=int) 
                s[x > 0] = 1
                implication.update(dict(zip(self.layers_mapping[idx], s.flatten().numpy().astype(dtype=bool))))
                idx += 1
        return implication

    @torch.no_grad()
    def get_concrete(self, x):
        x = x.view(self.input_shape)
        idx = 0
        implication = {}
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                implication.update(dict(zip(self.layers_mapping[idx], x.view(-1))))
                idx += 1
            x = layer(x)
        return implication

    @torch.no_grad()
    def forward_layer(self, x, lid):
        relu_idx = 0
        # print(lid)
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                relu_idx += 1
            if relu_idx <= lid:
                continue
            # print(layer)
            x = layer(x)
        return x



def inference_onnx(path: str, *inputs: np.ndarray) -> list[np.ndarray]:
    sess = ort.InferenceSession(onnx.load(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, inputs)))


def add_batch(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (1, shape[0])
    
    if shape[0] not in [-1, 1]:
        return (1, *shape)
    
    return shape

def _parse_onnx(path: str) -> tuple:
    # print('Loading ONNX with customized quirks:', custom_quirks)
    onnx_model = onnx.load(path)
    
    onnx_inputs = [node.name for node in onnx_model.graph.input]
    initializers = [node.name for node in onnx_model.graph.initializer]
    inputs = list(set(onnx_inputs) - set(initializers))
    inputs = [node for node in onnx_model.graph.input if node.name in inputs]
    
    onnx_input_dims = inputs[0].type.tensor_type.shape.dim
    onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
    
    orig_input_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_input_dims)
    orig_output_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_output_dims)
    
    batched_input_shape = add_batch(orig_input_shape)
    batched_output_shape = add_batch(orig_output_shape)

    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks={'Reshape': {'fix_batch_size': True}})
    pytorch_model.eval()

    pytorch_model.to(torch.get_default_dtype())
    
    is_nhwc = pytorch_model.is_nhwc
    
    # print(pytorch_model)
    # print(batched_input_shape, batched_output_shape)
    
    # check conversion
    correct_conversion = True
    try:
        batch = 2
        dummy = torch.randn(batch, *batched_input_shape[1:], dtype=torch.get_default_dtype())
        # print(dummy.shape)
        output_onnx = torch.cat([torch.from_numpy(inference_onnx(path, dummy[i].view(orig_input_shape).float().numpy())[0]).view(batched_output_shape) for i in range(batch)])
        # print('output_onnx:', output_onnx)
        output_pytorch = pytorch_model(dummy.permute(0, 3, 1, 2) if is_nhwc else dummy).detach().numpy()
        # print('output_pytorch:', output_pytorch)
        correct_conversion = np.allclose(output_pytorch, output_onnx, 1e-5, 1e-5)
    except:
        raise 

    if is_nhwc:
        assert len(batched_input_shape) == 4
        n_, h_, w_, c_ = batched_input_shape
        batched_input_shape = (n_, c_, h_, w_)
    
    return pytorch_model, batched_input_shape, batched_output_shape, is_nhwc



class ONNXParser:

    def __init__(self, filename, dataset):

        # force_convert = False
        # if dataset == 'mnist':
        #     input_shape = (1, 1, 28, 28)
        #     n_output = 10
        # elif dataset == 'cifar':
        #     input_shape = (1, 3, 32, 32)
        #     n_output = 10
        # elif dataset == 'acasxu':
        #     input_shape = (1, 5)
        #     n_output = 5
        #     force_convert = True
        # else:
        #     raise 

        # model, is_channel_last = load_model_onnx(filename, input_shape[1:], force_convert=force_convert)
        
        model, input_shape, output_shape, is_nhwc = _parse_onnx(filename)
        if 'convBigRELU__PGD' in filename or dataset == 'test':
            model = onnx2pytorch_old.ConvertModel(onnx.load(filename), experimental=True)
            model = nn.Sequential(*list(model.modules())[1:])

        model = model.eval()

        self.pytorch_model = PyTorchModelWrapper(model)
        self.pytorch_model.n_input = math.prod(input_shape)
        self.pytorch_model.n_output = math.prod(output_shape)
        self.pytorch_model.input_shape = input_shape
        self.pytorch_model.is_nhwc = is_nhwc


    