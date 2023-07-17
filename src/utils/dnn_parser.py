from pprint import pprint
import sortedcontainers
import torch.nn as nn
import onnx2pytorch
import numpy as np
import torch
import math

from utils.read_onnx import ONNXParser
from utils.read_nnet import NetworkNNET
import settings

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p is not None)


class DNNParser:

    def parse(filename, dataset, device=torch.device('cpu')):
        if filename.lower().endswith('.nnet'):
            # assert dataset in ['acasxu', 'test']
            net = DNNParser.parse_nnet(filename)
        elif filename.lower().endswith('.onnx'):
            net = DNNParser.parse_onnx(filename, dataset)
        else:
            print(f'Error extention: {filename}')
            raise NotImplementedError
        net.device = device
        net.dataset = dataset
        return net.to(device)

    def parse_nnet(filename):
        model = NetworkNNET(filename)

        layers_mapping = {}
        idx = 1
        lid = 0
        for layer in model.layers[1:]: # discard input layer
            if isinstance(layer, nn.Linear):
                layers_mapping[lid] = sortedcontainers.SortedList()
                for i in range(layer.weight.shape[1]): # #nodes in layer
                    layers_mapping[lid].add(idx)
                    idx += 1
                lid += 1

        model.layers_mapping = layers_mapping
        return model

    def parse_onnx(filename, dataset):

        model = ONNXParser(filename, dataset)
        pytorch_model = model.pytorch_model
        # x = torch.randn(pytorch_model.input_shape, dtype=settings.DTYPE)

        # count = 1
        # layers_mapping = {}
        # idx = 0
        
        # print([isinstance(_, nn.Module) for _ in pytorch_model.layers.modules()])
        # print([_ for _ in list(pytorch_model.layers.modules())[1:]])
        # exit()
        # for layer in pytorch_model.layers.modules():
        #     if isinstance(layer, nn.ReLU):
        #         layers_mapping[idx] = sortedcontainers.SortedList(range(count, count+x.numel()))
        #         idx += 1
        #         count += x.numel()
        #     x = layer(x)

        # pytorch_model.layers_mapping = layers_mapping
        
        return DNNParser.add_layer_mapping(pytorch_model)


    def add_layer_mapping(model):
        # forward to record relu shapes
        x = torch.randn(model.input_shape)
        assert x.shape[0] == 1
        output_pytorch = model(x)
        # extract boolean abstraction
        count = 1
        layers_mapping = {}
        idx = 0
        for k, v in model.activation.items():
            layers_mapping[idx] = sortedcontainers.SortedList(range(count, count+np.prod(v)))
            idx += 1
            count += np.prod(v)
        model.layers_mapping = layers_mapping
        assert count > 1, "Only supports ReLU activation"
        print("Number of SAT variables:", count - 1)
        print("Number of parameters:", get_num_params(model))
        # print(layers_mapping)
        # exit()
        
        return model