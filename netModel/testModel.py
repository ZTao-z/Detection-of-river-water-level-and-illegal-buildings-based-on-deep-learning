from resnet import resnet101
import torch

if __name__ == '__main__':
    model = resnet101()
    input = torch.rand(2,3,512,512)
    res = model(input)
    print(model)