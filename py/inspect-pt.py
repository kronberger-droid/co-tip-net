import torch

state_dict = torch.load("pretrained_weights/model.pt", weights_only=True)
for key, tensor in state_dict.items():
    print(f"{key}: {tensor.shape}")
