import torch

def get_device():
  print("--- GPU EVALUATION ---")

  if torch.cuda.is_available():
    print("Using GPU.\n")
    return torch.device("cuda")
  else:
    print("Using CPU.\n")
    return torch.device("cpu")