import torch
import torch.nn as nn

def DANN_eval(model, test_source, test_target, dl_params, device):
  """
  Inputs
    model: trained DANN model
    test_source: source test data
    test_target: target test data
    dl_params: dataloading parameters (batch_size, num_workers)
  Outputs
    source_acc: accuracy on source test
    target_acc: accuracy on target test
    model_acc: accuracy on source + target test
  """
  batch_size, num_workers = dl_params

  source_correct = 0
  source_test_iterator = iter(torch.utils.data.DataLoader(test_source, batch_size = batch_size, num_workers = num_workers))
  
  target_correct = 0
  target_test_iterator = iter(torch.utils.data.DataLoader(test_target, batch_size = batch_size, num_workers = num_workers))
  
  softmax = nn.Softmax(dim = -1)
  for s_ip, s_op in source_test_iterator:
    s_ip, s_op = s_ip.to(device), s_op.to(device)
    model_op, _ = model(s_ip, torch.tensor([1.]))
    model_op = torch.argmax(softmax(model_op), dim = -1)
    source_correct += len(torch.where(model_op == s_op)[0])
  source_acc = source_correct / len(test_source)

  for t_ip, t_op in target_test_iterator:
    t_ip, t_op = t_ip.to(device), t_op.to(device)
    model_op, _ = model(t_ip, torch.tensor([1.]))
    model_op = torch.argmax(softmax(model_op), dim = -1)
    target_correct += len(torch.where(model_op == t_op)[0])
  target_acc = target_correct / len(test_target)
  
  return source_acc, target_acc