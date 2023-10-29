import torch
import torch.nn as nn
import torchvision
from tqdm import trange
from Modules.DANN.DANN import DANN, DA_parameter

lab_loss = nn.CrossEntropyLoss(reduction = "sum")
dom_loss = nn.BCELoss(reduction = "sum")

def collate_fn(batch):
  return torch.stack([x[0].expand(3, -1, -1) for x in batch]), torch.hstack([torch.tensor([x[1]]) for x in batch])

def DANN_train(train_source, train_target, deploy_target, validation_set, hp, args):
  """
  Assumes the source dataset is larger than the target dataset
  Inputs
    train_source: source image training data
    train_target: target image training data
    test_target: current episode target image data
    validation_set: validation set against which to measure validation loss in training
    hp: hyperparameters (epoch, lr, batch_size, num_workers, weight on domain loss)
    args: learning device (CPU, GPU), tensorboard SummaryWriter
  Output
    model: trained model augmented with current episodes data
    train_target: augmented target training data
  """
  epochs, lr, batch_size, num_workers, domain_loss_weight = hp
  device, writer = args

  train_target = torch.utils.data.ConcatDataset((train_target, deploy_target))

  train_source_dataloader = torch.utils.data.DataLoader(train_source, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle = True)
  train_target_dataloader = torch.utils.data.DataLoader(train_target, batch_size = batch_size, num_workers = num_workers, pin_memory = True, sampler = torch.utils.data.RandomSampler(train_target, replacement = True, num_samples = len(train_source)), collate_fn = collate_fn)
  validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size, num_workers = num_workers)

  # Overwrite model for training
  model = DANN()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

  for epoch in trange(epochs):
    train_loss = 0. # Train the model, evaluate training loss
    
    train_source_domain_loss = 0. # component losses
    train_source_label_loss = 0.
    train_target_domain_loss = 0.

    train_source_iterator = iter(train_source_dataloader)
    train_target_iterator = iter(train_target_dataloader)
    validation_iterator = iter(validation_dataloader)

    for i, ((s_ip, s_op), (t_ip, t_op)) in enumerate(zip(train_source_iterator, train_target_iterator)):
      s_ip, s_op = s_ip.to(device), s_op.to(device) # transfer inputs, outputs to GPU
      t_ip, t_op = t_ip.to(device), t_op.to(device)
      lambd = DA_parameter(epoch, i, epochs, len(train_source_iterator)).to(device)

      optimizer.zero_grad()
      source_label, source_domain = model.forward(s_ip, lambd)
      target_label, target_domain = model.forward(t_ip, lambd)

      source_label_loss = lab_loss(source_label, s_op)
      source_domain_loss = dom_loss(source_domain, torch.ones(source_domain.shape).to(device))
      target_domain_loss = dom_loss(target_domain, torch.zeros(target_domain.shape).to(device))

      loss = source_label_loss + domain_loss_weight*(source_domain_loss + target_domain_loss) 
      train_source_label_loss += source_label_loss.item()
      train_source_domain_loss += domain_loss_weight*source_domain_loss.item()
      train_target_domain_loss += domain_loss_weight*target_domain_loss.item()

      mean_loss = loss / s_ip.shape[0] # normalize loss for gradient calculation
      mean_loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 0.001)
      optimizer.step()
      train_loss += loss.item() # track total loss
      writer.add_scalar("batch_mean_train_loss", mean_loss.item(), i + epoch*len(train_source_iterator)) # loss on current batch, source and target weighted equally, normalized by batch_size 
    writer.add_scalar("epoch_mean_train_loss", train_loss / len(train_source), epoch) # total loss on current epoch, comparable between different epochs as total data counts the same (after target weighting) 
    writer.add_scalar("epoch_mean_source_label_loss", train_source_label_loss / len(train_source), epoch) 
    writer.add_scalar("epoch_mean_source_domain_loss", train_source_domain_loss / len(train_source), epoch) 
    writer.add_scalar("epoch_mean_target_domain_loss", train_target_domain_loss / len(train_source), epoch)  

    val_lab_loss = 0. # evaluate validation loss
    val_dom_loss = 0.
    for t_ip, t_op in validation_iterator:
      t_ip, t_op = t_ip.to(device), t_op.to(device)

      target_label, target_domain = model.forward(t_ip)
      label_loss = lab_loss(target_label, t_op)
      domain_loss = dom_loss(target_domain, torch.zeros(target_domain.shape).to(device))
      
      val_lab_loss += label_loss.item()
      val_dom_loss += domain_loss.item()
    writer.add_scalar("epoch_mean_val_label_loss", val_lab_loss / len(validation_set), epoch) # total label loss on current epoch
    writer.add_scalar("epoch_mean_val_domain_loss", val_dom_loss / len(validation_set), epoch) # domain loss on current epoch
    writer.flush()
    scheduler.step()

  return model, train_target