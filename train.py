import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F

def eval_net(net, loader, criterion):
	net.eval()
	n_val = len(loader)
	acc = 0
	running_loss = 0
	i = 0
	with tqdm(total=n_val, desc='Validation round', position=0, unit = 'batch', leave=True) as pbar:
		for batch in enumerate(loader):
			with torch.no_grad():
				pass
		pbar.update()
		return acc

def train(net, train_loader, val_loader,
		epochs, optimiser, batch_size, 
		scheduler= None, 
		teacher=None, ES=False):
	criterion_data = nn.BCEWithLogitsLoss(); #add knowledge distillation loss
	losses = []
	val =[]
	for epochs in range(epochs):
		acc, val_loss = eval_net(net, val_loader, criterion_data)
		val_losses.append(val_loss)
		accuracies.append(acc)
		print(f"Validation accuracy {acc}")

		net.train()

		running_loss = 0
		i = 0
		with tqdm(total=len(train_loader), desc = "Training epoch: {}".format(epoch),
				position=0, unit = 'batch', leave=True) as pbar:
			for batch in enumerate(train_loader):
				optimiser.zero_grad()
				# inputs = batch[1][0].to(device = device, dtype=torch.float32)
				# labels = batch[1][1]
				# out = net(inputs)
				# target = torch.zeros(len(labels), 10).scatter_(1, labels.unsqueeze(1), 1.).to(device = device, dtype=torch.float32)
				loss = criterion_data(out, target)
				running_loss += loss.item()
				i += 1
				loss.backward()
				optimiser.step()
				pbar.update()

		print(f"\n Epoch {epoch}: Loss {running_loss/i}")
		losses.append(running_loss/i)

