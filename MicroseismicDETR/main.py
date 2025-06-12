# ---------- Microseismic Event Detection and Location with DETR ---------- #
# ----------                Author: Yuanyuan Yang                ---------- #

# ---------------------------------------------------------------------------
# ------ Import Libraries ------
import time
import torch
import argparse

from utils import *
from dataset import *
from model import *
from engine import *
from matcher import *


# ---------------------------------------------------------------------------
# device = avail_device()
# print(f'Device: {device} \n')

parser = argparse.ArgumentParser('Set arguments.', add_help=False)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
device = torch.device(args.device)


# ---------------------------------------------------------------------------
# ------ Define Data-related Parameters ------
n_train = 6500   # number of training samples
# n_valid = 1375   # number of validation samples

nx = 141         # number of space samples for each data
nt = 101         # number of time samples for each data
n1_label = 5     # length of class label for each data
n2_label = 10    # length of location label for each data


# ---------------------------------------------------------------------------
# ------ Define Training Hyperparameters ------
bs = 8                         # batch size
lr = 0.16                      # (initial) learning rate
n_epoch = 1600                 # number of training epochs

criterion_cls = nn.BCELoss()   # the criterion for classification task
criterion_loc = nn.MSELoss()   # the criterion for location task

weight_cls = 0.2               # the weight for classification loss term in the loss function
weight_cost_class = 1          # relative weight of the classification loss in the matching cost
weight_cost_location = 4       # relative weight of the location loss in the matching cost

nsr = 0.1/3.                   # noise to signal ratio (random Gaussian noise)


# ---------------------------------------------------------------------------
# ------ Load Data ------
# - Training Set -
train_data = read_data("../data/training_data.bin", n_train, nx, nt)
train_label_cls, train_label_loc = read_label("../data/training_label.bin", n_train, n1_label, n2_label)
train_label_loc = norm_label_loc(train_label_loc)

# # - Validation Set -
# valid_data = read_data("../data/validation_data.bin", n_valid, nx, nt)
# valid_label_cls, valid_label_loc = read_label("../data/validation_label.bin", n_valid, n1_label, n2_label)
# valid_label_loc = norm_label_loc(valid_label_loc)

# print(f'Number of Training   Samples: {train_data.numpy().shape[0]}')
# print(f'Number of Validation Samples: {valid_data.numpy().shape[0]}')


# ---------------------------------------------------------------------------
# ------ Characterize Data ------
# To check the data distribution and also for data normalization.
train_mean = train_data.mean()
# train_std  = train_data.std()
train_std  = ((train_data.std())**2 + nsr**2)**0.5

# valid_mean = valid_data.mean()
# # valid_std  = valid_data.std()
# valid_std  = ((valid_data.std())**2 + nsr**2)**0.5


# ---------------------------------------------------------------------------
# ------ Build Network ------
set_seed(10)
network = DETRnet(num_classes=1, hidden_dim=64, nheads=8, num_encoder_layers=4, num_decoder_layers=4)
# network.load_state_dict(torch.load('../network/Initial_Network.pt'))


# ---------------------------------------------------------------------------
# ------ Build Matcher ------
matcher = build_matcher(weight_cost_class=weight_cost_class, weight_cost_location=weight_cost_location)


# ---------------------------------------------------------------------------
# ------ Train Network ------
# - Information Sent to GPU -
network = network.to(device)
train_mean = train_mean.to(device)
train_std  = train_std.to(device)
# valid_mean = valid_mean.to(device)
# valid_std  = valid_std.to(device)

print('Start training...')
tic = time.time()

# Define the optimizer and its scheduler.
optimizer = torch.optim.SGD(network.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.25, last_epoch=-1)

# - Network Training -
for epoch in range(1, n_epoch+1):
	# Train one epoch over the entire training set.
	train_loss_cls, train_loss_loc, train_loss = train(network,
													   criterion_cls, criterion_loc,
													   matcher, weight_cls,
													   optimizer,
													   train_data, train_label_cls, train_label_loc,
													   bs, nsr, train_mean, train_std,
													   device)

	# Write down the loss.
	if (epoch == 1) or (not epoch%5):
	#if epoch >= 1:
		print(epoch, optimizer.state_dict()['param_groups'][0]['lr'], train_loss_cls, file=open("../scripts/Training_Loss_cls.txt", "a"))
		print(epoch, optimizer.state_dict()['param_groups'][0]['lr'], train_loss_loc, file=open("../scripts/Training_Loss_loc.txt", "a"))
		print(epoch, optimizer.state_dict()['param_groups'][0]['lr'], train_loss, file=open("../scripts/Training_Loss.txt", "a"))

		# valid_loss_cls, valid_loss_loc, valid_loss = evaluate(network,
		# 													  criterion_cls, criterion_loc,
		# 													  matcher, weight_cls,
		# 													  valid_data, valid_label_cls, valid_label_loc,
		# 													  nsr, valid_mean, valid_std,
		# 													  device)
		# print(epoch, valid_loss_cls, file=open("../scripts/Validation_Loss_cls.txt", "a"))
		# print(epoch, valid_loss_loc, file=open("../scripts/Validation_Loss_loc.txt", "a"))
		# print(epoch, valid_loss, file=open("../scripts/Validation_Loss.txt", "a"))

	# Update the learning rate of optimizer.
	scheduler.step()

	# Save the network every 50 epochs.
	if not epoch%50:
		torch.save(network.state_dict(), f'../network/Trained_Network_ep{epoch}.pt')

# - Network Saving -
torch.save(network.state_dict(), '../network/Trained_Network.pt')


# ---------------------------------------------------------------------------
# ------ WELL DONE! ------
toc = time.time()
print(f'Training Time (hours): {((toc - tic) / 3600.):.1f}')
