import torch
import torch.nn as nn


class DETRnet(nn.Module):
	"""
	DETR architecture-based network for Microseismic Event Detection and Location.
	"""

	def __init__(self, num_classes, hidden_dim=64, nheads=8, num_encoder_layers=4, num_decoder_layers=4):
		super().__init__()

		# CNN BACKBONE to extract a compact feature representation
		self.backbone_conv1 = nn.Conv2d(1,   32,  kernel_size=15, stride=1, padding=7)
		self.backbone_conv2 = nn.Conv2d(32,  128, kernel_size=13, stride=1, padding=6)
		self.backbone_conv3 = nn.Conv2d(128, 512, kernel_size=11, stride=1, padding=5)
		self.backbone_norm1 = nn.BatchNorm2d(32)
		self.backbone_norm2 = nn.BatchNorm2d(128)
		self.backbone_norm3 = nn.BatchNorm2d(512)
		self.backbone_activate = nn.ReLU()
		self.backbone_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

		# CONVERSION LAYER to reduce the channel dimension
		self.conversion = nn.Conv2d(512, hidden_dim, kernel_size=1)

		# TRANSFORMER
		self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, activation='gelu')

		# FFN to decode the N output embeddings from Transformer into class and location predictions
		self.linear_class = nn.Linear(hidden_dim, num_classes + 1)      # extra 1 for no_event class
		self.linear_location = nn.Linear(hidden_dim, 2)                 # X and Z source locations

		# Spatial Positional Encodings for feature maps
		self.row_embed = nn.Parameter(torch.rand(20, hidden_dim // 2))
		self.col_embed = nn.Parameter(torch.rand(20, hidden_dim // 2))

		# Object Queries as the input to Transformer decoder
		self.query_pos = nn.Parameter(torch.rand(5, hidden_dim))


	def forward(self, inputs):
		batch_size = inputs.shape[0]

		# Propagate inputs through the CNN BACKBONE.
		x = self.backbone_conv1(inputs)
		x = self.backbone_norm1(x)
		x = self.backbone_activate(x)
		x = self.backbone_pool(x)

		x = self.backbone_conv2(x)
		x = self.backbone_norm2(x)
		x = self.backbone_activate(x)
		x = self.backbone_pool(x)

		x = self.backbone_conv3(x)
		x = self.backbone_norm3(x)
		x = self.backbone_activate(x)
		x = self.backbone_pool(x)

		# Convert from 512 to hidden_dim feature planes.
		h = self.conversion(x)
		# h of size (batch_size, hidden_dim, H, W)

		# Construct positional encodings for feature planes.
		H, W = h.shape[-2:]
		pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
						 self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),], dim=-1)
		pos = pos.flatten(0, 1).unsqueeze(1).repeat(1, batch_size, 1)
		# pos of size (H * W, batch_size, hidden_dim)

		# Propagate through the TRANSFORMER.
		h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
							 self.query_pos.unsqueeze(1).repeat(1, batch_size, 1)).transpose(0, 1)
		# encoder input of size (H * W, batch_size, hidden_dim)
		# decoder input of size (num_queries, batch_size, hidden_dim)
		# transformer output of size: the same as decoder input size
		# h of size (batch_size, num_queries, hidden_dim)

		# Project TRANSFORMER outputs to class and location predictions through FFN.
		return {'pred_logits': self.linear_class(h), 'pred_locations': self.linear_location(h).sigmoid()}
		# outputs["pred_logits"] size: (batch_size, num_queries, num_classes + 1)
		# outputs["pred_locations"] size: (batch_size, num_queries, 2)