import os
import sys
import pdb
import tqdm
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import pytorch_lightning as pl
import torch.nn.functional as F
from src.bert import BertModel, BertReward
from transformers.models.bert.configuration_bert import BertConfig

class local_trainer(pl.LightningModule):
	def __init__(self, train_loader, val_loader, test_dataset, args, eval_mode=False):
		super().__init__()

		self.config = BertConfig()

		self.config.vocab = 100
		self.config.num_labels = 1 # regression output

		self.reward_model = BertReward(self.config)

		self.lr = args.lr
		self.weight_decay = args.weight_decay
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_dataset = test_dataset
		self.args = args
		self.eval_mode = eval_mode

		#self.automatic_optimization = False
	
	def forward(self, pixel_values, pixel_mask):
		outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
		return outputs
	
	def common_step(self, batch, batch_idx, return_outputs=None):

		feat = batch['query_document_embedding'].to(self.device)
		avg_click = torch.clamp(batch['click'].sum(dim=1)/3.0, max=1.0).to(self.device)
		pos_idx = batch['position']

		out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click)

		pdb.set_trace()

		return loss, loss_dict
	
	def training_step(self, batch, batch_idx): # automatic training schedule
		
		loss, loss_dict = self.common_step(batch, batch_idx)
		# logs metrics for each training_step,
		# and the average across the epoch
		#values = {k:v for k,v in loss_dict.items()}
		short_map = {'loss_ce':'ce','loss_giou':'giou','cardinality_error':'car','training_loss':'tr','loss_bbox':'bbox', 'query_loss':'QL'}
		self.log("tr", loss, prog_bar=True)
		for k,v in loss_dict.items():
			#self.log("train_" + k, v.item(), prog_bar=True)
			self.log(short_map[k], v.item(), prog_bar=True)

		return loss

	def on_after_backward(self, *args):
		return

	def on_train_epoch_end(self):
		self.lr_scheduler.step()
		if self.current_epoch and self.current_epoch%self.args.save_epochs == 0:
			self.save(self.current_epoch)

	def validation_step(self, batch, batch_idx):
		loss = 0.0
		return loss
	
	def save(self, epoch):
		print('\n Saving at epoch ', epoch, file=self.args.log_file)
		torch.save({
					'model': self.model.state_dict(),
					'optimizer': self.optimizer.state_dict(),
					'lr_scheduler': self.lr_scheduler.state_dict(),
					'epoch': epoch,
					#'args': self.args,
				}, os.path.join(self.args.output_dir, f'checkpoint{epoch:02}.pth'))
	
	def resume(self, load_path=''):
		print('\n Resuming model for task ', self.task_id, ' from : ',load_path, file=self.args.log_file)
		if load_path:
			checkpoint = torch.load(load_path, map_location='cpu')
			missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model'], strict=False)
	
	
	def configure_optimizers(self):

		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
								weight_decay=self.weight_decay)

		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_drop)

		return self.optimizer

	def train_dataloader(self):
		return self.train_dataloader

	def val_dataloader(self):
		return self.val_dataloader

class Evaluator():
	def __init__(self, processor, test_dataset, test_dataloader, coco_evaluator, 
			  task_label2name, args, local_trainer=None, PREV_INTRODUCED_CLS=0, 
			  CUR_INTRODUCED_CLS=20, local_eval=0, task_id=0, task_name=None):
		
		self.processor = processor
		self.local_trainer = local_trainer
		if local_trainer:
			self.model = local_trainer.model
		else:
			self.model = None

	def evaluate(self):
		args = self.args
		model = self.model