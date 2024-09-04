import os
import sys
import pdb
import tqdm
import torch
import wandb
import numpy as np
import torch.nn as nn
from copy import deepcopy
import pytorch_lightning as pl
import torch.nn.functional as F
from src.bert import BertModel, BertReward
from transformers.models.bert.configuration_bert import BertConfig

from src.utils import binary_accuracy

class local_trainer(pl.LightningModule):
	def __init__(self, train_loader, val_loader, test_dataset, args, eval_mode=False):
		super().__init__()

		self.config = BertConfig()

		self.config.vocab = 100
		self.config.num_labels = 1 # regression output
		self.config.problem_type = args.problem_type

		self.reward_model = BertReward(self.config)

		self.lr = args.lr
		self.weight_decay = args.weight_decay
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_dataset = test_dataset
		self.args = args
		self.eval_mode = eval_mode
		self.tr_acc, self.tr_loss = 0.0, 0.0
		self.val_acc, self.val_loss = 0.0, 0.0

		#self.automatic_optimization = False
	
		# def forward(self, pixel_values, pixel_mask):
		# 	outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
		# 	return outputs
	
	def common_step(self, batch, batch_idx):

		feat = batch['query_document_embedding'].to(self.device)
		avg_click = torch.clamp(batch['click'].sum(dim=1), max=1.0).to(self.device)
		pos_idx = batch['position']

		out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click)

		return out, avg_click
	
	def training_step(self, batch, batch_idx): # automatic training schedule
		
		out_dict, labels = self.common_step(batch, batch_idx)
		loss = out_dict['loss']
		logits = out_dict['logits']

		acc = binary_accuracy(logits.squeeze(), labels)

		wandb_out = {"tr_loss": loss, "tr_acc":acc}
		self.log("tr_loss", loss, prog_bar=True)
		self.log("tr_acc", acc, prog_bar=True)

		self.tr_acc += float(acc)
		self.tr_loss += float(loss)
		
		if self.args.use_wandb and self.trainer.global_rank==0:
			wandb.log(wandb_out)

		return loss

	# def on_after_backward(self, *args):
	# 	return

	def on_train_epoch_end(self):

		tr_acc_avg = self.tr_acc/self.trainer.num_training_batches
		tr_loss_avg = self.tr_loss/self.trainer.num_training_batches

		self.lr_scheduler.step()
		if self.current_epoch and self.current_epoch%self.args.save_epochs == 0:
			self.save(self.current_epoch)
			
		if self.trainer.global_rank==0:
			print('\n Train acc after ', self.current_epoch, ' epochs : ',\
							tr_acc_avg, '  loss : ',tr_loss_avg, file=self.args.log_file)
			

			wandb_out = {"tr_loss_global": tr_acc_avg, "tr_acc_global":tr_loss_avg}
			
			if self.args.use_wandb:
				wandb.log(wandb_out)
		
		self.tr_acc = 0.0
		self.tr_loss = 0.0

	def validation_step(self, batch, batch_idx):
		
		out_dict, labels = self.common_step(batch, batch_idx)
		loss = out_dict['loss']
		logits = out_dict['logits']

		acc = binary_accuracy(logits.squeeze(), labels)

		wandb_out = {"val_loss": loss, "val_acc":acc}
		self.log("val_loss", loss, prog_bar=True)
		self.log("val_acc", acc, prog_bar=True)

		self.val_acc += float(acc)
		self.val_loss += float(loss)
		
		if self.args.use_wandb and self.trainer.global_rank==0:
			wandb.log(wandb_out)

		return
	
	def on_validation_epoch_end(self):

		#pdb.set_trace()

		val_acc_avg = self.val_acc/self.trainer.num_val_batches[0]
		val_loss_avg = self.val_loss/self.trainer.num_val_batches[0]

		if self.trainer.global_rank==0:
			print('\n Val acc after ', self.current_epoch, ' epochs : ',\
							val_acc_avg, '  loss : ',val_loss_avg, file=self.args.log_file)
			
			wandb_out = {"val_loss_global": val_acc_avg, "val_acc_global":val_loss_avg}
			
			if self.args.use_wandb:
				wandb.log(wandb_out)
		
		self.val_acc = 0.0
		self.val_loss = 0.0
	
	def save(self, epoch):
		if self.trainer.global_rank==0:
			print('\n Saving at epoch ', epoch, file=self.args.log_file)

		torch.save({
					'model': self.reward_model.state_dict(),
					'optimizer': self.optimizer.state_dict(),
					'lr_scheduler': self.lr_scheduler.state_dict(),
					'epoch': epoch,
					#'args': self.args,
				}, os.path.join(self.args.output_dir, f'checkpoint{epoch:02}.pth'))
	
	def resume(self, load_path=''):
		
		#if self.trainer.global_rank==0:
		print('\n Resuming model from : ',load_path, file=self.args.log_file)
		
		if load_path:
			checkpoint = torch.load(load_path, map_location='cpu')
			missing_keys, unexpected_keys = self.reward_model.load_state_dict(checkpoint['model'], strict=False)
	
	
	def configure_optimizers(self):

		self.optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=self.lr,
								weight_decay=self.weight_decay)

		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_drop)

		return self.optimizer

	def train_dataloader(self):
		return self.train_dataloader

	def val_dataloader(self):
		return self.val_dataloader

class Evaluator():
	def __init__(self, processor, test_dataset, test_dataloader, 
			  args, local_trainer=None, local_eval=0):
		
		self.processor = processor
		self.local_trainer = local_trainer
		if local_trainer:
			self.model = local_trainer.model
		else:
			self.model = None

	def evaluate(self):
		args = self.args
		model = self.model