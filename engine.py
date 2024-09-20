import os
import sys
import pdb
import tqdm
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn
from copy import deepcopy
import pytorch_lightning as pl
import torch.nn.functional as F
from src.bert import BertModel, BertReward
from transformers.models.bert.configuration_bert import BertConfig
#from bbm.src.model import CrossEncoder

from src.utils import binary_accuracy, sample_without_replacement_with_prob as sample_perturb, sample_swap

class local_trainer(pl.LightningModule):
	def __init__(self, train_loader, val_loader, test_dataset, args, eval_mode=False):
		super().__init__()

		self.config = BertConfig()

		self.config.vocab = 100
		self.config.num_labels = 1 # regression output
		self.config.problem_type = args.problem_type
		self.config.max_position_embeddings = args.max_positions_PE

		if args.use_doc_feat:
		# 	self.config.doc_feat_len=0
			self.config.hidden_size+=12

		self.reward_model = BertReward(self.config)
		# self.cross_enc = CrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_naive-pointwise")

		self.lr = args.lr
		self.weight_decay = args.weight_decay
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_dataset = test_dataset
		self.args = args
		self.eval_mode = eval_mode
		self.tr_acc, self.tr_loss = 0.0, 0.0
		self.val_acc, self.val_loss = 0.0, 0.0
		# self.cls_token_save = []
		# self.cls_label_save = []
		# self.cls_prob_save = []

		self.save_output = {'cls_token_save':[], 'cls_label_save':[], 'cls_prob_save':[]}

		self.evaluator = Evaluator(args=self.args)

		# self.automatic_optimization = False
	
		# def forward(self, pixel_values, pixel_mask):
		# 	outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
		# 	return outputs
	
	def common_step(self, batch, batch_idx):

		feat = torch.tensor(batch['query_document_embedding'], dtype=torch.float).to(self.device)
		click = torch.tensor(batch['click']).to(self.device)
		
		avg_click = torch.clamp(click.sum(dim=1), max=1.0).to(self.device) # hard labels

		if self.args.soft_labels: # soft labels
			gt_binary_labels = torch.clamp(click.sum(dim=1), max=1.0)
			avg_click = (self.args.soft_base + click.sum(dim=1) * self.args.soft_gain)*gt_binary_labels
			avg_click = torch.clamp(avg_click, max=1.0).to(self.device)

		pos_idx = torch.tensor(batch['position']).to(self.device)

		if self.args.eval or self.args.debug:
			for i,j in enumerate(pos_idx):
				n = batch['n'][i]
	
				if self.args.sampling_type == 'rand_perturb':
					pos_idx[i][0:n] = sample_perturb(delta=self.args.delta_retain, pos=pos_idx[i][0:n].float())
				else:
					pos_idx[i][0:n] = sample_swap(pos=pos_idx[i][0:n].float(), click=click, fn=self.args.sampling_type)

		#pdb.set_trace()

		doc_feats = None

		if self.args.use_doc_feat:
			doc_feats = torch.cat([
								torch.tensor(batch['bm25'], dtype=torch.float).to(self.device).unsqueeze(2), 
								torch.tensor(batch['tf'], dtype=torch.float).to(self.device).unsqueeze(2),
								torch.tensor(batch['idf'], dtype=torch.float).to(self.device).unsqueeze(2),
								torch.tensor(batch['tf_idf'], dtype=torch.float).to(self.device).unsqueeze(2),
						 		torch.tensor(batch['ql_jelinek_mercer_short'], dtype=torch.float).to(self.device).unsqueeze(2),
								torch.tensor(batch['ql_jelinek_mercer_long'], dtype=torch.float).to(self.device).unsqueeze(2), 
								torch.tensor(batch['ql_dirichlet'], dtype=torch.float).to(self.device).unsqueeze(2)],
							dim=2)
			
			# pdb.set_trace()
			# doc_feats = min_max_normalize(doc_feats)

			doc_feats = torch.cat([doc_feats, torch.zeros((batch['bm25'].shape[0],batch['bm25'].shape[1],5)).to(self.device)],dim=2)
			
			feat = torch.cat([feat,doc_feats], dim=2) #TODO: Pass doc_feats through MLP before concat

		# if self.args.use_model_preds:
		# 	# with torch.device('cpu'):
		# 	row,col = batch['tokens'].shape[0], batch['tokens'].shape[1] 
		# 	new_batch = {'tokens':batch['tokens'].reshape(row*col,128),
		# 				'attention_mask':batch['attention_mask'].reshape(row*col,128),
		# 				'token_types':batch['token_types'].reshape(row*col,128)}
		# 	pred = np.array(self.cross_enc(new_batch).click)
		# 	avg_click = torch.sigmoid(torch.tensor(pred).to(self.device).view(row,col)) >= 0.5
		# 	avg_click = torch.clamp(avg_click.int().sum(dim=1), max=1.0).to(self.device)
		
		# pdb.set_trace()

		out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click, doc_feats=doc_feats)
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
		cls_token = out_dict['cls_token']

		acc = binary_accuracy(logits.squeeze(), labels, soft=self.args.soft_labels)

		if self.args.debug:
			if batch_idx>10:
				return

		wandb_out = {"val_loss": loss, "val_acc":acc}
		self.log("val_loss", loss, batch_size=self.args.batch_size, prog_bar=True)
		self.log("val_acc", acc, batch_size=self.args.batch_size, prog_bar=True)

		self.val_acc += float(acc)
		self.val_loss += float(loss)
		
		if self.trainer.global_rank==0:
			if self.args.use_wandb:
				wandb.log(wandb_out)
				#pdb.set_trace()
		if self.args.save_cls and batch_idx < self.trainer.num_val_batches[0] - 1:
			self.save_output['cls_token_save'].append(cls_token.detach().cpu().squeeze())
			self.save_output['cls_label_save'].append(labels.detach().cpu().squeeze())
			self.save_output['cls_prob_save'].append(torch.sigmoid(logits.squeeze()).detach().cpu())

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

			#pdb.set_trace()

		if self.args.save_cls:
			torch.save(torch.stack(self.save_output['cls_token_save']),os.path.join(self.args.output_dir,'saved_cls'))
			torch.save(torch.stack(self.save_output['cls_label_save']),os.path.join(self.args.output_dir,'saved_label'))
		
		self.val_acc = 0.0
		self.val_loss = 0.0
	
		if self.trainer.global_rank==0 and self.args.n_viz:
			self.evaluator.plot_tsne(dim=self.config.hidden_size, saved_params=self.save_output)
	
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
	def __init__(self,  args, local_trainer=None, local_eval=0):
		
		self.args = args
		self.local_trainer = local_trainer
		if local_trainer:
			self.model = local_trainer.model
		else:
			self.model = None

		perplexity = 30.0
		if args.debug:
			perplexity = 5
		self.tsne = TSNE(perplexity=perplexity)

		if not self.args.save_fname:
			self.args.save_fname='viz.jpg'
		else:
			self.args.save_fname += '_viz.jpg'

	def evaluate(self):
		args = self.args
		model = self.model

	def plot_tsne(self, dim=780, saved_path=None, saved_params=None, model=None):
		
		n_viz = self.args.n_viz

		if saved_path:
			a = torch.load(saved_path+model+'/saved_cls')
			b = torch.load(saved_path+model+'/saved_label')
		else:
			cls_token = torch.stack(saved_params['cls_token_save'])
			cls_label = torch.stack(saved_params['cls_label_save'])
			cls_prob = torch.stack(saved_params['cls_prob_save'])

		# if self.args.debug:
		# 	pdb.set_trace()

		print('\n Plotting TSNE .... \n')
		embed = self.tsne.fit_transform(cls_token[0:n_viz].view(-1,dim))
		cls_pred = (cls_prob >= 0.5).float()
		cls_prob = (10*cls_prob).int()

		title = ['cls_GT_labels','cls_pred_labels','cls_prob']
		k=0
		
		fig, ax = plt.subplots(1, 3, figsize=(24,8), dpi=220)
		scatter = []

		for i,j in zip(ax,[cls_label,cls_pred,cls_prob]):
			scatter.append(i.scatter(embed[:, 0], embed[:, 1], c=j[0:n_viz].view(-1), cmap='jet', s=50, alpha=0.7))

			# Add a colorbar
			plt.colorbar(scatter[-1])

			i.set_title("t-SNE:: "+title[k])
			#i.xlabel("t-SNE Component 1")
			#i.ylabel("t-SNE Component 2")
			i.plot()
			i.set_aspect('equal')
			k+=1

		plt.savefig(os.path.join(self.args.output_dir,self.args.save_fname), bbox_inches='tight', pad_inches=0.1)