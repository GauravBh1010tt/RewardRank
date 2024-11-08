import os
import sys
import pdb
import tqdm
import torch
import wandb
import glob
import math
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import re
from sklearn.manifold import TSNE
import torch.nn as nn
from copy import deepcopy
import pytorch_lightning as pl
import torch.nn.functional as F
from src.bert import BertModel, BertReward, BertArranger
from scipy.interpolate import make_interp_spline
from scipy.ndimage.filters import gaussian_filter1d
from transformers.models.bert.configuration_bert import BertConfig
from src.ultr_models import infer_ultr
from src.losses import PiRank_Loss, neuralNDCG
from src.loss_utils import hard_sort_group_parallel as hard_sort

# from bbm.src.model import CrossEncoder, IPSCrossEncoder, PBMCrossEncoder

# from bbm.src.model import CrossEncoder

from src.utils import sample_without_replacement_with_prob as sample_perturb, sample_swap, distance_prob, soft_sort_group_parallel as soft_sort
from src.utils import eval_ultr, get_ndcg, binary_accuracy, merge_images

class local_trainer(pl.LightningModule):
	def __init__(self, train_loader, val_loader, test_dataset, args, eval_mode=False):
		super().__init__()

		self.config = BertConfig()
		self.args = args

		self.config.vocab = 100
		self.config.num_labels = 1 # regression output
		self.config.problem_type = args.problem_type
		self.config.max_position_embeddings = args.max_positions_PE
		self.config.use_word_embed = False

		if args.use_doc_feat:
		# 	self.config.doc_feat_len=0
			self.config.hidden_size+=12

		if not args.train_ranker_lambda:
			self.config.use_pos_embed = True
			self.reward_model = BertReward(self.config)
		
		if args.train_ranker:
			self.config.use_pos_embed = False
			#self.config.num_labels = args.max_items_QG # max_items in QGs
			self.config.concat_feats = self.args.concat_feats
			self.arranger = BertArranger(self.config)
			#print('\nLoading reward model ...\n')
			if args.train_ranker_lambda or args.eval:
				if args.load_path:
					self.resume(load_path=args.load_path, model='arranger')
			else:
				self.resume(load_path=args.load_path_reward, model='reward')
				if args.pretrain_ranker:
					self.resume(load_path=args.load_path_ranker, model='arranger')

		if args.eval_ultr:
			#from bbm.src.model import IPSCrossEncoder
			#self.ips_model = IPSCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_ips-pointwise")
			self.ips_model = None

		self.lr = args.lr
		self.weight_decay = args.weight_decay
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_dataset = test_dataset
		self.eval_mode = eval_mode
		self.tr_acc_reward, self.tr_loss_reward = 0.0, 0.0
		self.tr_acc_ranker, self.tr_loss_ranker = 0.0, 0.0

		self.val_acc = defaultdict(lambda: 0.0)
		self.val_loss = defaultdict(lambda: 0.0)
		# self.cls_token_save = []
		# self.cls_label_save = []
		# self.cls_prob_save = []

		self.save_output = {'cls_token_save':[], 'cls_label_save':[], 'cls_prob_save':[]}

		self.evaluator = Evaluator(args=self.args)

		# self.automatic_optimization = False
	
		# def forward(self, pixel_values, pixel_mask):
		# 	outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
		# 	return outputs

		# if self.args.perturbation_sampling and self.args.ultr_models:
		# 	if self.args.ultr_models == 'ips':
		# 		self.ultr_model = IPSCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_ips-pointwise")
		# 	else:
		# 		self.ultr_model = PBMCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_twotower")
	
	def common_step(self, batch, batch_idx):

		feat = torch.tensor(batch['query_document_embedding'], dtype=torch.float).to(self.device)
		doc_mask = torch.tensor(batch['mask'], dtype=torch.float).to(self.device)

		if self.args.eval_rels:	
			click = torch.tensor(batch['label']).to(self.device)
		else:
			click = torch.tensor(batch['click']).to(self.device)
			pos_idx = torch.tensor(batch['position']).to(self.device)

			if self.args.ultr_models:
				#pdb.set_trace()
				examination = torch.tensor(batch['examination_'+self.args.ultr_models]).to(self.device)
				relevance = torch.tensor(batch['relevance_'+self.args.ultr_models]).to(self.device)

		if self.args.perturbation_sampling and self.args.eval:
			for i,j in enumerate(pos_idx):
				n = batch['n'][i]

				if not self.args.ultr_models:
					if self.args.sampling_type == 'rand_perturb':
						pos_idx[i][0:n] = sample_perturb(delta=self.args.delta_retain, pos=pos_idx[i][0:n].float(), click=click[i])
					else:
						pos_idx[i][0:n] = sample_swap(pos=pos_idx[i][0:n].float(), click=click[i], fn=self.args.sampling_type)
				else:
					if self.args.sampling_type == 'rand_perturb':
						pos_idx[i][0:n] = sample_perturb(delta=self.args.delta_retain, pos=pos_idx[i][0:n].float())
					else:
						pos_idx[i][0:n] = sample_swap(pos=pos_idx[i][0:n].float(), click=[examination[i][0:n], relevance[i][0:n]], fn=self.args.sampling_type, ultr_mod=True)
					examination = infer_ultr(pos_idx=pos_idx, device=self.device)

		if self.args.ultr_models and not self.args.eval_rels: # ultr model predictions
			prob_click = examination * torch.sigmoid(relevance)
			prob_noclick = torch.prod(1-prob_click, dim=1) # padding is automatically handled by mul with 1.0
			prob_atleast_1click = 1 - prob_noclick
			avg_click = prob_atleast_1click
			#pdb.set_trace()
		elif self.args.soft_labels: # soft GT labels
			gt_binary_labels = torch.clamp(click.sum(dim=1), max=1.0)
			avg_click = (self.args.soft_base + click.sum(dim=1) * self.args.soft_gain)*gt_binary_labels
			avg_click = torch.clamp(avg_click, max=1.0).to(self.device)
		else: # hard GT labels
			avg_click = torch.clamp(click.sum(dim=1), max=1.0).to(self.device)

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

		#TODO: fix dummy index

		#pdb.set_trace()

		out_dict = defaultdict(lambda: {})

		if self.args.train_ranker:
			#pdb.set_trace()
			out_dict = self.arranger(inputs_embeds=feat, doc_feats=doc_feats, attention_mask=doc_mask)
			#pdb.set_trace()
			#logits = torch.sigmoid(out_dict['logits'])
			cls_token_arranger = out_dict['cls_token']

			logits = out_dict['logits']
			
			mask_padded = torch.arange(feat.size(1)).unsqueeze(0) >= torch.tensor(batch['n']).unsqueeze(1)
			mask_padded = mask_padded.to(self.device)
			#logits = out_dict['logits']

			if self.args.ultr_models and not self.args.eval_rels:
				#max_items = min(self.args.max_items_QG, prob_click.shape[1])
				labels = prob_click # TODO: relevance - ndcg 
			else:
				#max_items = min(self.args.max_items_QG, click.shape[1])
				labels = click

			loss_fn = PiRank_Loss()

			if self.args.train_ranker_lambda:
				#loss_fn = neuralNDCG
				loss = loss_fn(logits, labels.float(), device=self.device, dummy_indices=mask_padded)
				out_dict['ranker'] = {'loss':loss.mean(),'logits':logits,'labels':labels, 'cls_token':cls_token_arranger}
				return out_dict
			
			p_hat = soft_sort(s=logits, dummy_indices=mask_padded)
			
			if self.args.ste:
				perm_mat_backward = p_hat
				perm_mat_forward = hard_sort(logits, dummy_indices=mask_padded)
				p_hat = perm_mat_backward + (perm_mat_forward - perm_mat_backward).detach()

			#pdb.set_trace()
			ranker_loss = loss_fn(logits, labels.float(), device=self.device, dummy_indices=mask_padded)

			out = self.reward_model(inputs_embeds=feat, soft_position_ids=p_hat, labels=avg_click, doc_feats=doc_feats, attention_mask=doc_mask)
			out_dict['ranker'] = {'loss':ranker_loss.mean(),'logits':logits,'labels':labels, 'cls_token':cls_token_arranger}
		else:
			out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click, doc_feats=doc_feats, attention_mask=doc_mask)
		
		#eval_ultr(batch=batch, pred_scores=logits, ips_model=self.ips_model, device=self.device)
		out_dict['reward'] = {'loss':out['loss'],'logits':out['logits'],'labels':avg_click, 'cls_token':out['cls_token']}

		return out_dict
	
	
	def training_step(self, batch, batch_idx): # automatic training schedule
		
		out_dict = self.common_step(batch, batch_idx)

		use_soft=False

		if self.args.soft_labels or self.args.ultr_models:
			use_soft=True

		if self.args.train_ranker:
			acc_ranker = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), return_dcg=self.args.use_dcg)
			
			self.log("tr_loss_ranker", out_dict['ranker']['loss'], prog_bar=True, sync_dist=True)
			self.log("tr_acc_ranker", acc_ranker, prog_bar=True, sync_dist=True)

			if not self.args.train_ranker_lambda:
			# 	acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
			# 	self.log("tr_acc_reward", acc_reward, prog_bar=True, sync_dist=True)
			# 	self.log("tr_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True)

			# 	return out_dict['reward']['loss']
			
			# return out_dict['ranker']['loss']
				if not self.args.ultr_models:
					acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
					#self.val_acc['0.5_thres'] += acc_reward
					self.log("tr_acc_reward", acc_reward, prog_bar=True, sync_dist=True, batch_size=1)
				else:
					prob1 = torch.stack([1-torch.sigmoid(out_dict['reward']['logits'].squeeze()), torch.sigmoid(out_dict['reward']['logits'].squeeze())], dim=1) # batch x 2
					prob2 = torch.stack([1-out_dict['reward']['labels'], out_dict['reward']['labels']], dim=1)

					#pdb.set_trace()
					#for i in ['tv']:
					tv_reward = distance_prob(prob1, prob2, distance_type="tv")
					self.val_acc['tv'] += tv_reward
					self.log("tr_TV_reward", tv_reward, prog_bar=True, sync_dist=True, batch_size=1)

				self.log("tr_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True)
				
				if self.args.cls_reg:

					cls_loss = nn.MSELoss()

					loss_cls = self.args.cls_reg_lr * cls_loss(out_dict['reward']['cls_token'], out_dict['ranker']['cls_token'])
					loss = out_dict['reward']['loss'] + loss_cls
					self.log("tr_loss_cls", loss_cls, prog_bar=True, sync_dist=True, batch_size=1)
				else:
					loss = out_dict['reward']['loss']
				return loss
			
			return out_dict['ranker']['loss']
		else:
			#acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
			self.log("tr_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True)

			if not self.args.ultr_models:
				acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
				#self.val_acc['0.5_thres'] += acc_reward
				self.log("tr_acc_reward", acc_reward, prog_bar=True, sync_dist=True)
			else:
				prob1 = torch.stack([1-torch.sigmoid(out_dict['reward']['logits'].squeeze()), torch.sigmoid(out_dict['reward']['logits'].squeeze())], dim=1) # batch x 2
				prob2 = torch.stack([1-out_dict['reward']['labels'], out_dict['reward']['labels']], dim=1)

				tv_reward = distance_prob(prob1, prob2, distance_type="tv")
				#self.val_acc['tv'] += tv_reward
				self.log("tr_TV_reward", tv_reward, prog_bar=True, sync_dist=True, batch_size=1)

			return out_dict['reward']['loss']

		#pdb.set_trace()

		#wandb_out = {"tr_loss": loss, "tr_acc":acc}

		# self.tr_acc += float(acc)
		# self.tr_loss += float(loss)

	# def on_after_backward(self):
	# 	self.arranger.classifier.weight.grad[self.batch_max_items:,:] = 0.0
	# 	self.arranger.classifier.bias.grad[self.batch_max_items:] = 0.0
		#pdb.set_trace()

	def on_train_epoch_end(self):

		# tr_acc_avg = self.tr_acc/self.trainer.num_training_batches
		# tr_loss_avg = self.tr_loss/self.trainer.num_training_batches

		self.lr_scheduler.step()
		if self.current_epoch and self.current_epoch%self.args.save_epochs == 0:
			self.save(self.current_epoch)
			
		# if self.trainer.global_rank==0:
		# 	print('\n Train acc after ', self.current_epoch, ' epochs : ',\
		# 					tr_acc_avg, '  loss : ',tr_loss_avg, file=self.args.log_file)
			

		# 	wandb_out = {"tr_loss_global": tr_acc_avg, "tr_acc_global":tr_loss_avg}
			
		# 	if self.args.use_wandb:
		# 		wandb.log(wandb_out)
		
		# self.tr_acc = 0.0
		# self.tr_loss = 0.0

	def validation_step(self, batch, batch_idx):
		
		out_dict = self.common_step(batch, batch_idx)
		#loss = out_dict['loss']
		#logits = out_dict['logits']

		if not self.args.train_ranker:
			cls_token = out_dict['reward']['cls_token']

		use_soft=False

		if self.args.soft_labels or self.args.ultr_models:
			use_soft=True

		if self.args.eval_ultr:
			acc_ranker = eval_ultr(batch=batch, pred_scores=out_dict['ranker']['logits'],
							device=self.device)
			self.val_acc['prob_atleast_1click'] += acc_ranker
			self.log("val_acc_prob_click", acc_ranker, prog_bar=True, sync_dist=True, batch_size=1)
		else:
			if self.args.train_ranker:
				
				acc_ranker = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), 
						return_dcg=self.args.use_dcg, mask=batch['mask']*1, use_rax=self.args.use_rax) # mask will be used for RAX. automatically fixed for torchmetrics
				
				self.val_acc['ndcg'] += acc_ranker
				self.val_loss['ranker'] += out_dict['ranker']['loss']

				self.log("val_loss_ranker", out_dict['ranker']['loss'], prog_bar=True, sync_dist=True,  batch_size=1)
				self.log("val_acc_ndcg_ranker", acc_ranker, prog_bar=True, sync_dist=True,  batch_size=1)

				if not self.args.train_ranker_lambda:
					# acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
					# self.log("val_acc_reward", acc_reward, prog_bar=True, sync_dist=True,  batch_size=1)
					# self.log("val_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True,  batch_size=1)

					if not self.args.ultr_models:
						acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
						self.val_acc['0.5_thres'] += acc_reward
						self.log("val_acc_reward", acc_reward, prog_bar=True, sync_dist=True, batch_size=1)
					else:
						prob1 = torch.stack([1-torch.sigmoid(out_dict['reward']['logits'].squeeze()), torch.sigmoid(out_dict['reward']['logits'].squeeze())], dim=1) # batch x 2
						prob2 = torch.stack([1-out_dict['reward']['labels'], out_dict['reward']['labels']], dim=1)

						#pdb.set_trace()
						#for i in ['tv']:
						tv_reward = distance_prob(prob1, prob2, distance_type="tv")
						self.val_acc['tv'] += tv_reward
						self.log("val_TV_reward", tv_reward, prog_bar=True, sync_dist=True, batch_size=1)
						
						
						acc_ranker = eval_ultr(batch=batch, pred_scores=out_dict['ranker']['logits'],
							 					device=self.device)
						self.val_acc['prob_atleast_1click'] += acc_ranker
						self.log("val_acc_prob_click", acc_ranker, prog_bar=True, sync_dist=True, batch_size=1)
					
					self.log("val_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True, batch_size=1)
			else:
				#pdb.set_trace()
				self.log("val_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True, batch_size=1)
				self.val_loss['reward'] += out_dict['reward']['loss']

				if not self.args.ultr_models:
					acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
					self.val_acc['0.5_thres'] += acc_reward
					self.log("val_acc_reward", acc_reward, prog_bar=True, sync_dist=True, batch_size=1)
				else:
					prob1 = torch.stack([1-torch.sigmoid(out_dict['reward']['logits'].squeeze()), torch.sigmoid(out_dict['reward']['logits'].squeeze())], dim=1) # batch x 2
					prob2 = torch.stack([1-out_dict['reward']['labels'], out_dict['reward']['labels']], dim=1)

					#pdb.set_trace()
					#for i in ['tv']:
					tv_reward = distance_prob(prob1, prob2, distance_type="tv")
					self.val_acc['tv'] += tv_reward
					self.log("val_TV_reward", tv_reward, prog_bar=True, sync_dist=True, batch_size=1)

		# if self.args.debug:
		# 	if batch_idx>10:
		# 		return

		# wandb_out = {"val_loss": loss, "val_acc":acc}

		# if self.args.eval_ultr:
		# 		self.log("val_loss", loss, batch_size=self.args.batch_size, prog_bar=True, sync_dist=True)
		# 		self.log("val_prob_1click@IPS", acc, batch_size=self.args.batch_size, prog_bar=True, sync_dist=True)
		# else:
		# 	if self.args.train_ranker:
		# 		self.log("val_loss", loss, batch_size=self.args.batch_size, prog_bar=True, sync_dist=True)
		# 		self.log("val_dcg@10", acc, batch_size=self.args.batch_size, prog_bar=True, sync_dist=True)
		# 	else:
		# 		self.log("val_loss", loss, batch_size=self.args.batch_size, prog_bar=True, sync_dist=True)
		# 		self.log("val_acc", acc, batch_size=self.args.batch_size, prog_bar=True, sync_dist=True)

		# #self.val_acc += float(acc)
		# self.val_loss += float(loss)
		
		# if self.trainer.global_rank==0:
		# 	if self.args.use_wandb:
		# 		wandb.log(wandb_out)
		# 		#pdb.set_trace()
		# #pdb.set_trace()

		if self.args.save_cls:
			if batch_idx < self.trainer.num_val_batches[0] - 1:
				self.save_output['cls_token_save'].extend(cls_token.detach().cpu().squeeze())
				self.save_output['cls_label_save'].extend(out_dict['reward']['labels'].detach().cpu().squeeze())
				self.save_output['cls_prob_save'].extend(torch.sigmoid(out_dict['reward']['logits'].squeeze()).detach().cpu())

		return
	
	def on_validation_epoch_end(self):

		#pdb.set_trace()
		val_acc_avg = defaultdict(lambda: float)

		for key in self.val_acc.keys():
			acc = self.val_acc[key]/self.trainer.num_val_batches[0]

		if self.trainer.global_rank==0:
			print('\n Val acc "',key ,'" after ', self.current_epoch, ' epochs : ',\
							acc, file=self.args.log_file)

		if self.args.save_cls:
			
			Path(os.path.join(self.args.output_dir,'saved_tensor')).mkdir(parents=True, exist_ok=True)

			torch.save(torch.stack(self.save_output['cls_token_save']),os.path.join(self.args.output_dir,'saved_tensor','saved_cls'))
			torch.save(torch.stack(self.save_output['cls_label_save']),os.path.join(self.args.output_dir,'saved_tensor','saved_label'))
			torch.save(torch.stack(self.save_output['cls_prob_save']),os.path.join(self.args.output_dir,'saved_tensor','saved_prob'))
	
		if self.trainer.global_rank==0 and self.args.n_viz:
			#print (self.args.n_viz)
			if self.args.eval or self.current_epoch==35:
				self.evaluator.plot_tsne(dim=self.config.hidden_size, saved_params=self.save_output, val_acc=0.0)
		
		self.val_acc = defaultdict(lambda: 0.0)
		self.val_loss = defaultdict(lambda: 0.0)
		self.save_output = {'cls_token_save':[], 'cls_label_save':[], 'cls_prob_save':[]}
	
	def save(self, epoch):
		if self.trainer.global_rank==0:
			print('\n Saving at epoch ', epoch, file=self.args.log_file)

		Path(os.path.join(self.args.output_dir,'checkpoints')).mkdir(parents=True, exist_ok=True)

		if self.args.train_ranker:
			model = self.arranger
		else:
			model = self.reward_model

		torch.save({
					'model': model.state_dict(),
					'optimizer': self.optimizer.state_dict(),
					'lr_scheduler': self.lr_scheduler.state_dict(),
					'epoch': epoch,
					#'args': self.args,
				}, os.path.join(self.args.output_dir,'checkpoints', f'checkpoint{epoch:02}.pth'))
	
	def resume(self, load_path='', model='reward'):
		
		#if self.trainer.global_rank==0:
		print('\n Resuming ',model,'_model from : ',load_path)
		print('\n Resuming ',model,'_model from : ',load_path, file=self.args.log_file)
		
		if load_path:
			checkpoint = torch.load(load_path, map_location='cpu')

			if model=='reward':
				missing_keys, unexpected_keys = self.reward_model.load_state_dict(checkpoint['model'], strict=False)

				if self.args.train_ranker:
					for id, (name, params) in enumerate(self.reward_model.named_parameters()):
						params.requires_grad = False
			else:
				missing_keys, unexpected_keys = self.arranger.load_state_dict(checkpoint['model'], strict=False)

	
	def configure_optimizers(self):

		if self.args.train_ranker:
			self.optimizer = torch.optim.AdamW(self.arranger.parameters(), lr=self.lr,
								weight_decay=self.weight_decay)
		else:
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
			self.save_fname='org_viz.jpg'
			self.local_save = 'Mod:\n'+args.output_folder
		else:
			self.local_save = args.save_fname
			self.save_fname = args.save_fname+'_viz.jpg'

		self.save_path = os.path.join(args.output_dir,'figs')

		Path(self.save_path).mkdir(parents=True, exist_ok=True)

	def evaluate(self):
		args = self.args
		model = self.model

	def plot_train_val(self, log_file_path, output_image_path='train_val_accuracy_loss.jpg'):
		"""
		Function to plot training and validation accuracy and loss from a log file.
		
		Parameters:
		log_file_path (str): Path to the log file containing accuracy and loss data.
		output_image_path (str): Path to save the output image file.
		"""
		
		# Initialize lists to store extracted values
		epochs = []
		train_acc = []
		val_acc = []
		train_loss = []
		val_loss = []

		# Regular expressions to extract the relevant values
		val_acc_pattern = r'Val acc after\s+(\d+)\s+epochs\s+:\s+([\d\.]+)\s+loss\s+:\s+([\d\.]+)'
		train_acc_pattern = r'Train acc after\s+(\d+)\s+epochs\s+:\s+([\d\.]+)\s+loss\s+:\s+([\d\.]+)'

		# Read the log file
		with open(log_file_path, 'r') as f:
			log_data = f.readlines()

		# Parse the log data
		for line in log_data:
			val_acc_match = re.search(val_acc_pattern, line)
			train_acc_match = re.search(train_acc_pattern, line)

			if val_acc_match:
				epoch = int(val_acc_match.group(1))
				val_acc.append(float(val_acc_match.group(2)))
				val_loss.append(float(val_acc_match.group(3)))
				epochs.append(epoch)

			if train_acc_match:
				train_acc.append(float(train_acc_match.group(2)))
				train_loss.append(float(train_acc_match.group(3)))

		# Create a figure and set the size
		plt.figure(figsize=(12, 10))

		# Plot accuracy (subplot 1)
		plt.subplot(2, 1, 1)
		plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
		plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.title('Train vs Validation Accuracy')
		plt.legend()
		plt.grid(True)

		# Plot loss (subplot 2)
		plt.subplot(2, 1, 2)
		plt.plot(epochs, train_loss, label='Train Loss', marker='o')
		plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('Train vs Validation Loss')
		plt.legend()
		plt.grid(True)

		# Adjust the layout to avoid overlapping
		plt.tight_layout()

		# Save the combined figure as a JPG file
		plt.savefig(output_image_path, format='jpg')

		# Show the plot (optional, can be commented out if not needed)
		plt.show()

	def plot_tsne(self, dim=780, saved_path=None, saved_params=None, model=None, val_acc=0.0):
		
		n_viz = self.args.n_viz

		if saved_path:
			a = torch.load(saved_path+model+'/saved_cls')
			b = torch.load(saved_path+model+'/saved_label')
		else:
			cls_token = torch.stack(saved_params['cls_token_save'])
			cls_label = torch.stack(saved_params['cls_label_save'])
			cls_prob = torch.stack(saved_params['cls_prob_save'])
			cls_prob_org = torch.load(os.path.join(self.args.output_dir,'saved_tensor', 'saved_prob'))
			cls_label_org = torch.load(os.path.join(self.args.output_dir,'saved_tensor', 'saved_label'))

		# if self.args.debug:
		# 	pdb.set_trace()

		if not self.args.save_cls and os.path.exists(os.path.join(self.args.output_dir,'saved_tensor', 'saved_tnse_embed')) and not self.args.force_tnse:
			print('\n Loading TSNE embed .... \n')
			embed = torch.load(os.path.join(self.args.output_dir,'saved_tensor', 'saved_tnse_embed'))
		else:
			print('\n Generating TSNE embed .... \n')
			embed = self.tsne.fit_transform(cls_token[0:n_viz].view(-1,dim))
			torch.save(torch.tensor(embed),os.path.join(self.args.output_dir,'saved_tensor','saved_tnse_embed'))

		#cls_pred = (cls_prob >= 0.5).float()

		#pdb.set_trace()
		prob1 = torch.stack([1-cls_prob, cls_prob], dim=1) # batch x 2
		#prob2 = torch.stack([1-cls_prob_org, cls_prob_org], dim=1)
		prob2 = torch.stack([1-cls_label, cls_label], dim=1)

		accuracy, score_shift = defaultdict(lambda: float), defaultdict(lambda: float)

		# for i in ['l1','l2','mse','kl']:
		# 	score_shift[i] =  math.floor(distance_prob(prob1, prob2, distance_type=i) * 100) / 100

		# 	if self.args.ultr_models:
		# 		accuracy[i] = math.floor(val_acc[i] * 10) / 10


		score_shift['tv'] =  math.floor(distance_prob(prob1, prob2, distance_type='tv') * 100) / 100

		if self.args.ultr_models:
			accuracy['tv'] = math.floor(val_acc['tv'] * 100) / 100
		else:
			accuracy['0.5_thres'] = math.floor(val_acc['0.5_thres'] * 10) / 10

		fig, ax = plt.subplots(1, 5, figsize=(50,10), dpi=220)

		#pdb.set_trace()

		# if not self.args.ultr_models:
		# 	accuracy['0.5_thres'] = math.floor(val_acc['0.5_thres'] * 10) / 10
		# 	fig.text(0.04, 0.5, self.local_save+'\n\n Val_performance: \n'+str(accuracy['0.5_thres'])+'\n\n Score_shift: \n KL: '+str(score_shift['kl'])+'\n L1: '+\
		# 														str(score_shift['l1'])+'\n L2: '+str(score_shift['l2'])+'\n MSE: '+str(score_shift['mse']), 
		#    														va='center', fontsize=16, fontweight='bold')
		# else:
		# 	fig.text(0.04, 0.5, self.local_save+'\n\n Val_performance: \n KL: '+str(accuracy['kl'])+'\n L1: '+str(accuracy['l1'])+'\n L2: '+str(accuracy['l2'])+'\n MSE: '+str(accuracy['mse'])+\
		# 										'\n\n Score_shift: \n KL: '+str(score_shift['kl'])+'\n L1: '+str(score_shift['l1'])+'\n L2: '+str(score_shift['l2'])+'\n MSE: '+\
		# 											str(score_shift['mse']), va='center', fontsize=16, fontweight='bold')

		if not self.args.ultr_models:
			fig.text(0.04, 0.5, self.local_save+'\n\n Val_perf %: \n'+str(accuracy['0.5_thres'])+'\n\n Score_shift %: \n TV: '+str(score_shift['tv']),
		   														va='center', fontsize=16, fontweight='bold')
		else:
			fig.text(0.04, 0.5, self.local_save+'\n\n Val_perf %: \n TV: '+str(accuracy['tv'])+'\n\n Score_shift %: \n TV: '+str(score_shift['tv']), 
																va='center', fontsize=16, fontweight='bold')


		# if self.args.save_cls:
		# 	fig.text(0.04, 0.5, self.args.model+'\nval_acc: '+str(accuracy)+'\n KL_div: '+str(kl_div), va='center', fontsize=16, fontweight='bold')
		# else:

		plt.subplots_adjust(wspace=0.04, hspace=0.1)

		title = ['cls_GT_labels','cls_pred_labels','cls_prob']
		k=0
		
		scatter = []

		for i,j in zip(ax,[cls_label,cls_prob]):
			scatter.append(i.scatter(embed[:, 0], embed[:, 1], c=j[0:n_viz].view(-1), cmap='jet', s=50, alpha=0.7))

			# Add a colorbar
			#plt.colorbar(scatter[-1])
			plt.colorbar(scatter[-1], shrink=1.0)

			i.set_title("t-SNE:: "+title[k])
			#i.xlabel("t-SNE Component 1")
			#i.ylabel("t-SNE Component 2")
			i.plot()
			#i.set_aspect('equal')
			k+=1

		#pdb.set_trace()

		idx_y = torch.argsort(cls_label_org[:n_viz].view(-1))

		gt_scores = cls_label_org[:n_viz].view(-1)[idx_y]
		pred_scores = cls_label[:n_viz].view(-1)[idx_y]

		diff_y = cls_label_org[:n_viz].view(-1) - cls_label[:n_viz].view(-1)
		idx_delta = torch.argsort(diff_y)

		diff_y = diff_y[idx_delta]

		scatter_x = np.array(range(len(gt_scores)))
		scatter_y = np.array(pred_scores)

		# Define points that represent the line
		line_x = np.array(range(len(gt_scores)))
		line_y = np.array(gt_scores)  # Example of line points

		# Calculate the difference (distance) from the line for each scatter point
		# Euclidean distance between scatter points and the nearest line points
		distances = np.sqrt((scatter_x - line_x)**2 + (scatter_y - line_y)**2)

		# Create the scatter plot, with color based on distance from the line
		#plt.figure(figsize=(8, 6))
		scatter_plot = ax[2].scatter(scatter_x, scatter_y, c=distances, cmap='coolwarm', s=40)

		ax[2].plot(line_x, line_y, color='black', linestyle='--', marker='o', label='Line Points')
		ax[2].set_title("Shift score distibution ULTR : y|y'")

		# Add a color bar to show the distance scale
		plt.colorbar(scatter_plot, label='Distance from Line Points')

		#####################################################################################################

		idx_p = torch.argsort(cls_prob_org[:n_viz].view(-1))

		gt_scores = cls_prob_org[:n_viz].view(-1)[idx_p]
		pred_scores = cls_prob[:n_viz].view(-1)[idx_p]

		gt_scores_wrt_y = cls_prob_org[:n_viz].view(-1)[idx_delta]
		pred_scores_wrt_y = cls_prob[:n_viz].view(-1)[idx_delta]

		diff_p = gt_scores_wrt_y - pred_scores_wrt_y

		scatter_x = np.array(range(len(gt_scores)))
		scatter_y = np.array(pred_scores)

		# Define points that represent the line
		line_x = np.array(range(len(gt_scores)))
		line_y = np.array(gt_scores)  # Example of line points

		# Calculate the difference (distance) from the line for each scatter point
		# Euclidean distance between scatter points and the nearest line points
		distances = np.sqrt((scatter_x - line_x)**2 + (scatter_y - line_y)**2)

		# Create the scatter plot, with color based on distance from the line
		#plt.figure(figsize=(8, 6))
		scatter_plot = ax[3].scatter(scatter_x, scatter_y, c=distances, cmap='coolwarm', s=40)

		#pdb.set_trace()

		# Plot the line using the points
		ax[3].plot(line_x, line_y, color='black', linestyle='--', marker='o', label='Line Points')
		ax[3].set_title("Shift score distibution Reward : p|p'")

		# Add a color bar to show the distance scale
		plt.colorbar(scatter_plot, label='Distance from Line Points')

		#####################################################################################################


		scatter_x = np.array(range(len(diff_p)))
		scatter_y = np.array(diff_p)

		# Define points that represent the line
		line_x = np.array(range(len(diff_y)))
		line_y = np.array(diff_y)  # Example of line points

		# Calculate the difference (distance) from the line for each scatter point
		# Euclidean distance between scatter points and the nearest line points
		distances = np.sqrt((scatter_x - line_x)**2 + (scatter_y - line_y)**2)

		# Create the scatter plot, with color based on distance from the line
		#plt.figure(figsize=(8, 6))
		scatter_plot = ax[4].scatter(scatter_x, scatter_y, c=distances, cmap='coolwarm', s=40)

		#pdb.set_trace()

		# Plot the line using the points
		ax[4].plot(line_x, line_y, color='black', linestyle='--', marker='o', label='Line Points')
		ax[4].set_title("Shift score delta : (y-y') | (p-p')")

		# Add a color bar to show the distance scale
		plt.colorbar(scatter_plot, label='Distance from Line Points')

		#####################################################################################################

		plt.savefig(os.path.join(self.save_path,self.save_fname), bbox_inches='tight', pad_inches=0.1)

		print ('\n saving... \n', os.path.join(self.save_path,self.save_fname))

		if self.args.merge_imgs:
			all_files = glob.glob(self.save_path+'/*')
			all_files.sort()
			merge_images(all_files, output_path=os.path.join(self.args.output_dir,'eval_viz.jpg'),direction='vertical')