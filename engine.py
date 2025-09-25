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
import pandas as pd
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
from src.losses import PiRank_Loss, listMLE, listNet, listwise_softmax_ips, pointwise_sigmoid_ips, lambdaLoss
from src.loss_utils import hard_sort_group_parallel as hard_sort

from src.utils import distance_prob, soft_sort_group_parallel as soft_sort
from src.utils import eval_ultr, get_ndcg, binary_accuracy, loss_urcc, eval_ultr_ideal
from src.pg_rank import compute_pg_rank_loss

class local_trainer(pl.LightningModule):
	def __init__(self, train_loader, val_loader, test_dataset, args, eval_mode=False):
		super().__init__()

		self.config = BertConfig()
		self.args = args

		self.config.vocab = 100
		self.config.num_labels = 1 # classification output
		self.config.problem_type = args.problem_type
		self.config.max_position_embeddings = args.max_positions_PE
		self.config.use_word_embed = False
		self.config.per_item_feats = args.per_item_feats
		self.config.concat_feats = self.args.concat_feats
		self.config.lin_pos = self.args.lin_pos
		if self.args.ultr_models:
			self.config.ips_exp = True
		else:
			self.config.ips_exp = False
		self.config.reward_loss_type = args.reward_loss_type
		self.config.reward = False
	
		if args.use_doc_feat:
			self.config.hidden_size+=12

		if not args.train_ranker_naive:
			self.config.use_pos_embed = True
			self.reward_model = BertReward(self.config)
			self.config.reward = True
		
		if args.train_ranker:
			self.config.use_pos_embed = False
			self.arranger = BertArranger(self.config)
			if args.train_ranker_naive or args.eval:
				if args.load_path:
					self.resume(load_path=args.load_path, model='arranger')
			else:
				self.resume(load_path=args.load_path_reward, model='reward')
				if args.pretrain_ranker:
					self.resume(load_path=args.load_path_ranker, model='arranger')
		elif args.load_path_reward:
			self.resume(load_path=args.load_path_reward, model='reward')

		if args.eval_ultr:
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
		self.stats = defaultdict(lambda: [])
		self.save_output = {'reward_pred':[], 'pur_prob':[], 'reward_prod':[], 'reward_correction_wt':[]}
		#self.evaluator = Evaluator(args=self.args)


	def common_step(self, batch, batch_idx):

		feat = torch.tensor(batch['query_document_embedding'], dtype=torch.float).to(self.device)
		doc_mask = torch.tensor(batch['mask'], dtype=torch.float).to(self.device)

		#pos_idx = torch.arange(batch['position'].shape[1]).repeat(batch['position'].shape[0],1).to(self.device) # contigous positions starting from 0

		#print (self.trainer.validating)
		#pdb.set_trace()
		if self.args.llm_exp:
			purchase_prob = torch.tensor(batch['purchase_prob'], dtype=torch.float).to(self.device)
			#pos_idx = 1 + torch.arange(batch['position'].shape[1]).repeat(batch['position'].shape[0],1).to(self.device)
			pos_idx = torch.arange(batch['position'].shape[1]).repeat(batch['position'].shape[0],1).to(self.device)
		elif self.args.use_org_feats and self.trainer.validating:
			click = torch.tensor(batch['label']).to(self.device)
			#pos_idx = torch.arange(batch['label'].shape[1]).repeat(batch['label'].shape[0],1).to(self.device)
		else:
			click = torch.tensor(batch['click']).to(self.device)
			pos_idx = torch.tensor(batch['position']).to(self.device)
			#pos_idx = 1 + torch.arange(batch['position'].shape[1]).repeat(batch['position'].shape[0],1).to(self.device)

			if self.args.ultr_models:
				#examination = infer_ultr(pos_idx=1+pos_idx, device=self.device)
				#pdb.set_trace()
				examination = torch.tensor(batch['examination_'+self.args.ultr_models]).to(self.device)
				relevance = torch.tensor(batch['relevance_'+self.args.ultr_models]).to(self.device)

		if self.args.llm_exp:
			avg_click = purchase_prob
		elif self.args.ultr_models and not self.args.eval_rels: # ultr model predictions
			prob_click = examination * torch.sigmoid(relevance)
			prob_noclick = torch.prod(1-prob_click, dim=1) # padding is automatically handled by mul with 1.0
			prob_atleast_1click = 1 - prob_noclick
			avg_click = prob_atleast_1click
		elif self.args.soft_labels: # soft GT labels
			gt_binary_labels = torch.clamp(click.sum(dim=1), max=1.0)
			avg_click = (self.args.soft_base + click.sum(dim=1) * self.args.soft_gain)*gt_binary_labels
			avg_click = torch.clamp(avg_click, max=1.0).to(self.device)
		else: # hard GT labels
			#pdb.set_trace()
			#print ('here')
			click = click*doc_mask
			avg_click = torch.clamp(click.sum(dim=1), max=1.0).to(self.device)

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
			
			doc_feats = torch.cat([doc_feats, torch.zeros((batch['bm25'].shape[0],batch['bm25'].shape[1],5)).to(self.device)],dim=2)	
			feat = torch.cat([feat,doc_feats], dim=2) #TODO: Pass doc_feats through MLP before concat

		out_dict = defaultdict(lambda: {})
  
		if self.args.llm_exp:
			num_items = 8
			one_hot = np.eye(num_items)[batch['item_position']]
			one_hot = torch.tensor(one_hot).to(self.device)
			labels_per_item = one_hot
			labels = one_hot	
		elif self.args.ultr_models and not self.args.eval_rels:
			labels_per_item = prob_click
			labels = prob_click
		else:
			labels_per_item = click
			labels = click

		valid_f = torch.ones(feat.size(0), device=self.device)   
  
		if self.args.llm_exp:
			num_items = 8
			
			# one_hot = np.eye(num_items)[batch['item_position']]
			# one_hot = torch.tensor(one_hot).to(self.device)
			# labels_per_item = one_hot
			# labels = one_hot	

			positions = torch.as_tensor(batch['item_position'], device=self.device)   # shape: [B]
			B = positions.size(0)
			valid = (positions >= 0)
			valid_f = valid.float()
			one_hot = torch.zeros(B, num_items, device=self.device)                   # [B, C]

			if valid.any():
				one_hot[valid] = torch.nn.functional.one_hot(
					positions[valid], num_classes=num_items
				).float()
			
			labels = one_hot.float()
			labels_per_item = one_hot.float()

		elif self.args.ultr_models and not self.args.eval_rels:
			labels_per_item = prob_click
			labels = prob_click
			#pdb.set_trace()
			sampled_clicks = torch.bernoulli(prob_click)*doc_mask
		else:
			labels_per_item = click
			labels = click
			sampled_clicks = click
			#pdb.set_trace()

		if self.args.train_ranker:
			out_dict = self.arranger(inputs_embeds=feat, doc_feats=doc_feats, attention_mask=doc_mask)
			cls_token_arranger = out_dict['cls_token']

			#pdb.set_trace()

			logits = out_dict['logits']

			if self.args.llm_exp:
				mask_padded = torch.arange(feat.size(1)).unsqueeze(0) >= torch.tensor([8]*feat.size(0)).unsqueeze(1)
				mask_padded = mask_padded.to(self.device)
			else:			
				mask_padded = torch.arange(feat.size(1)).unsqueeze(0) >= torch.tensor(batch['n']).unsqueeze(1)
				mask_padded = mask_padded.to(self.device)
			
			if self.args.rank_loss == "pirank":
				loss_fn = PiRank_Loss()
			elif self.args.rank_loss == "list_net":
				loss_fn = listNet
			elif self.args.rank_loss == "list_mle":
				loss_fn = listMLE
			elif self.args.rank_loss == "ips_point":
				loss_fn = pointwise_sigmoid_ips
			elif self.args.rank_loss == 'ips_list':
				loss_fn = listwise_softmax_ips
			elif self.args.rank_loss == 'lambdarank':
				loss_fn = lambdaLoss

			if self.args.train_ranker_naive:
				#loss_fn = neuralNDCG
				if self.args.ips_sampling:
					#pdb.set_trace()
					if self.args.rank_loss in ["ips_point", 'ips_list']:
						exam = torch.tensor(batch['examination_'+self.args.ultr_models]).to(self.device)
						#pdb.set_trace()
						loss = loss_fn(logits, examination_probs=exam, clicks=sampled_clicks.float(), 
					 						query_ids=batch['query_id'], dummy_indices=mask_padded, device=self.device)
					else:
						loss = loss_fn(logits, sampled_clicks.float(), device=self.device, dummy_indices=mask_padded)
					out_dict['ranker'] = {'loss':loss.mean(),'logits':logits,'labels':labels, 'cls_token':cls_token_arranger}
				else:
					loss = loss_fn(logits, labels.float(), device=self.device, dummy_indices=mask_padded)
					out_dict['ranker'] = {'loss':loss.mean(),'logits':logits,'labels':labels, 'cls_token':cls_token_arranger}
				return out_dict
			
			if self.args.urcc_loss:
				loss = loss_urcc(logits, labels.float(), device=self.device, mask = mask_padded, reward_mod=self.reward_model,
					 			inputs_embeds=feat, position_ids=pos_idx, doc_feats=doc_feats, 
						   		labels_click = labels_per_item, attention_mask=doc_mask, avg_lables=avg_click)
			
			if self.args.pgrank_loss:
				# if not self.args.pgrank_disc:
				# 	loss = compute_pg_rank_loss(scores=logits, targets=labels, args=self.args, 
				# 					mask=mask_padded, device=self.device)
				# else:
				loss = compute_pg_rank_loss(scores=logits, targets=labels, args=self.args, 
							mask=mask_padded, device=self.device, reward_mod=self.reward_model,
							inputs_embeds=feat, position_ids=pos_idx, doc_feats=doc_feats, 
							labels_click = labels_per_item, attention_mask=doc_mask)
				
			if self.args.pgrank_loss or self.args.urcc_loss or self.args.train_ranker_naive: # don't need reward maximization
				out_dict['ranker'] = {'loss':loss.mean(),'logits':logits,'labels':labels, 'cls_token':cls_token_arranger}
				return out_dict
			elif self.args.use_org_feats and self.trainer.validating:
				out_dict['ranker'] = {'loss':torch.tensor([0.0], device=self.device),'logits':logits,'labels':labels, 'cls_token':cls_token_arranger}
				return out_dict
			else:	# reward maximization
				p_hat = soft_sort(s=logits, dummy_indices=mask_padded, 
					  temperature=self.args.soft_sort_temp) # rows = items; columns = positions; row[0] = i1*p(i[1]) + i2*p(i[2]) ....
				
				pdb.set_trace()
				if self.args.ste:
					perm_mat_backward = p_hat
					perm_mat_forward = hard_sort(logits, dummy_indices=mask_padded)
					p_hat = perm_mat_backward + (perm_mat_forward - perm_mat_backward).detach()

				loss_fn = PiRank_Loss()
				ranker_loss = loss_fn(logits, labels.float(), device=self.device, dummy_indices=mask_padded)
				
				# if self.args.llm_exp:
				# 	#pdb.set_trace()
				# 	ranker_loss_mean = (ranker_loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)
				# else:
				# 	ranker_loss_mean = ranker_loss.mean()

				#with torch.no_grad():
				out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, 
							soft_position_ids=p_hat, labels=avg_click, doc_feats=doc_feats, attention_mask=doc_mask)
	
				# out_dict['ranker'] = {'loss':ranker_loss_mean,'loss_samples':ranker_loss, 'logits':logits,'labels':labels, 'cls_token':cls_token_arranger}
				out_dict['ranker'] = {'loss':ranker_loss.mean(), 'logits':logits,'labels':labels, 'cls_token':cls_token_arranger}
		else:
			# if self.args.use_label_weights:
			# 	label_wt = batch['label_weights']
			# 	label_wt = torch.tensor(label_wt, dtype=torch.float).to(self.device)
			# else:
			# 	label_wt = None

			if self.args.ips_sampling: # checking for masking
				avg_click = torch.clamp(sampled_clicks.sum(dim=1), max=1.0).to(self.device) # P(C>=0) = 0/1
				out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click, doc_feats=doc_feats, 
						   labels_click = sampled_clicks, attention_mask=doc_mask)
			else:
				out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click, doc_feats=doc_feats, 
							labels_click = labels_per_item, attention_mask=doc_mask)
			
		out_dict['reward'] = {'loss':out['loss'],'logits':out['logits'],'labels':avg_click, 'cls_token':out['cls_token'], 
						'logits_per_item': out['per_item_logits'], 'loss_per_item': out['per_item_loss']}

		if self.args.reward_correction:
			with torch.no_grad():
				out_prod = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click, doc_feats=doc_feats, 
							   labels_click = labels_per_item, attention_mask=doc_mask)
			rewards_prod = torch.sigmoid(out_prod['logits'])
			#epsilon = 1e-6  # Small constant for numerical stability
			residual = torch.abs(avg_click - rewards_prod)
			#correction_wt =  torch.abs(rewards_prod - self.args.residual_coef) * residual) / rewards_prod
			correction_wt = 1 - (self.args.residual_coef * residual)
			out_dict['reward']['correction_wt'] = correction_wt
			out_dict['reward']['logits_prod'] = out_prod['logits']

			ranker_loss_proxy = (ranker_loss * valid_f * (1-correction_wt)).sum() / valid_f.sum().clamp_min(1.0)
			out_dict['ranker']['loss_proxy_mean'] = ranker_loss_proxy
			#pdb.set_trace()
		return out_dict
	
	def training_step(self, batch, batch_idx): # automatic training schedule
		
		out_dict = self.common_step(batch, batch_idx)
		mask = 1.0*torch.tensor(batch['mask']).to(self.device)

		use_soft=False

		if self.args.soft_labels or self.args.ultr_models:
			use_soft=True

		if self.args.train_ranker:
			acc_ranker = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), 
									mask=mask.detach().cpu())
			
			self.log("tr_loss_ranker", out_dict['ranker']['loss'], prog_bar=True, sync_dist=True)
			self.log("tr_acc_ranker", acc_ranker, prog_bar=True, sync_dist=True)
   
			if self.args.use_wandb and self.trainer.global_rank==0:
				wandb.log({"tr_loss_ranker":out_dict['ranker']['loss'], "tr_acc_ranker":acc_ranker})


			if not self.args.train_ranker_naive: # reward-arranger training schedule

				if self.args.urcc_loss or self.args.pgrank_loss:
					return out_dict['ranker']['loss'] # return urcc loss without the reward model's loss

				if not self.args.ultr_models:
					acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
					#self.val_acc['0.5_thres'] += acc_reward
					self.log("tr_acc_reward", acc_reward, prog_bar=True, sync_dist=True, batch_size=1)
				else:
					prob1 = torch.stack([1-torch.sigmoid(out_dict['reward']['logits'].squeeze()), torch.sigmoid(out_dict['reward']['logits'].squeeze())], dim=1) # batch x 2
					prob2 = torch.stack([1-out_dict['reward']['labels'], out_dict['reward']['labels']], dim=1)

					tv_reward = distance_prob(prob1, prob2, distance_type="tv")
					self.val_acc['tv'] += tv_reward
					self.log("tr_TV_reward", tv_reward, prog_bar=True, sync_dist=True, batch_size=1)

				rewards = torch.sigmoid(out_dict['reward']['logits'])

				# pdb.set_trace()

				if self.args.reward_correction:
					w = out_dict['reward']['correction_wt']
					rewards = rewards * w

				loss_reward = -self.args.reward_loss_reg * rewards.mean() # maximising the utility of reward model

				if self.args.reward_correction and self.args.reward_plus_proxy:
					# loss_proxy = (1-w) * out_dict['ranker']['loss_samples']
					loss_proxy = out_dict['ranker']['loss_proxy_mean']
					loss_reward += loss_proxy

				self.log("tr_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True)
				
				if self.args.use_wandb and self.trainer.global_rank==0:
					wandb.log({"tr_loss_reward":out_dict['reward']['loss']})

				# if self.args.use_soft_perm_loss:
				# 	loss_reward += self.args.soft_perm_loss_reg * out_dict['ranker']['soft_perm_loss']
				# 	self.log("tr_loss_soft_perm", out_dict['ranker']['soft_perm_loss'], prog_bar=True, sync_dist=True)
				
				# if self.args.cls_reg:
				# 	cls_loss = nn.MSELoss()
				# 	loss_cls = self.args.cls_reg_lr * cls_loss(out_dict['reward']['cls_token'], out_dict['ranker']['cls_token'])
				# 	loss_reward = loss_reward + loss_cls
				# 	self.log("tr_loss_cls", loss_cls, prog_bar=True, sync_dist=True, batch_size=1)
					
				return loss_reward
			else:
				return out_dict['ranker']['loss'] # naive ranker training loss 
		else:
			if self.args.factual:
				acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
				# self.val_acc['0.5_thres'] += acc_reward
				self.log("tr_acc_reward", acc_reward, prog_bar=True, sync_dist=True)
				
				if self.args.use_wandb and self.trainer.global_rank==0:
					wandb.log({"tr_acc_reward":acc_reward})
			else:
				prob1 = torch.stack([1-torch.sigmoid(out_dict['reward']['logits'].squeeze()), torch.sigmoid(out_dict['reward']['logits'].squeeze())], dim=1) # batch x 2
				prob2 = torch.stack([1-out_dict['reward']['labels'], out_dict['reward']['labels']], dim=1)

				tv_reward = distance_prob(prob1, prob2, distance_type="tv")
				self.log("tr_TV_reward", tv_reward, prog_bar=True, sync_dist=True, batch_size=1)
				
				if self.args.use_wandb and self.trainer.global_rank==0:
					wandb.log({"tr_TV_reward":tv_reward})

			loss_reward = torch.tensor(0.0, dtype=out_dict['reward']['loss'].dtype, device=out_dict['reward']['loss'].device)

			if self.args.reward_loss_cls:
				loss_reward += out_dict['reward']['loss'] # cls_loss
				self.log("tr_cls_loss_reward", loss_reward, prog_bar=True, sync_dist=True)

				if self.args.use_wandb and self.trainer.global_rank==0:
					wandb.log({"tr_cls_loss_reward":loss_reward})

			if self.config.per_item_feats:
				per_item_loss = self.args.reward_loss_reg_peritem*out_dict['reward']['loss_per_item'].squeeze()
				loss_reward += per_item_loss

				self.log("tr_peritem_loss_reward", per_item_loss, prog_bar=True, sync_dist=True)
	
				if self.args.use_wandb and self.trainer.global_rank==0:
					wandb.log({"tr_peritem_loss_reward":per_item_loss})
				
				if self.args.use_wandb and self.trainer.global_rank==0:
					wandb.log({"tr_loss_reward":per_item_loss})

			self.log("tr_loss_reward", loss_reward, prog_bar=True, sync_dist=True)
			return loss_reward
	
	def on_train_epoch_end(self):
		self.lr_scheduler.step()
		if self.current_epoch and self.current_epoch%self.args.save_epochs == 0:
			self.save(self.current_epoch)

	def validation_step(self, batch, batch_idx):
		
		out_dict = self.common_step(batch, batch_idx)
		mask = 1.0*torch.tensor(batch['mask']).to(self.device)

		# if not self.args.train_ranker:
		# 	cls_token = out_dict['reward']['cls_token']

		use_soft=False

		if self.args.soft_labels or self.args.ultr_models:
			use_soft=True
		#print (self.args.reward_sanity)
		#pdb.set_trace()
		if self.args.reward_sanity:
			self.save_output['reward_pred'].append(torch.sigmoid(out_dict['reward']['logits']).detach().cpu())
			self.save_output['pur_prob'].append(out_dict['reward']['labels'].detach().cpu())
			self.save_output['reward_prod'].append(torch.sigmoid(out_dict['reward']['logits_prod'].detach().cpu()))
			self.save_output['reward_correction_wt'].append(out_dict['reward']['correction_wt'].detach().cpu())
		#pdb.set_trace()
		if self.args.eval_ultr:
			if self.args.choice_ideal:
				acc_ranker, rel_ndcg = eval_ultr_ideal(batch=batch, device=self.device, args=self.args)
			else:
				acc_ranker, rel_ndcg = eval_ultr(batch=batch, pred_scores=out_dict['ranker']['logits'],
								device=self.device, args=self.args)
			self.val_acc['prob_atleast_1click'] += acc_ranker
			self.val_acc['rel_ndcg'] += rel_ndcg

			self.stats['prob_atleast_1click'].append(acc_ranker.item())
			self.stats['rel_ndcg'].append(rel_ndcg)

			self.log("val_acc_prob_click", acc_ranker, prog_bar=True, sync_dist=True, batch_size=1)
			self.log("val_acc_rel_ndcg", rel_ndcg, prog_bar=True, sync_dist=True, batch_size=1)
		# elif self.args.eval_llm:
		# 	new_positions = eval_llm(batch=batch, pred_scores=out_dict['ranker']['logits'],
		# 						device=self.device, args=self.args)
		# 	feat = torch.tensor(batch['query_document_embedding'], dtype=torch.float).to(self.device)
		# 	doc_mask = torch.tensor(batch['mask'], dtype=torch.float).to(self.device)
		# 	out = self.reward_model(inputs_embeds=feat, position_ids=new_positions, 
		# 				   			attention_mask=doc_mask)
		# 	score = torch.sigmoid(out['logits']).mean()
		# 	self.val_acc['E(pur_prob)'] += score
		# 	self.log('E(pur_prob)', score, prog_bar=True, sync_dist=True, batch_size=1)
		else:
			if self.args.train_ranker:

				acc_ranker = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), 
						mask=mask.detach().cpu()) # mask will be used for RAX. automatically fixed for torchmetrics
				
				if self.args.use_org_feats:
					#self.args.use_dcg = True
					dcg_10 = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), 
							mask=mask.detach().cpu(), k=10, use_dcg=True)
					dcg_5 = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), 
							mask=mask.detach().cpu(), k=5, use_dcg=True)
					dcg_3 = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), 
							mask=mask.detach().cpu(), k=3, use_dcg=True)

					self.log("val_acc_dcg3_ranker", dcg_3, prog_bar=True, sync_dist=True,  batch_size=1)
					self.log("val_acc_dcg5_ranker", dcg_5, prog_bar=True, sync_dist=True,  batch_size=1)
					self.log("val_acc_dcg10_ranker", dcg_10, prog_bar=True, sync_dist=True,  batch_size=1)
					
				self.val_acc['ndcg'] += acc_ranker
				self.val_loss['ranker'] += out_dict['ranker']['loss']
    
				self.log("val_loss_ranker", out_dict['ranker']['loss'], prog_bar=True, sync_dist=True,  batch_size=1)
				self.log("val_acc_ndcg10_ranker", acc_ranker, prog_bar=True, sync_dist=True,  batch_size=1)

				if self.args.use_wandb and self.trainer.global_rank==0:
					wandb.log({"val_loss_ranker":out_dict['ranker']['loss'], "val_acc_ranker":acc_ranker})

				if self.args.urcc_loss or self.args.pgrank_loss or self.args.use_org_feats:
					return

				#if not self.args.train_ranker_naive:
				if not self.args.ultr_models and not self.args.train_ranker_naive:
					acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
					self.val_acc['0.5_thres'] += acc_reward
					self.log("val_acc_reward", acc_reward, prog_bar=True, sync_dist=True, batch_size=1)
	 
					if self.args.use_wandb and self.trainer.global_rank==0:
						wandb.log({"val_acc_reward":acc_reward})
				else:
					if not self.args.train_ranker_naive:
						prob1 = torch.stack([1-torch.sigmoid(out_dict['reward']['logits'].squeeze()), torch.sigmoid(out_dict['reward']['logits'].squeeze())], dim=1) # batch x 2
						prob2 = torch.stack([1-out_dict['reward']['labels'], out_dict['reward']['labels']], dim=1)

						tv_reward = distance_prob(prob1, prob2, distance_type="tv")
						self.val_acc['tv'] += tv_reward
						self.log("val_TV_reward", tv_reward, prog_bar=True, sync_dist=True, batch_size=1)
					
					if self.args.ultr_models:
						acc_ranker, rel_ndcg = eval_ultr(batch=batch, pred_scores=out_dict['ranker']['logits'],
											device=self.device, args=self.args)

						self.val_acc['prob_atleast_1click'] += acc_ranker
						self.val_acc['rel_ndcg'] += rel_ndcg
						self.log("val_acc_prob_click", acc_ranker, prog_bar=True, sync_dist=True, batch_size=1)
						self.log("val_acc_rel_ndcg", rel_ndcg, prog_bar=True, sync_dist=True, batch_size=1)
	
					if not self.args.train_ranker_naive:
						self.log("val_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True, batch_size=1)
			else:
				#pdb.set_trace()

				self.log("val_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True, batch_size=1)
				self.val_loss['reward'] += out_dict['reward']['loss']

				if self.args.factual:
					acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
					self.val_acc['0.5_thres'] += acc_reward
					self.log("val_acc_reward", acc_reward, prog_bar=True, sync_dist=True, batch_size=1)
	 
					if self.args.use_wandb and self.trainer.global_rank==0:
						wandb.log({"val_acc_reward":acc_reward})
				else:
					prob1 = torch.stack([1-torch.sigmoid(out_dict['reward']['logits'].squeeze()), torch.sigmoid(out_dict['reward']['logits'].squeeze())], dim=1) # batch x 2
					prob2 = torch.stack([1-out_dict['reward']['labels'], out_dict['reward']['labels']], dim=1)

					tv_reward = distance_prob(prob1, prob2, distance_type="tv")
					self.val_acc['tv'] += tv_reward
					self.log("val_TV_reward", tv_reward, prog_bar=True, sync_dist=True, batch_size=1)
	 
					if self.args.use_wandb and self.trainer.global_rank==0:
						wandb.log({"val_TV_reward":tv_reward})

		return
	
	def on_validation_epoch_end(self):

		val_acc_avg = defaultdict(lambda: float)
		mean = defaultdict(lambda: float)
		se = defaultdict(lambda: float)

		if not self.args.use_org_feats and self.args.eval_ultr:
			for key in self.val_acc.keys():
				acc = self.val_acc[key]/self.trainer.num_val_batches[0]
				se[key] = pd.DataFrame(self.stats[key]).std(ddof=1)/np.sqrt(len(self.stats[key]))
				mean[key] = pd.DataFrame(self.stats[key]).mean()

				if self.trainer.global_rank==0:
					print('\n Val "',key ,'" after ', self.current_epoch, ' epochs : ',\
									acc, file=self.args.log_file)
					print(f"\n{key}: {mean[key][0]:.4f} ± {se[key][0]:.4f}")
					print(f"\n{key}: {mean[key][0]:.4f} ± {se[key][0]:.4f}", file=self.args.log_file)

		if self.args.reward_sanity:
			#pdb.set_trace()
			df = pd.DataFrame({
					'reward_pred': torch.stack(self.save_output['reward_pred']).view(-1).tolist(),
					'pur_prob': torch.stack(self.save_output['pur_prob']).view(-1).tolist(),
					'reward_prod': torch.stack(self.save_output['reward_prod']).view(-1).tolist(),
					'reward_correction': torch.stack(self.save_output['reward_correction_wt']).view(-1).tolist()
				})
			df.to_csv(f'{self.args.output_dir}/reward_data.csv', index=False)

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
						params.requires_grad = False #TODO: check torch.no_grad
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