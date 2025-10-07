import os
import pdb
import torch
import wandb
import copy
import numpy as np
from collections import defaultdict
from pathlib import Path
import pandas as pd
from copy import deepcopy
import pytorch_lightning as pl
from src.bert import BertReward, BertArranger
from transformers.models.bert.configuration_bert import BertConfig
from src.utils import infer_ultr
from src.losses import PiRank_Loss, listMLE, listNet, lambdaLoss, loss_urcc
from src.loss_utils import hard_sort_group_parallel as hard_sort

from src.utils import soft_sort_group_parallel as soft_sort
from src.utils import eval_po, get_ndcg, binary_accuracy
from src.rl_baselines import compute_pg_rank_loss, compute_grpo_loss

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
		if self.args.po_eval:
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

		# if args.eval_ultr:
		# 	self.ips_model = None
		
		if args.grpo_loss:
			self.old_arranger = copy.deepcopy(self.arranger).eval()
			self.ref_arranger = copy.deepcopy(self.arranger).eval()
			self.burnout_period = True

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

		if self.args.lau_eval:
			purchase_prob = torch.tensor(batch['purchase_prob'], dtype=torch.float).to(self.device)
			pos_idx = torch.arange(batch['position'].shape[1]).repeat(batch['position'].shape[0],1).to(self.device)
		elif self.args.use_org_feats and self.trainer.validating:
			click = torch.tensor(batch['label']).to(self.device)
		else:
			click = torch.tensor(batch['click']).to(self.device)
			pos_idx = torch.tensor(batch['position']).to(self.device)

			if self.args.po_eval:
				examination = torch.tensor(batch['examination_ips']).to(self.device)
				relevance = torch.tensor(batch['relevance_ips']).to(self.device)

		out_dict = defaultdict(lambda: {})
		valid_f = torch.ones(feat.size(0), device=self.device)   

		if self.args.lau_eval:
			num_items = 8
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
			avg_click = purchase_prob
		elif self.args.po_eval and not self.args.eval_rels: # ultr model predictions
			prob_click = examination * torch.sigmoid(relevance)
			prob_noclick = torch.prod(1-prob_click, dim=1) # padding is automatically handled by mul with 1.0
			prob_atleast_1click = 1 - prob_noclick
			avg_click = prob_atleast_1click
			
			labels_per_item = prob_click
			labels = prob_click
			sampled_clicks = torch.bernoulli(prob_click)*doc_mask
		else: # hard GT labels
			click = click*doc_mask
			avg_click = torch.clamp(click.sum(dim=1), max=1.0).to(self.device)
			labels_per_item = click
			labels = click
			sampled_clicks = click

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


		if self.args.train_ranker: # Training schedule for rankers
			out_dict = self.arranger(inputs_embeds=feat, doc_feats=doc_feats, attention_mask=doc_mask)

			logits = out_dict['logits']

			if self.args.lau_eval:
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
			elif self.args.rank_loss == 'lambdarank':
				loss_fn = lambdaLoss

			if self.args.train_ranker_naive:
				if self.args.ips_sampling:
					loss = loss_fn(logits, sampled_clicks.float(), device=self.device, dummy_indices=mask_padded)
					out_dict['ranker'] = {'loss':loss.mean(),'logits':logits,'labels':labels}
				else:
					loss = loss_fn(logits, labels.float(), device=self.device, dummy_indices=mask_padded)
					out_dict['ranker'] = {'loss':loss.mean(),'logits':logits,'labels':labels}
				return out_dict
			
			if self.args.urcc_loss:
				loss = loss_urcc(logits, labels.float(), device=self.device, mask = mask_padded, reward_mod=self.reward_model,
					 			inputs_embeds=feat, position_ids=pos_idx, doc_feats=doc_feats, 
						   		labels_click = labels_per_item, attention_mask=doc_mask, avg_lables=avg_click)
			
			if self.args.pgrank_loss:
				loss = compute_pg_rank_loss(scores=logits, targets=labels, args=self.args, 
							mask=mask_padded, device=self.device, reward_mod=self.reward_model,
							inputs_embeds=feat, position_ids=pos_idx, doc_feats=doc_feats, 
							labels_click = labels_per_item, attention_mask=doc_mask)
				
			if self.args.grpo_loss:

				with torch.no_grad():
					old_scores = self.old_arranger(inputs_embeds=feat, doc_feats=doc_feats, 
									   attention_mask=doc_mask)['logits']
					ref_scores = self.ref_arranger(inputs_embeds=feat, doc_feats=doc_feats, 
									   attention_mask=doc_mask)['logits']
				
				loss = compute_grpo_loss(cur_scores=logits, old_scores=old_scores,ref_scores=ref_scores,
							 			targets=labels, args=self.args, mask=mask_padded, device=self.device, 
							 			reward_mod=self.reward_model, inputs_embeds=feat, labels_click = labels_per_item,
							   			attention_mask=doc_mask, burnout_period=self.burnout_period)
				
				
			if self.args.pgrank_loss or self.args.urcc_loss or self.args.grpo_loss: # don't need reward maximization
				out_dict['ranker'] = {'loss':loss.mean(),'logits':logits,'labels':labels}
				return out_dict
			elif self.args.use_org_feats and self.trainer.validating:
				out_dict['ranker'] = {'loss':torch.tensor([0.0], device=self.device),'logits':logits,'labels':labels}
				return out_dict
			else:	# reward maximization
				p_hat = soft_sort(s=logits, dummy_indices=mask_padded, 
					  temperature=self.args.soft_sort_temp) # rows = items; columns = positions; row[0] = i1*p(i[1]) + i2*p(i[2]) ....
				
				if self.args.ste:
					perm_mat_backward = p_hat
					perm_mat_forward = hard_sort(logits, dummy_indices=mask_padded)
					p_hat = perm_mat_backward + (perm_mat_forward - perm_mat_backward).detach()

				loss_fn = PiRank_Loss() # this is only used for logging ranker loss but not used in training
				ranker_loss = loss_fn(logits, labels.float(), device=self.device, dummy_indices=mask_padded)

				out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, 
							soft_position_ids=p_hat, labels=avg_click, doc_feats=doc_feats, attention_mask=doc_mask)
	
				out_dict['ranker'] = {'loss':ranker_loss.mean(), 'logits':logits,'labels':labels, 'reward_logits':out['logits']} # reward_logit = cls_token
		
		else: # Training schedule for reward model
			if self.args.ips_sampling: # checking for masking
				avg_click = torch.clamp(sampled_clicks.sum(dim=1), max=1.0).to(self.device) # P(C>=0) = 0/1
				out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click, doc_feats=doc_feats, 
						   labels_click = sampled_clicks, attention_mask=doc_mask)
			else:
				out = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click, doc_feats=doc_feats, 
							labels_click = labels_per_item, attention_mask=doc_mask)
			
		out_dict['reward'] = {'loss':out['loss'],'logits':out['logits'],'labels':avg_click, 'cls_token':out['cls_token'], 
						'logits_per_item': out['per_item_logits'], 'loss_per_item': out['per_item_loss']}

		if self.args.reward_correction: # reward correctin based on logged policy
			with torch.no_grad():
				out_prod = self.reward_model(inputs_embeds=feat, position_ids=pos_idx, labels=avg_click, doc_feats=doc_feats, 
							   labels_click = labels_per_item, attention_mask=doc_mask)
				
				# pred_scores_padded = torch.where(~doc_mask, -8e+8, logits)
    
				# sorted_indices = torch.argsort(pred_scores_padded, dim=1, descending=True)
				# new_positions = torch.empty_like(sorted_indices).to(self.device)
				# new_positions.scatter_(1, sorted_indices, torch.arange(logits.size(1)).unsqueeze(0).expand_as(logits).to(self.device))
				
				# out_ranker = self.reward_model(inputs_embeds=feat, position_ids=new_positions, labels=avg_click, doc_feats=doc_feats, 
				# 			   labels_click = labels_per_item, attention_mask=doc_mask)

				out_ranker = torch.sigmoid(out['logits']) # reward model out after looking over ranker's output
			
			rewards_prod = torch.sigmoid(out_prod['logits'])
			#residual = torch.abs(avg_click - rewards_prod)
			residual = torch.abs(out_ranker - rewards_prod)
			correction_wt = 1 - (self.args.residual_coef * residual)
			out_dict['reward']['correction_wt'] = correction_wt
			out_dict['reward']['logits_prod'] = out_prod['logits']

			# ranker_loss_proxy = (ranker_loss * valid_f * (1-correction_wt)).sum() / valid_f.sum().clamp_min(1.0)
			# out_dict['ranker']['loss_proxy_mean'] = ranker_loss_proxy
		return out_dict
	
	def training_step(self, batch, batch_idx): # automatic training schedule
		
		out_dict = self.common_step(batch, batch_idx)
		mask = 1.0*torch.tensor(batch['mask']).to(self.device)

		if self.args.train_ranker: # update schedule for ranker training
			acc_ranker = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), 
									mask=mask.detach().cpu()) # computing ndcg based on LTR loss, only for logging purpose
			
			self.log("tr_loss_ranker", out_dict['ranker']['loss'], prog_bar=True, sync_dist=True)
			self.log("tr_acc_ranker", acc_ranker, prog_bar=True, sync_dist=True)
   
			if self.args.use_wandb and self.trainer.global_rank==0:
				wandb.log({"tr_loss_ranker":out_dict['ranker']['loss'], "tr_acc_ranker":acc_ranker})

			if not self.args.train_ranker_naive: # reward-arranger training schedule
				if self.args.urcc_loss or self.args.pgrank_loss:
					return out_dict['ranker']['loss'] # return urcc/pgrank loss without the reward model's loss
				if self.args.grpo_loss:
					self.old_arranger = copy.deepcopy(self.arranger).eval()
					return out_dict['ranker']['loss'] # return grpo loss without the reward model's loss
				
				rewards = torch.sigmoid(out_dict['reward']['logits'])
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
				return loss_reward
			else:
				return out_dict['ranker']['loss'] # naive ranker training loss 
			
		else:  # update schedule for reward training
			acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze())
			self.log("tr_acc_reward", acc_reward, prog_bar=True, sync_dist=True)
			
			if self.args.use_wandb and self.trainer.global_rank==0:
				wandb.log({"tr_acc_reward":acc_reward})

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

		if self.args.grpo_loss:
			self.ref_arranger = copy.deepcopy(self.arranger).eval()
			self.burnout_period = False

		if self.current_epoch and self.current_epoch%self.args.save_epochs == 0:
			self.save(self.current_epoch)

	def validation_step(self, batch, batch_idx):
		
		out_dict = self.common_step(batch, batch_idx)
		mask = 1.0*torch.tensor(batch['mask']).to(self.device)

		if self.args.reward_sanity:
			self.save_output['reward_pred'].append(torch.sigmoid(out_dict['ranker']['reward_logits']).detach().cpu())
			self.save_output['pur_prob'].append(out_dict['reward']['labels'].detach().cpu())
			self.save_output['reward_prod'].append(torch.sigmoid(out_dict['reward']['logits_prod'].detach().cpu()))
			self.save_output['reward_correction_wt'].append(out_dict['reward']['correction_wt'].detach().cpu())

		if self.args.eval:
			acc_ranker, rel_ndcg = eval_po(batch=batch, pred_scores=out_dict['ranker']['logits'],
								device=self.device, args=self.args)
			self.val_acc['prob_atleast_1click'] += acc_ranker
			self.val_acc['rel_ndcg'] += rel_ndcg

			self.stats['prob_atleast_1click'].append(acc_ranker.item())
			self.stats['rel_ndcg'].append(rel_ndcg)

			self.log("val_acc_prob_click", acc_ranker, prog_bar=True, sync_dist=True, batch_size=1)
			self.log("val_acc_rel_ndcg", rel_ndcg, prog_bar=True, sync_dist=True, batch_size=1)
		else:
			if self.args.train_ranker:

				acc_ranker = get_ndcg(out_dict['ranker']['logits'].detach().cpu(), out_dict['ranker']['labels'].detach().cpu(), 
						mask=mask.detach().cpu()) # mask will be used for RAX. automatically fixed for torchmetrics
				
				if self.args.use_org_feats:
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

				if self.args.urcc_loss or self.args.pgrank_loss or self.args.use_org_feats or self.args.grpo_loss:
					return

				if not self.args.po_eval and not self.args.train_ranker_naive:
					acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze())
					self.val_acc['0.5_thres'] += acc_reward
					self.log("val_acc_reward", acc_reward, prog_bar=True, sync_dist=True, batch_size=1)
				else:
					
					if self.args.po_eval:
						acc_ranker, rel_ndcg = eval_po(batch=batch, pred_scores=out_dict['ranker']['logits'],
											device=self.device, args=self.args)

						self.val_acc['prob_atleast_1click'] += acc_ranker
						self.val_acc['rel_ndcg'] += rel_ndcg
						self.log("val_acc_prob_click", acc_ranker, prog_bar=True, sync_dist=True, batch_size=1)
						self.log("val_acc_rel_ndcg", rel_ndcg, prog_bar=True, sync_dist=True, batch_size=1)
	
					if not self.args.train_ranker_naive:
						self.log("val_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True, batch_size=1)
			else:
				self.log("val_loss_reward", out_dict['reward']['loss'], prog_bar=True, sync_dist=True, batch_size=1)
				self.val_loss['reward'] += out_dict['reward']['loss']

				acc_reward = binary_accuracy(out_dict['reward']['logits'].squeeze(), out_dict['reward']['labels'].squeeze(), soft=use_soft)
				self.val_acc['0.5_thres'] += acc_reward
				self.log("val_acc_reward", acc_reward, prog_bar=True, sync_dist=True, batch_size=1)
	
				if self.args.use_wandb and self.trainer.global_rank==0:
					wandb.log({"val_acc_reward":acc_reward})

		return
	
	def on_validation_epoch_end(self):

		val_acc_avg = defaultdict(lambda: float)
		mean = defaultdict(lambda: float)
		se = defaultdict(lambda: float)

		if not self.args.use_org_feats and self.args.po_eval:
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
					print("\nsetting reward grad as False\n")
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