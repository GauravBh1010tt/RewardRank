# CF_ranking
Counter-factual reward ranking

| Architecture |        $$\mathbb{E}[P(C\ge 1)]$$        |      $$NDCG@10_{rel}$$       |
| :------------------: | :------------------: | :------------------: |
|         `Ranker Naive`         |         `1`          |         `8`          |
| * | * | * |
|         `RR_hard`         |         `43.5`          |         `8`          |
|         `RR_soft`         |         `44.2`          |         `8`          |
|         `RR_hard_pre`        |         `43.6`          |         `8`          |
|         `RR_soft_pre`        |         `43.4`          |         `8`          |
|         `RR_hard_pre_cls`         |         `43.8`          |         `8`          |
|         `RR_soft_pre_cls`         |         `44.3`          |         `8`          |
| * | * | * |


## Nov 13
- Debug attention masking
    - Performance fluctuation when using masking during inference

## Nov 12
- reproduce results on UBC server
    - fix ranker_org issue
- create ULTR dataset
- run ULTR experiments
    - reward training
    - ranker training

## Nov 7

#### Notes:
- detach gradients from disc
    - multiple gradients could cancel out each other
- policy regularization: different ways
- contrastive regularization: counterfactual + score 
    - can use contrastive generation of counterfactuals using the oracle
- sub-distillation
    - distillation from labels/logits - ?

#### Updates:
- 1

## Nov 5
- ips_org click prob for QG: org positions
- eval_ultr: use examination prob from propensities file: don't use ips for inference
- relevance ndcg@ips
- reverse examination ablation
- outperform ULTR positions information
- Relevance of arranger insteaed of IPS
- Data+Outputs 
- Upload data
- checkpoints -> S3 + local_drive

## Nov 1
- padding for loss and evaluation metric (NDCG +) : reverify
- validation loss value
- cls_token regularization fore reward model - ?

## Oct 31

- padding for position matrix
- padding for set-aware attention
- padding for loss and evaluation metric
- soft-permutation matrix interpretation
- soft-permutation over item arrangements
- hard-coded discriminators
- Debug - use ultr-model as disc or a simpler dummy model