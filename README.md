# Pytorch MiniLM

Unofficial Pytorch Reimplementation of MiniLM and MiniLM v2. (Incompleted)

- MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers (Neruips 2020)
- MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers (ACL 2021 Findings)

## Examples

1. Generate the corpus
```
python generate_corpus.py --cache_dir /input/dataset --corpus_dir /input/osilab-nlp/wikipedia
```

2. Generate the datasets
```
python generate_data.py \
        --train_corpus /input/osilab-nlp/wikipedia/corpus.txt \
        --bert_model ./models/bert-base-uncased \
        --output_dir ./data \
        --do_lower_case --reduce_memory
```

3. Pretrain
```
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    run_pretrain.py \
    --pregenerated_data ./data \
    --cache_dir ./cache \
    --epochs 4 \
    --gradient_accumulation_steps 1 \
    --train_batch_size 8 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --student_model ./models/bert-base-uncased \
    --masked_lm_prob 0.15 \
    --do_lower_case --fp16 --scratch
```

4. Finetune
```
python -m torch.distributed.launch --nproc_per_node=4 \
        run_finetune.py --model ./models/bert-base-uncased \
        --data_dir ./glue_data \
        --task_name RTE \
        --output_dir ./outputs \
        --do_lower_case --fp16 \
        --num_train_epochs 5 \
        --learning_rate 2e-05 \
        --eval_step 50 \
        --max_seq_length 128 \
        --train_batch_size 8
```

## Experiments (To Be Continued)

MiniLM (BERT with 4 Layers, 312 Dims) 

|                                | Accuracy (%)|
| -------------------------------| ----------- |
| RTE                            | 65.70%      | 
| SST-2                          | 86.85%      |

## Issues
- (22.01.01) <s>Unknown error occurs in finetuning code with multi-gpu setting in RTX 3090 (CUDA VER 11.4)</s> (Solved).
- (22.01.04) Complete the pretrain code on tiny size dataset (<s>Wikipedia datasets with 100 documents</s> also done with 6M documents).
- (22.01.05) <s>Learning Rate presents as zero if using knowledge distillation.</s> (Solved)
- (22.01.07) <s>Unknown error occurs in pretraining code with more than 3 GPUs. Our code works well on 2 GPUs server.</s> (Solved)
- (22.01.11) <s>If we do not use --reduce_memory option, the code do not make any errors on multiple GPU</s> (with gpu numbers > 3, Solved). 

## TODO
- [X] Generate wikipedia corpus and generate dataset
- [X] Pretraining on multi-gpu setting
- [X] Finetuning on multi-gpu setting

## References
- https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT
- https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/AutoTinyBERT