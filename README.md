# Improving Translation Faithfulness of Large Language Models via Augmenting Instructions
## Environment

python 3.8.3

transformers 4.28.0.dev0

deepspeed==0.8.3

numpy==1.21

torch==2.0.1+cu117

accelerate==0.16.0

datasets==2.9.0

sentencepiece

sacrebleu

## Dataset
### training set
Parrot-hint: open-source at https://github.com/wxjiao/ParroT

OverMiss: file **train_data/overmiss_hf.json**
### test set
Flores: directory **test/Flores**

WMT22/WMT22-concat/WMT22-zero-shot : directory **test/WMT22**
## Train
- for LLaMA-7b: train_scripts/finetune_4gpu_llama.sh
- for BLOOMZ-3b: train_scripts/finetune_8gpu.sh
- for BLOOMZ-7b1-mt: train_scripts/finetune_4gpu.sh
## inference
path **infer_scripts/run_infer.sh**
