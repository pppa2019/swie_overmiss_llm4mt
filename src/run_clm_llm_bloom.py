#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset
from typing import List
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from my_trainer import My_Trainer
from transformers import Trainer
from trainable_multiL_ins_bloom import InsEmbBloomForCausalLM
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
# xxx: 2023-03-21
import copy


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# xxx: 2023-03-21
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    only_optimize_layers: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Only optimize the parameters of some layers."})
    emb_mix_alpha: Optional[float] = field(default=0)
    freeze_embed: Optional[bool] = field(default=True)
    ins_pool_method: Optional[str] = field(default='mean')
    cat_post_ins: Optional[str] = field(default="False")
    ins_fuse_method: Optional[str] = field(default='add')
    ins_hidden_size: Optional[int] = field(default=256)
    add_ins_transform: Optional[bool] = field(default=False)
    ins_transform_dropout: Optional[float] = field(default=0)
    ins_transform_activate: Optional[str] = field(default='Tanh')
    warm_up_alpha: Optional[bool] = field(default=False)
    alpha_warmup_step: Optional[int] = field(default=100)
    origin_setting: Optional[bool] = field(default=False)
    qkv_only: Optional[bool] = field(default=False)
    selected_layer: Optional[List[int]] = field(default=None)
    segment_size: Optional[int] = field(default=None)
    emb_text: Optional[str] = field(default=None)
    custom_alpha: Optional[float] = field(default=False)
    weighted_seg_id: Optional[int] = field(default=None)
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            if len(data_args.train_file.split(',')) > 0:
                data_files["train"] = data_args.train_file.split(',')
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # import ipdb;ipdb.set_trace()
    seg_id = tokenizer.encode('###')[0]

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        config.origin_settint = model_args.origin_setting
        if not model_args.origin_setting:
            config.alpha = model_args.emb_mix_alpha
            config.alpha_rate = 1
            config.cat_post_ins = model_args.cat_post_ins
            config.ins_pool_method = model_args.ins_pool_method
            config.ins_fuse_method = model_args.ins_fuse_method
            config.ins_hidden_size = model_args.ins_hidden_size
            config.add_ins_transform = model_args.add_ins_transform
            config.ins_transform_dropout = model_args.ins_transform_dropout
            config.ins_transform_activate = model_args.ins_transform_activate
            config.selected_layer = model_args.selected_layer
            config.seg_id = seg_id
            if model_args.segment_size is not None:
                config.segment_size = model_args.segment_size
            if model_args.emb_text is not None:
                config.emb_text = model_args.emb_text
            if model_args.custom_alpha:
                config.custom_alpha = model_args.custom_alpha
            if model_args.weighted_seg_id:
                config.weighted_seg_id = model_args.weighted_seg_id
        # config.log_dir = os.path.join(training_args.output_dir, "run_cosine")
        # AutoModelForCausalLM
        if model_args.origin_setting:
            model_class = AutoModelForCausalLM
        else:
            model_class = InsEmbBloomForCausalLM
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            # ignore_mismatched_sizes=True
        )
        if not model_args.origin_setting:
            for ins_fuse in model.transformer.ins_fuses:
                if ins_fuse is not None:
                    torch.nn.init.constant_(ins_fuse.weight, 0.5)
    else:
        # AutoModelForCausalLM
        model = model_class.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # xxx: 2023-06-07, freeze expect embedding
    # for name, param in model.named_parameters():
    #     if 'embed' not in name:
    #         param.requires_grad = False
    #     print(name, param.requires_grad)
    # xxx: 2023-03-21, add padding
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    tokenizer.padding_side = "right"
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # xxx: 2023-03-21, add special tokens
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # def tokenize_function(examples):
    #     with CaptureLogger(tok_logger) as cl:
    #         output = tokenizer(examples[text_column_name])
    #     # clm input could be much much longer than block_size
    #     if "Token indices sequence length is longer than the" in cl.out:
    #         tok_logger.warning(
    #             "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
    #             " before being passed to the model."
    #         )
    #     return output
    #
    # with training_args.main_process_first(desc="dataset map tokenization"):
    #     if not data_args.streaming:
    #         tokenized_datasets = raw_datasets.map(
    #             tokenize_function,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="Running tokenizer on dataset",
    #         )
    #     else:
    #         tokenized_datasets = raw_datasets.map(
    #             tokenize_function,
    #             batched=True,
    #             remove_columns=column_names,
    #         )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= block_size:
    #         total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    # xxx: 2023-03-14
    # In Opt, "<pad>": 1
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    # xxx: 2023-05-17, leave padding to DataCollatorForSeq2Seq for batch padding and avoid unnecessary paddings


    def preprocess_function(examples):
        with CaptureLogger(tok_logger) as cl:
            # xxx: 2023-04-07; text: target, prefix: source
            padding = "max_length"  # or False
            text = examples[text_column_name]  # may have multiple strings
            if "prefix" in column_names:
                prefix = examples["prefix"]  # may have multiple strings
                instruction = []
                for p in prefix:
                    if model_args.emb_text == "simple":
                        if "\n\n### Input:" in p:
                            # 2023-07-04;@xxx simple instruction
                            instruction.append(p.split("### Input:")[0].split("### Instruction:")[-1].strip())
                        else:
                            instruction.append(p.split("### Response:")[0].split("### Instruction:")[-1].strip())
                    elif model_args.emb_text == "input_only":
                        if "### Input:" in p:
                            instruction.append(p.split("### Input:")[1].split('###')[0])
                        else:
                            instruction.append("None")
                    elif model_args.emb_text == "till_input":
                        instruction.append(p.split("### Response:")[0])
                    elif model_args.emb_text == "only_prefix":
                        instruction.append(p.split("###")[0])
                    else:
                        if "### Input:" in p:
                            instruction.append(p.split("### Input:")[0])
                        else:
                            instruction.append(p.split("### Response:")[0])
                    # input embedding
                    # instruction.append(p.split("\n\n### Input:")[1].split('\n\n###')[0])
                text = [s + t for s, t in zip(prefix, text)]
                
                prefix_tokenized = tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False)
                if model_args.ins_pool_method=='mlp':
                    ins_tokenized = tokenizer(instruction, truncation=True, max_length=256, padding="max_length")
                else:
                    ins_tokenized = tokenizer(instruction, truncation=True, max_length=block_size, padding=False)
                if not model_args.origin_setting:
                    text_tokenized["instruction"] = ins_tokenized["input_ids"]
                    text_tokenized["ins_attention_mask"] = ins_tokenized["attention_mask"]
                labels = copy.deepcopy(text_tokenized["input_ids"])
                prefix_lengths = [len(p) for p in prefix_tokenized["input_ids"]]
                for label, prefix_len in zip(labels, prefix_lengths):  # Do not compute loss for prompt inputs
                    label[:prefix_len] = [IGNORE_INDEX] * prefix_len  # [IGNORE_INDEX for i in range(prefix_len)]
            else:
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False)
                labels = copy.deepcopy(text_tokenized["input_ids"])
            text_tokenized["labels"] = labels
            
            
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return text_tokenized

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # with training_args.main_process_first(desc="grouping texts together"):
    #     if not data_args.streaming:
    #         lm_datasets = tokenized_datasets.map(
    #             group_texts,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc=f"Grouping texts in chunks of {block_size}",
    #         )
    #     else:
    #         lm_datasets = tokenized_datasets.map(
    #             group_texts,
    #             batched=True,
    #         )

    # xxx: 2023-03-17
    with training_args.main_process_first(desc="example per line with padding"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Tokenize with padding",
            )
        else:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
            )

    # sum([len(sent)>512 for sent in lm_datasets['train']['input_ids']])
    # mean([len(sent) for sent in lm_datasets['train']['instruction']])
    if training_args.do_train:
        #if "train" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # xxx: print samples
        logger.info("xxx: Showcase the tokenized training samples.")
        for i in range(3):
            print(next(iter(train_dataset)))
    # import  ipdb;ipdb.set_trace()
    if training_args.do_eval:
        #if "validation" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    training_args.only_optimize_layers = model_args.only_optimize_layers
    print("debug: training_args.only_optimize_layers", training_args.only_optimize_layers)
    if model_args.only_optimize_layers:
        for name, param in model.named_parameters():
            if "embed" in name and model_args.freeze_embed and 'segment' not in name:
                param.requires_grad = False
            if "transformer." in name:
                param.requires_grad = False
                if model_args.qkv_only:
                    if "mlp.dense" in name or 'layernorm' in name or 'self_attention.dense.' in name:
                        continue
                for i in training_args.only_optimize_layers:
                    if f".{i}." in name:
                        param.requires_grad = True
                        print("trainable parameters", name)
                        break
            else:
                param.requires_grad = True
                print("trainable parameters", name)




    # Initialize our Trainer
    if model_args.origin_setting:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            #data_collator=default_data_collator,
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                            padding=True, label_pad_token_id=IGNORE_INDEX),
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        )
    else:
        trainer = My_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            #data_collator=default_data_collator,
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                            padding=True, label_pad_token_id=IGNORE_INDEX),
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            warm_up_alpha=model_args.warm_up_alpha,
            alpha_warmup_step=model_args.alpha_warmup_step
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()