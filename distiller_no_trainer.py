# %%
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from dataclasses import dataclass

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers.testing_utils import CaptureLogger
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

# %%
import torch.nn as nn
from torch.nn import functional as F
def kl_torch(s_logits, t_logits, temperature):
    ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
    loss_ce = (
        ce_loss_fct(
            F.log_softmax(s_logits / temperature, dim=-1),
            F.softmax(t_logits / temperature, dim=-1),
        )
        * (temperature) ** 2
    )
    return loss_ce

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# Make one log on every process with the configuration for debugging.
logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# %%
def get_model(MODEL_PATH, torch_dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        # device_map="auto",
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
    
def is_identical(t1, t2):
    return (
        type(t1) == type(t2) and
        t1.vocab_size == t2.vocab_size and
        t1.model_max_length == t2.model_max_length and
        t1.padding_side == t2.padding_side and
        t1.truncation_side == t2.truncation_side and
        t1.bos_token == t2.bos_token and
        t1.eos_token == t2.eos_token and
        t1.pad_token == t2.pad_token
    )

# %%
@dataclass
class Args:
    weight_decay: float = 0.0
    learning_rate: float = 2e-5
    preprocessing_num_workers: int = 32
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 16
    teacher_name: str = "Qwen/Qwen2-1.5B"
    student_name: str = "nguyenthanhdo/Qwen2-1.5B-tierce"
    block_size: int = 512
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 16
    max_train_steps: int = None
    num_train_epochs: int = 3
    num_warmup_steps: int = 0
    checkpointing_steps: str = "epoch"
    data_dir: str = "../data/c4"
    train_file: str = "sample.jsonl"
    data_cache: str = "../data/c4_cache"
    eval_split: float = 0.02
    seed: int = 69
    
args = Args()
set_seed(args.seed)

# %%
## INITIATE Acclerator
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
accelerator.wait_for_everyone()

# %%
TEACHER_PATH = args.teacher_name
STUDENT_PATH = args.student_name

teacher, t_tokenizer = get_model(TEACHER_PATH)
student, s_tokenizer = get_model(STUDENT_PATH)
if is_identical(s_tokenizer, t_tokenizer):
    tokenizer = t_tokenizer

if args.block_size > tokenizer.model_max_length:
    logger.warning(
        f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
        f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    )
block_size = min(args.block_size, tokenizer.model_max_length)
print(f"Max token length: {block_size}")

# %%
def tokenize_function(examples):
    output = tokenizer(examples["text"])
    return output

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    """
    We drop the small remainder, and if the total_length < block_size we exclude 
    this batch and return an empty dict. We could add padding if the model 
    supported it instead of this drop, you can customize this part to your needs.
    """
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


with accelerator.main_process_first():
    lm_datasets = []
    path = Path(args.data_dir)
    files = [file.name for file in path.glob("*.jsonl")]
    if args.train_file:
        assert (path / args.train_file) in path.glob("*"), (
            f"File {args.train_file} does not exist in the data directory {args.data_dir}"
        )
        files = [args.train_file]
    logger.info(f"Loading training data from {files}")
    for idx, file in enumerate(tqdm(files)):
        data_file = os.path.join(path, file)
        filename = ''.join(file.split(".")[:-1])
        cache_path = os.path.join(
            args.data_cache, 
            filename+f"_{block_size}"
        )
        os.makedirs(cache_path, exist_ok=True)
        try:
            lm_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
            logger.info(f'training dataset {file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset(
                "json", 
                data_files=[data_file], 
                # cache_dir=cache_dir,
                keep_in_memory=False, 
                num_proc=args.preprocessing_num_workers,
                split="train"
            )
            logger.info(f">>> {raw_dataset}")
            
            column_names = raw_dataset.column_names
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers, 
                remove_columns=column_names,
                load_from_cache_file=True, 
                desc="Running tokenizer on dataset",
            )
            
            lm_dataset = tokenized_dataset.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers, 
                load_from_cache_file=True, 
                desc=f"Grouping texts in chunks of {block_size}",
            )
            lm_dataset.save_to_disk(cache_path)
        ## end of iter
        assert type(lm_dataset) == datasets.Dataset, (
            f"Loaded lm_dataset is of type {type(lm_dataset)}, expected type is {datasets.Dataset}"
        )
        lm_datasets.append(lm_dataset)
    ## out of for loop
    assert len(lm_datasets) > 0, "No datasets are loaded"
    lm_datasets = (
        concatenate_datasets(lm_datasets) if len(lm_datasets) > 1
        else lm_datasets[0]
    )
    # %%
    splits = lm_datasets.train_test_split(test_size=args.eval_split)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        text_example = tokenizer.decode(train_dataset[index]["input_ids"])
        logger.info(f">>> Sample {index} of the training set: {text_example}.")


# %%
class TeacherStudent(nn.Module):
    def __init__(
        self,
        teacher: nn.Module = None,
        student: nn.Module = None,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        
    def forward(self):
        pass
        
class Distiller(nn.Module):
    def __init__(
        self,
        teacher: nn.Module = None,
        student: nn.Module = None,
        tokenizer: AutoTokenizer = None,
        train_dataset: datasets.Dataset = None,
        eval_dataset: datasets.Dataset = None,
        temperature: float = 2.0,
        alpha_ce: float = 0.7,
        alpha_clm: float = 0.3,
        args: Args = None
    ):
        super().__init__()
        logger.info("Initializing Distiller")
        # self.teacher = teacher
        # self.student = student
        self.pair = TeacherStudent(
            teacher=teacher,
            student=student
        )
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        # DATALOADER:
        logger.info("Initializing Dataloader")
        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, 
            collate_fn=default_data_collator, 
            batch_size=self.args.per_device_train_batch_size
        )
        self.eval_dataloader = DataLoader(
            eval_dataset, 
            collate_fn=default_data_collator, 
            batch_size=self.args.per_device_eval_batch_size
        )
        
        # OPTIMIZER
        logger.info("Initializing Optimizer")
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.pair.student.named_parameters() 
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.pair.student.named_parameters() 
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=args.learning_rate
        )
        
        # SCHEDULER
        logger.info("Initializing Scheduler")
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        
        self.scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=(
                args.max_train_steps if overrode_max_train_steps
                else args.max_train_steps * accelerator.num_processes
            ),
        )

        # HEHE
        (self.pair,
         # self.student, 
         # self.teacher, 
         self.optimizer, 
         self.train_dataloader, 
         self.eval_dataloader, 
         self.scheduler) = accelerator.prepare(
            # self.student, 
            # self.teacher, 
            self.pair,
            self.optimizer, 
            self.train_dataloader, 
            self.eval_dataloader, 
            self.scheduler
        )
        # self.teacher = accelerator.prepare(self.teacher)
        # self.teacher_device = "cuda:0"
        # self.teacher = self.teacher.to(self.teacher_device)
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")

        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_clm = alpha_clm

    def forward(self, batch):
        student_outputs = self.pair.student(**batch, output_hidden_states=True)
        with torch.no_grad():
            # teacher_outputs = self.teacher(
            #     **{k: v.to(self.teacher_device) for k, v in batch.items()},
            #     output_hidden_states=True
            # )
            # torch.cuda.empty_cache()
            teacher_outputs = self.pair.teacher(**batch, output_hidden_states=True)
            # torch.cuda.empty_cache()
        return student_outputs, teacher_outputs

    def compute_loss(
        self, 
        student_outputs, 
        teacher_outputs,
    ):
        ## format
        s_logits, s_hidden_states = (
            student_outputs["logits"], 
            student_outputs["hidden_states"]
        )
        t_logits, t_hidden_states = (
            teacher_outputs["logits"], 
            teacher_outputs["hidden_states"]
        )
        assert s_logits.size() == t_logits.size()
        s_logits = s_logits.view(-1, s_logits.size(-1))
        t_logits = t_logits.view(-1, t_logits.size(-1))

        total_loss, loss_clm, loss_ce = 0, 0, 0
        ## clm loss
        if self.alpha_clm > 0:
            loss_clm = student_outputs.loss
            total_loss += loss_clm
        
        ## logits loss.
        if self.alpha_ce > 0:
            loss_ce = kl_torch(s_logits, t_logits, temperature=self.temperature)
            total_loss += loss_ce
            
        ## total loss
        # total_loss = self.alpha_ce * loss_ce + self.alpha_clm * loss_clm
        return dict(
            loss_clm=loss_clm,
            loss_ce=loss_ce,
            total_loss=total_loss
        )
        
    def optimize(self, loss):
        accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def eval(self):
        args = self.args
        self.student.eval()
        losses = []
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.student(**batch)
    
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(
                loss.repeat(args.per_device_eval_batch_size)
            ))
    
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
            
        return eval_loss, perplexity
        
    def begin_training_log(self):
        args = self.args
        total_batch_size = (
            args.per_device_train_batch_size * 
            accelerator.num_processes * 
            args.gradient_accumulation_steps
        )
        
        logger.info(f"***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        
    def train(self):
        self.begin_training_log()
        args = self.args
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        progress_bar.update(completed_steps)
        """
        Let's go get some drinks!
        """
        for epoch in range(starting_epoch, args.num_train_epochs):
            self.pair.student.train()
            self.pair.teacher.eval()
            active_dataloader = self.train_dataloader
            for step, batch in enumerate(active_dataloader):
                # Main training loop
                with accelerator.accumulate(self.pair.student):
                    student_outputs, teacher_outputs = self.forward(batch)
                    # logger.info(f"student_outputs: {student_outputs.logits.requires_grad}")
                    # logger.info(f"teacher_outputs: {teacher_outputs.logits.requires_grad}")
                    losses = self.compute_loss(student_outputs, teacher_outputs)
                    self.optimize(losses["total_loss"])

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    # logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                    loss_message = []
                    for k, v in losses.items():
                        loss = "{:.2f}".format(v)
                        loss_message += [f" {k}: {loss} "]
                    logger.info(f"Train loss: {'||'.join(loss_message)}")
                    progress_bar.update(1)
                    completed_steps += 1

                # # Step checking.
                # if isinstance(checkpointing_steps, int):
                #     if completed_steps % checkpointing_steps == 0:
                #         output_dir = f"step_{completed_steps}"
                #         if args.output_dir is not None:
                #             output_dir = os.path.join(args.output_dir, output_dir)
                #         accelerator.save_state(output_dir)
                # if completed_steps >= args.max_train_steps:
                #     break
        
            # EVAL
            # eval_loss, perplexity = self.eval()
            # logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
        
            # SAVE CHECKPOINTS
            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

if __name__ == "__main__":
    # %%
    distiller = Distiller(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args
    )
    
    # %%
    distiller.train()
