import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
import wandb
import functools

from pathlib import Path
from tqdm import tqdm
from time import time
import math
import os

from src.data import ELI5DatasetS2S, load_data, format_docs

def prepare_dataset(logger, args, data_type="train"):
    """
    Load dataset from disk

    :param
        logger: logging
                log information
        args: Arguments
                contains hyper-parameters
        data_type: str
                specify training or evaluation dataset

    :return
        dataset: src.data.Dataset
                custome Dataset object
    """
    if args.logger:
        logger.info("Creating Dataset...")

    examples = load_data(args.train_data if data_type == "train" else args.eval_data)
    example_docs = format_docs(examples)
    dataset = ELI5DatasetS2S(examples, document_cache=example_docs)

    if args.logger:
        logger.info("Creating Dataset is done.")

    return dataset

def prepare_training_stuff(logger, args):
    if args.logger:
        logger.info("Creating model and tokenizer...")

    full_model_name = args.model_name + '-' + args.model_size 
    
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name).to(args.device)
    if args.model_path is not None:
        param_dict = torch.load(args.model_path)  # has model weights, optimizer, and scheduler states
        model.load_state_dict(param_dict["model"])

    if args.logger:
        logger.info("Creating model and tokenizer is done")
    return tokenizer, model

class Trainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, args, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        self.logger = logger
        self.optimizer = None
        self.scheduler = None
        self.model_save_name = self.args.model_name +  '-' + self.args.model_size

        # if self.args.is_main:
        #     try:
        #         wandb.login()
        #         self.wandb_logger = True
        #         wandb.init(
        #             project='answer-generation-with-T5',
        #             name='Answer Generation With T5')
        #     except:
        #         self.wandb_logger = False
        #         if self.args.logger:
        #             self.logger.warning("Wandb is not available.")
        # else:
        #     self.wandb_logger = False

        torch.manual_seed(self.args.seed)

    def make_qa_s2s_batch(self, qa_list, tokenizer, max_len=64, max_a_len=360, device="cuda:0"):
        q_ls = [q for q, a in qa_list]
        a_ls = [a for q, a in qa_list]
        q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, pad_to_max_length=True)
        q_ids, q_mask = (
            torch.LongTensor(q_toks["input_ids"]).to(device),
            torch.LongTensor(q_toks["attention_mask"]).to(device),
        )
        a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(max_len, max_a_len), pad_to_max_length=True)
        a_ids, a_mask = (
            torch.LongTensor(a_toks["input_ids"]).to(device),
            torch.LongTensor(a_toks["attention_mask"]).to(device),
        )
        labels = a_ids[:, 1:].contiguous().clone()
        labels[a_mask[:, 1:].contiguous() == 0] = -100
        model_inputs = {
            "input_ids": q_ids,
            "attention_mask": q_mask,
            "decoder_input_ids": a_ids[:, :-1].contiguous(),
            "labels": labels}   
        return model_inputs
    def train_qa_s2s_epoch(self, model, dataset, tokenizer, optimizer, scheduler, args, e=0, curriculum=False):
      model.train()
      # make iterator 

      if curriculum:
          train_sampler = SequentialSampler(dataset)
      else:
          train_sampler = RandomSampler(dataset)

      model_collate_fn = functools.partial(
          self.make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device=args.device
      )
      data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
      epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
      # accumulate loss since last print
      loc_steps = 0
      loc_loss = 0.0
      st_time = time()
      for step, batch_inputs in enumerate(epoch_iterator):
          outputs = model(**batch_inputs)
          loss = outputs.loss
          loss.backward()

          # optimizer
          if step % args.backward_freq == 0:
              optimizer.step()
              scheduler.step()
              model.zero_grad()

          # some printing within the epoch
          if args.logger:
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0 or step == 1:
                self.logger.info(
                    "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        e, step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                    )
                )
                print(
                    "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        e, step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                    )
                )
                loc_loss = 0
                loc_steps = 0

    def eval_qa_s2s_epoch(self):
        self.model.eval()
        # make iterator
        train_sampler = SequentialSampler(self.eval_dataset)
        model_collate_fn = functools.partial(
            self.make_qa_s2s_batch, tokenizer=self.tokenizer, max_len=self.args.max_length, device=self.args.device
        )
        data_loader = DataLoader(self.eval_dataset, batch_size=self.args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
        epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
        # accumulate loss since last print
        loc_steps = 0
        loc_loss = 0.0
        st_time = time()
        with torch.no_grad():
            for step, batch_inputs in enumerate(epoch_iterator):
                outputs = self.model(**batch_inputs)
                loss = outputs.loss
                # log within the epoch
                if self.args.logger:
                    loc_loss += loss.item()
                    loc_steps += 1
                    if step % self.args.print_freq == 0:
                        self.logger.info("{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(step, len(self.eval_dataset) // self.args.batch_size, loc_loss / loc_steps, time() - st_time))
        if self.args.logger:
            self.logger.info("Total \t L: {:.3f} \t -- {:.3f}".format(loc_loss / loc_steps, time() - st_time,))

    def train(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=400,
            num_training_steps=(self.args.num_epochs + 1) * math.ceil(len(self.train_dataset) / self.args.batch_size),
        )

        

        for e in range(self.args.num_epochs):
            self.train_qa_s2s_epoch(
                self.model,
                self.train_dataset,
                self.tokenizer,
                self.optimizer,
                self.scheduler,
                self.args,
                e,
                curriculum=(e == 0))

            m_save_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }

            if self.args.logger:
                self.logger.info("Saving model {}_{}".format(self.model_save_name, e))
            self.eval_qa_s2s_epoch(self)
            torch.save(m_save_dict, os.path.join(self.args.checkpoint_path, "{}_{}.pth".format(self.model_save_name, e)))