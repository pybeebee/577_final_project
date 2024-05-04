### INSTRUCTION TUNE LONGT5-BASE USING THE CURATED INSTRUCTION DATA
import wandb
import argparse
import numpy as np
import os
import torch
from datetime import datetime
import transformers
from datasets import load_dataset
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, LongT5Config, LongT5ForConditionalGeneration
from evaluate import load
import nltk
import json
import re, string
from unidecode import unidecode
from collections import Counter
import pandas as pd
from datasets import Dataset, DatasetDict
import zeroscrolls_metrics as z
# nltk.download('punkt')

def main(args):

    ### Setup wandb logging
    if int(os.environ["LOCAL_RANK"])==0: # ONLY TRACK IF IT'S THE MAIN PROCESS
        wandb.login()
        run = wandb.init(
            project=args.wandb_project_name,
            group=args.wandb_group,
            )   
        
    ### Set up output dirs for model and checkpoints
    if "data_splits_regen" in args.json_dir:
        data_type = args.json_dir[args.json_dir.index("data_splits_regen")+18:]
    elif "data_splits_mistral_abstractify" in args.json_dir:
        data_type = args.json_dir[args.json_dir.index("data_splits_mistral_abstractify")+39:]
    elif "data_splits_mistral" in args.json_dir:
        data_type = args.json_dir[args.json_dir.index("data_splits_mistral")+20:]
    
    output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_").replace("-", "_"), f"wandb_group_{args.wandb_group}") #, f"wandb_{run.name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # write args to file for easy reference later on
    import sys
    try:
        from shlex import quote as cmd_quote
    except ImportError:
        from pipes import quote as cmd_quote
    cmdline = " ".join(map(cmd_quote, sys.argv[1:]))
    args_file_path = os.path.join(output_dir, "args.txt")
    f = open(args_file_path, "a")
    try:
        f.write(f"\n\n[[[[[[[[[[[[ WANDB RUN NAME: {run.name} ]]]]]]]]]]]]")
        f.write(f"\n\n[[[[[[[[[[[[ WANDB RUN LINK: {run.url} ]]]]]]]]]]]]\n\n")
    except:
        pass
    f.write(cmdline)
    f.close()

    # checkpoint dir 
    checkpoint_dir = os.path.join(output_dir,"checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # final model dir:
    model_save_dir = os.path.join(output_dir,"final_model")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    def preprocess_function(examples):
        # append /s token to documents if desired
        inputs = [doc+args.suffix for doc in examples['instruction']]
        outputs = [ans for ans in examples['answer']]

        # tokenize document instances
        model_inputs = tokenizer(inputs, 
                                max_length=4096, 
                                padding="max_length",  # CHECK: NEED THIS?
                                truncation=True)#, return_tensors="pt")

        labels = tokenizer(
            text_target=outputs, 
            max_length=512, 
            padding="max_length",
            truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    
    # DEPRECATED F1/EM calculator 
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        em_metric = load("exact_match") #https://huggingface.co/spaces/evaluate-metric/exact_match
        f1_metric = load("f1") # https://huggingface.co/spaces/evaluate-metric/f1

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        em_result = em_metric.compute(predictions=decoded_preds,
                                   references=decoded_labels, 
                                   ignore_case=True,
                                   ignore_punctuation=True,
                                   ignore_numbers=False,
        )
        predictions = list(predictions.flatten())
        labels = list(labels.flatten())
        if len(predictions) < len(labels):
            predictions += [0] * int(len(labels)-len(predictions))
        if len(predictions) > len(labels):
            labels += [0]*(len(predictions)-len(labels))
        f1_result = f1_metric.compute(predictions=predictions, # requires numerical input!!
                                      references=labels,
                                      average="micro",
                                      )
        results = {"exact_match": round(em_result["exact_match"], 4),
                   "f1": round(f1_result["f1"], 4)
                   }
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        results["mean_gen_len"] = np.mean(prediction_lens)
        results["min_gen_len"] = min(prediction_lens)
        results["max_gen_len"] = max(prediction_lens)
        
        # if True:
        if int(os.environ["LOCAL_RANK"])==0:
            wandb.log(results)

        return results

    # use EM and F1 score since instruction examples follow QA format for the most part
    def compute_f1_em(eval_pred):
        predictions, labels = eval_pred
        results = {}

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        em_score = z.compute_em_instr_tune(decoded_preds, decoded_labels)
        f1_score = z.compute_f1_instr_tune(decoded_preds, decoded_labels)
        
        results["em"] = round(em_score, 3)
        results["f1"] = round(f1_score, 3)

        # Add stats regarding generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        results["mean_gen_len"] = np.mean(prediction_lens)
        results["min_gen_len"] = min(prediction_lens)
        results["max_gen_len"] = max(prediction_lens)

        # if True:
        if int(os.environ["LOCAL_RANK"])==0:
            wandb.log(results)
        return results
    
    ### Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = LongT5Config.from_pretrained(args.model_name)
    model = LongT5ForConditionalGeneration.from_pretrained(args.model_name, config=config)

    ### Prepare dataset
    train_path = os.path.join(args.json_dir,'train.json')
    valid_path = os.path.join(args.json_dir,'valid.json')
    test_path = os.path.join(args.json_dir,'test.json')
    
    train_df = pd.read_json(train_path, lines=True)
    train_df['answer'] = train_df['answer'].astype(str)
    valid_df = pd.read_json(valid_path, lines=True)
    valid_df['answer'] = valid_df['answer'].astype(str)
    test_df = pd.read_json(test_path, lines=True)
    test_df['answer'] = test_df['answer'].astype(str)
    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)
    test_ds = Dataset.from_pandas(test_df)
    dataset = DatasetDict()
    dataset["train"] = train_ds
    dataset["validation"] = valid_ds
    dataset["test"] = test_ds

    # tell # samples directly if needed
    if args.train_num_samples:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(min(train_df.shape[0], args.train_num_samples)))
    if args.valid_num_samples:
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(min(valid_df.shape[0], args.valid_num_samples)))
    if args.test_num_samples:
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(min(test_df.shape[0], args.test_num_samples)))

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    ### Training parameters
    # select correct syntax for trainer based on chosen optimizer
    if args.optimizer=="adafactor":
        optimizer = Adafactor(model.parameters(),
                lr=args.lr,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False)
        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoint_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            learning_rate=args.lr,
            lr_scheduler_type=args.lr_scheduler_type,
            per_device_train_batch_size=args.per_train_batch_size,
            per_device_eval_batch_size=args.per_eval_batch_size, 
            gradient_accumulation_steps=args.grad_acc_steps,
            optim="adafactor",
            adafactor=True,
            # max_steps=20,  # don't use if use num_train_epochs
            num_train_epochs=args.num_train_epochs,
            save_steps=args.save_steps, 
            eval_steps=args.eval_steps, 
            warmup_ratio=args.warmup_ratio,
            logging_steps=5,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            predict_with_generate=True,
            generation_config="google/long-t5-tglobal-base",
            generation_max_length=512,
            generation_num_beams=1,
            fp16=args.fp16,
            push_to_hub=False,
            report_to="wandb",
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_f1_em,
            optimizers=(optimizer, None)
        )
    else:  
        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoint_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            weight_decay=0.001,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-06,
            learning_rate=args.lr,
            lr_scheduler_type=args.lr_scheduler_type,
            optim="adamw_torch_fused",
            per_device_train_batch_size=args.per_train_batch_size,
            per_device_eval_batch_size=args.per_eval_batch_size, 
            gradient_accumulation_steps=args.grad_acc_steps,
            # max_steps=2,  # don't use if use num_train_epochs
            num_train_epochs=args.num_train_epochs,
            save_steps=args.save_steps, 
            eval_steps=args.eval_steps, 
            warmup_ratio=args.warmup_ratio,
            logging_steps=5,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            predict_with_generate=True,
            generation_config="google/long-t5-tglobal-base",
            generation_max_length=512,
            generation_num_beams=1,
            fp16=args.fp16,
            push_to_hub=False,
            report_to="wandb",
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_f1_em,
        )

    trainer.train()

    trainer.save_model(model_save_dir)
    print("FINAL MODEL SAVED TO:",model_save_dir)

    test_results = trainer.predict(
        tokenized_dataset["test"], 
        metric_key_prefix="test",
        max_new_tokens=512,
        num_beams=1, # use greedy decoding instead of beam search
    )
    predictions = test_results.predictions
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    preds_file_path = os.path.join(output_dir, f"preds.json")
    with open(preds_file_path, 'w') as f_out:
        json.dump(preds, f_out, indent=4)
    print(f"SAVED TEST PREDS, MOVING ON TO SAVE TEST SCORES")

    scores = test_results.metrics

    scores_file_path = os.path.join(output_dir, f"scores.txt")
    with open(scores_file_path, 'w') as f_out:
        json.dump(scores, f_out, indent=4)
    print(f"SCORES:", scores)
    print(f"SCORES SAVED TO", os.path.join(output_dir, f"scores.txt"))

    # if True:
    if int(os.environ["LOCAL_RANK"])==0:
        wandb.log(scores)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of path to appropriate train/val/test.json folder
    parser.add_argument("--json_dir", type=str) # TYPE denotes the specific instruction mix being used
    parser.add_argument("--wandb_project_name", 
                        type=str, 
                        default="longt5b_instr_tune",
                        ) 
    parser.add_argument("--wandb_group", type=str) 
    parser.add_argument("--model_name", type=str) 
    
    # directory to save instruction-tuned final model & checkpoints
    parser.add_argument("--output_dir",  
                        type=str,
                        ) # default: .../it_exploration_1/models/longt5_instr_tune

    # training args
    parser.add_argument("--lr", 
                        type=float, 
                        default=0.001,
                        ) 
    parser.add_argument("--lr_scheduler_type", 
                        type=str, 
                        default="constant",
                        ) 
    parser.add_argument("--optimizer", 
                        type=str, 
                        default="adafactor",
                        ) 
    parser.add_argument("--warmup_ratio", 
                        type=float, 
                        default=0,
                        ) 
    parser.add_argument("--save_total_limit", type=int, default=2) 
    parser.add_argument("--num_gpus", type=int) 
    parser.add_argument("--per_train_batch_size", type=int) 
    parser.add_argument("--per_eval_batch_size", type=int) 
    parser.add_argument("--grad_acc_steps", type=int) 
    parser.add_argument("--save_steps", type=int) 
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--fp16", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--resume_from_checkpoint", 
                        action="store_true",
                        default=False, 
                        ) 

    # dataset preparation args
    parser.add_argument("--train_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--valid_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--test_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    # parser.add_argument("prefix", 
    #                     type=str,
    #                     default=None,
    #                     help="prefix to add to start of each document for summarization")
    parser.add_argument("--suffix", 
                        type=str,
                        default="",
                        help="suffix to add to end of each document for summarization")
    
    args = parser.parse_args()

    main(args)
