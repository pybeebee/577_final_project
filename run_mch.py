# File to run 0-shot prompting evaluation and scoring on MultiNews & HotpotQA datasets
# Adapted from ZeroScrolls evaluation script
import json
import os
import sys
from datetime import datetime
import random
from evaluate import load
import zeroscrolls_metrics as z
import nltk
from statistics import geometric_mean
import numpy as np
import torch
from datasets import load_dataset
from fire import Fire
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, LongT5Config, T5ForConditionalGeneration, AutoModelForCausalLM, AutoModel, LEDForConditionalGeneration
from transformers import set_seed as hf_set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_to_max_input_tokens = {
    "google/flan-t5-xxl": 8192,
    "google/flan-t5-xl": 8192,
    "google/flan-t5-large": 8192,
    "google/flan-t5-base": 8192,
    "google/flan-t5-small": 8192,
    "google/flan-ul2": 8192,
    "bigscience/T0pp": 8192,
    "google-t5/t5-3b": 8192,
    "google/long-t5-tglobal-xl": 8192,
    "togethercomputer/Llama-2-7B-32K-Instruct": 32000,
    "THUDM/chatglm3-6b": 7168,
    "THUDM/chatglm3-6b-base": 7168,
    "THUDM/chatglm2-6b": 7168,
    "meta-llama/Llama-2-7b-hf": 4096,
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "mistralai/Mistral-7B-Instruct-v0.2": 8192,
    "biu-nlp/QAmden": 4096,
    "allenai/PRIMERA": 4096,
    "allenai/led-large-16384": 16384,
}

def removeprefix(given_string, prefix):
    return given_string[len(prefix):].replace("Summary:","").replace("Answer:","").strip()

def removesuffix(given_string, suffix):
    return given_string[:-len(suffix)]

def sanitize_text(text: str, lowercase: bool = False) -> str: # from qamden
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text

datasets = [
        'multi_news',
        'hotpot_qa',
        # 'wiki_hop',
        # 'multi_x_science_sum',
        # 'QMDSCNN',
    ]
prefixes = {
    "multi_news": "You are given several news articles, separated by '|||||'. Summarize the articles in one paragraph.",
    "hotpot_qa": "You are given a question and multiple supporting documents separated by '|||||'. Answer the question as concisely as you can, using a single phrase if possible. If the question is a yes/no question, answer 'yes' or 'no'.",
    "wiki_hop": "You are given multiple supporting documents separated by '|||||', a question, and list of answer candidates. Answer the question as concisely as you can, by selecting from the provided answer candidates.",
    "multi_x_science_sum": "You are given several documents, separated by '|||||'. Summarize the documents in one paragraph.",
    "QMDSCNN": "You are given a query and an article. Answer the query as concisely as you can by summarizing the relevant information from the article.",
}
suffixes = {
    "multi_news": "Summary:\n",
    "hotpot_qa": "Answer:\n",
    "wiki_hop": "Answer:\n",
    "multi_x_science_sum": "Summary:\n",
    "QMDSCNN": "Summary:\n",
}
truncation_separators = {
    "multi_news": "... [The rest of the articles are omitted]\n\n",
    "hotpot_qa": "... [The rest of the documents are omitted]\n\n",
    "wiki_hop": "... [The rest of the documents are omitted]\n\n",
    "multi_x_science_sum": "... [The rest of the documents are omitted]\n\n",
    "QMDSCNN": "... [The rest of the article is omitted]\n\n",
}
    # ... [The rest of the article is omitted]
    # ... [The rest of the story is omitted]
    # ... [The rest of the episode script is omitted]
    # ... [The rest of the paragraphs are omitted]
    # ... [The rest of the summaries are omitted]

def add_prefix_suffix(example_input, dataset_name, using_chat_model):
    prefix = prefixes[dataset_name]
    if using_chat_model and dataset_name not in {"multi_news", "multi_x_science_sum"}: ## TODO: ADD NON QA DATASETS TO LIST
        prefix += " Do not provide any explanation."
    prefix += "\n\n"
    
    example_input = prefix + example_input
    query_start_idx = len(example_input) # position where the first char in the query starts
    
    example_input += "\n\n" # also just the default suffix if no query to attach (i.e. if using chat model)
    if not using_chat_model:
        example_input += suffixes[dataset_name]
    query_end_idx = len(example_input) # note: using this formulation, need to subtract 1 if want to acess the last char in the query

    return example_input, query_start_idx, query_end_idx

def get_mn_input(example, dataset_name, using_chat_model):
    return "Articles:\n"+example["document"], example["summary"]

def get_hqa_input(example, dataset_name, using_chat_model):
    titles, sentences = example['context']["title"], example['context']["sentences"]
    paragraphs = [
        f"{title.strip()} {' '.join(sents).strip()}"
        for title, sents in zip(titles, sentences)
    ]
    context = f" ||||| ".join(paragraphs)
    
    example_input = sanitize_text(f"Question: {example['question'].strip()}\n\nSupporting Documents: {context.strip()}")

    return example_input, example["answer"]

def get_wikihop_input(example, dataset_name, using_chat_model):
    context = " ||||| ".join(example['supports']) 
    choices = ", ".join(example['candidates'])
    example_input = sanitize_text(f"Documents: {context.strip()}\n\nQuestion: {example['query'].strip()}\n\nAnswer Candidates: {choices.strip()}") 
    # import ipdb; ipdb.set_trace()
    return example_input, example['answer'] 

def get_multi_x_science_sum_input(example, dataset_name, using_chat_model):
    source_and_ref_abstracts = [example['abstract']] + example['ref_abstract']['abstract']
    context = f" ||||| ".join(source_and_ref_abstracts)
    return "Documents:\n"+context, example["related_work"]

def get_QMDSCNN_input(example, dataset_name, using_chat_model):
    return f"Query: {example['Query']}\n\nArticle: {example['Article']}", example['Summary']

format_input_fxns = {
    "multi_news": get_mn_input,
    "hotpot_qa": get_hqa_input,
    "wiki_hop": get_wikihop_input,
    "multi_x_science_sum": get_multi_x_science_sum_input,
    "QMDSCNN": get_QMDSCNN_input,
}

def process_model_input(tokenizer, example, max_tokens, device, dataset_name, model_name, using_llama, using_chat_model):
    # format the input appropriately for the task
    example_input, example_answer = format_input_fxns[dataset_name](example, dataset_name, using_chat_model)

    # add appropriate instruction prefix and answer header suffix
    example['input'], example['query_start_index'], example['query_end_index'] = add_prefix_suffix(example_input, dataset_name, using_chat_model)
    example['truncation_separator'] = truncation_separators[dataset_name]
        
    if using_llama and using_chat_model:
        example["input"] = "<<SYS>>\nYou are a helpful assistant that processes information from multiple documents.\n<</SYS>>\n[INST]\nUser:\n" + example["input"] + "\n[/INST]\nAssistant:"
        example['query_start_index']+=59
        example['query_end_index']+=59

    if "mistralai" in model_name:
        example["input"] = f"<s>[INST] {example['input']} [/INST]"
        example['query_start_index']+=10
        example['query_end_index']+=10

    tokenized_input_full = tokenizer(example["input"], return_tensors="pt").input_ids.to(device)
    if tokenized_input_full.shape[1] <= max_tokens:
        return tokenized_input_full, example_answer

    seperator_and_query_text = example['truncation_separator'] + example["input"][example['query_start_index']:]
    tokenized_seperator_and_query = tokenizer(seperator_and_query_text, return_tensors="pt").input_ids.to(device)
    input_without_query = example['input'][:example['query_start_index']]
    tokenized_input_without_query = tokenizer(input_without_query, return_tensors="pt").input_ids.to(device)
    tokenized_input_without_query = tokenized_input_without_query[:,
                                    :max_tokens - tokenized_seperator_and_query.shape[1]]

    tokenized_input = torch.cat([tokenized_input_without_query, tokenized_seperator_and_query], dim=1)
    return tokenized_input, example_answer

def main(model_name="google/flan-t5-small", 
         generations_dir="eval_mch_results", 
         save_folder_name="",
         max_inp_length=-1,
         max_examples_per_task=-1,
         use_test_set=False,
         ):
    
    def compute_rouge(predictions, references):
        rouge_metric = load("rouge")

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in references]
        
        result = rouge_metric.compute(predictions=decoded_preds,
                                    references=decoded_labels, 
                                    use_stemmer=True)
        results = {k: round(v*100, 3) for k, v in result.items()}  
        rouge_scores = list(results.values())[:-1]

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        results["***********ROUGE_GM***********"] = round(geometric_mean(rouge_scores),3)

        return results

    seed = 43
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)
    print("Params:")
    print(f"model: {model_name}")
    if save_folder_name=="":
        generations_dir = os.path.join(generations_dir, model_name.replace("-", "_").strip("/home/gkl7/multidoc1/it_exploration_1/"))
    else: 
        generations_dir = os.path.join(generations_dir, save_folder_name, model_name.replace("-", "_").strip("/home/gkl7/multidoc1/it_exploration_1/"))
    print(f"generations_dir: {generations_dir}")
    print(f"max_examples_per_task: {max_examples_per_task}")
    print("=" * 50)
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time as start: {time}")

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name not in model_to_max_input_tokens.keys():
        max_input_length = 4096
    else:
        max_input_length = model_to_max_input_tokens[model_name]

    using_llama=False
    using_chat_model=False
    if "long-t5" in model_name or "longt5" in model_name or "long_t5" in model_name:
        model = LongT5ForConditionalGeneration.from_pretrained(model_name).to(device) #device_map="auto")#, torch_dtype=torch.float16) #config=config, 
    elif "glm" in model_name: 
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map='auto', torch_dtype=torch.float16)
        using_chat_model=True
    elif "mistralai" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    elif "llama" in model_name: 
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
        using_llama=True
        if "chat" in model_name:
            using_chat_model=True
    elif model_name=="togethercomputer/Llama-2-7B-32K-Instruct":
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct", trust_remote_code=False, torch_dtype=torch.float16, device_map="auto")
        using_llama=True
    elif "QAmden" in model_name or "allenai" in model_name:
        model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    else: 
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    
    model = model.eval()
    metrics = dict()

    print(f"{model} model loaded!, device:{model.device}")

    print("Will write to:", generations_dir)
    os.makedirs(generations_dir, exist_ok=True)

    # write args to file for easy reference later on
    import sys
    try:
        from shlex import quote as cmd_quote
    except ImportError:
        from pipes import quote as cmd_quote
    cmdline = " ".join(map(cmd_quote, sys.argv[1:]))
    args_file_path = os.path.join(generations_dir, "args.txt")
    f = open(args_file_path, "a")
    f.write(cmdline)
    f.close()

    for dataset in datasets:
        generations = dict()
        preds = []
        refs = []
        print(f"Processing {dataset}")
        time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        print(f"time as start {dataset}: {time}")
        print(f"Loading {dataset}")

        if not use_test_set:
            if dataset=="multi_news":
                data = load_dataset("multi_news", split="validation[:500]")
            elif dataset=="hotpot_qa":
                data = load_dataset("hotpot_qa", "distractor", split="validation[:500]")
            elif dataset=="wiki_hop":
                data = load_dataset("json", data_files={'validation':'/home/gkl7/multidoc1/it_exploration_1/qangaroo_v1.1/wikihop/dev.json'}, split='validation[:500]')
            elif dataset=="multi_x_science_sum":
                data = load_dataset("multi_x_science_sum", split="validation[:500]")
            elif dataset=="QMDSCNN":
                data = load_dataset("EJinHF/QMDSCNN", split="validation[:500]")
        else:
            if dataset=="multi_news":
                data = load_dataset("multi_news", split="test[:500]")
            elif dataset=="hotpot_qa":
                data = load_dataset("hotpot_qa", "distractor", split="test[:500]")
            elif dataset=="wiki_hop":
                data = load_dataset("json", data_files={'validation':'/home/gkl7/multidoc1/it_exploration_1/qangaroo_v1.1/wikihop/dev.json'}, split='validation[-500:]') # no test set so just take last 500 dev set examples
            elif dataset=="multi_x_science_sum":
                data = load_dataset("multi_x_science_sum", split="test[:500]")
            elif dataset=="QMDSCNN":
                data = load_dataset("EJinHF/QMDSCNN", split="test[:500]")
        print(f"Loaded {dataset}")

        total_examples = len(data)
        for i, example in enumerate(data):

            if 0 < max_examples_per_task == i:
                print(f"Reached {max_examples_per_task} for {dataset}. Breaking")
                break

            model_input, example_answer = process_model_input(tokenizer, example, max_input_length, device, dataset, model_name, using_llama, using_chat_model)

            prediction_token_ids = model.generate(model_input,
                                                max_new_tokens=1024,
                                                do_sample=False,
                                                top_p=0,
                                                top_k=0,
                                                temperature=1)
            predicted_text = tokenizer.decode(prediction_token_ids[0], skip_special_tokens=True)
            
            if using_llama or using_chat_model or "mistralai" in model_name:
                prefix = tokenizer.decode(model_input[0], skip_special_tokens=True)
                predicted_text = removeprefix(predicted_text, prefix)
            
            generations[i] = predicted_text
            preds.append(predicted_text)
            refs.append(example_answer)
            print("Finished example", i, "of", total_examples)

        out_file_path = os.path.join(generations_dir, f"preds_{dataset}.json")
        with open(out_file_path, 'w') as f_out:
            json.dump(generations, f_out, indent=4)

        # Evaluate with correct metric (as det. by official ZeroScrolls site)
        if dataset in {'multi_news', 'multi_x_science_sum', 'QMDSCNN'}:
            metrics[dataset] = compute_rouge(preds, refs)
            print(metrics[dataset])

        elif dataset == "hotpot_qa":
            metrics["hotpot_qa_f1"] = z.compute_f1_instr_tune(preds, refs)
            metrics["hotpot_qa_em"] = z.compute_em_instr_tune(preds, refs)
            print(metrics["hotpot_qa_f1"])
            print(metrics["hotpot_qa_em"])

        elif dataset == "wiki_hop":
            metrics["wiki_hop_f1"] = z.compute_f1_instr_tune(preds, refs)
            metrics["wiki_hop_em"] = z.compute_em_instr_tune(preds, refs)
            print(metrics["wiki_hop_f1"])
            print(metrics["wiki_hop_em"])


        print(f"Done generating {len(generations)} examples from {dataset}")
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time at end: {time}")
    print(f"Look for predictions and scores in {generations_dir}")
    print("FINAL METRICS:", metrics)
    scores_file_path = os.path.join(generations_dir, f"scores.json")
    with open(scores_file_path, 'w') as f_out:
        json.dump(metrics, f_out, indent=4)

if __name__ == '__main__':
    Fire(main)