### FUNCTION TO PROMPT GENERATOR MODEL TO GENERATE INSTRUCTIONS BASED ON PREDEFINED INSTRUCTION GENERATION TEMPLATES
from transformers import pipeline
import torch
import os
from datetime import datetime
import pdb
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import random
import argparse
from multiprocessing import Pool
import numpy as np
from itertools import combinations
from tqdm import tqdm
import re
from ast import literal_eval
import pandas as pd 
from vllm import LLM, SamplingParams
from itertools import zip_longest
import time

# Login to HF to access Meta LLAMA model
from huggingface_hub import login
login("")

SAMPLING_PARAMS = SamplingParams(max_tokens=512) # use default params, match OpenAI params
CHOICES = [2,3,4]
SUMMARIZ_PREFIXES = [
    'Summarize the main points of the documents below:\n', 
    'Summarize: ',
    "",
    ]
SUMMARIZ_SUFFIXES = [
    '\n\nSummary:',
    '\n\nWhat are the main points of these documents?'
    "",
]

# function to add system prompt to generated instruction template
def llama_ify(prompt):
    res = """<<SYS>>
You are a helpful assistant.
<</SYS>>
[INST]
User:
%s
[/INST]
Assistant:"""%(prompt)
    return res

######################################################################
###### Define methods to sample instruction generation prompts #######
######################################################################
def template_A_0(snippet1, snippet2):
    prompt = """You are a search engine. A person queried something in detail and the most relevant snippets about the query are as follows.
Query: X
Snippets: '%s', '%s'
What could the detailed query X be? Answer with a plausible question or instruction.
X:"""%(snippet1, snippet2)
    return prompt

def template_A_1_0(doc1, doc2, snippet1, snippet2):
    prompt = """'%s', '%s'
You are a search engine. A person queried something in detail about the documents above and the most relevant snippets about the query are as follows.
Query: X
Snippets: '%s', '%s'

What could the detailed query X be? Answer with a plausible question or instruction.
X:"""%(doc1, doc2, snippet1, snippet2)
    return prompt

def template_B_0_3(snippet1, snippet2):
    prompt = """Instruction: X
Snippets: '%s', '%s'

What kind of instruction could these two snippets be the answer to? You must answer with a specific question that can ONLY be answered by utilizing information in both snippets. You will be penalized if the question concerns only one snippet. Format your answer as plain text. Say "Not sure" if you can't come up with a high-quality question.
X:"""%(snippet1, snippet2)
    return prompt

def template_B_1_3(doc1, doc2, snippet1, snippet2):
    prompt = """'%s', '%s'

Instruction: X
Snippets: '%s', '%s'

What kind of instruction could these two snippets be the answer to? You must answer with a specific question that can ONLY be answered by utilizing information in both snippets. You will be penalized if the question concerns only one snippet. Format your answer as plain text. Say "Not sure" if you can't come up with a high-quality question. \nRead the question again: What kind of instruction could these two snippets be the answer to? You must answer with a specific question that can ONLY be answered by utilizing information in both snippets. You will be penalized if the question concerns only one snippet. Format your answer as plain text. Say "Not sure" if you can't come up with a high-quality question.
X:"""%(doc1, doc2, snippet1, snippet2)
    return prompt

def template_B_1_4(doc1, doc2, snippet1, snippet2):
    prompt = """'%s', '%s'

Instruction: X
Snippets: '%s', '%s'

What kind of instruction could these two snippets be the answer to? You must answer with a specific question that can ONLY be answered by utilizing information in both snippets. You will be penalized if the question concerns only one snippet. Format your answer as plain text. Say "Not sure" if you can't come up with a high-quality question.
X:"""%(doc1, doc2, snippet1, snippet2)
    return prompt

def template_D_3(doc1, doc2):
    prompt = """Below are two documents. Select 3 sentences that are most pertinent to the content of the documents. Then generate a single question or instruction that can ONLY be answered or responded to using ALL 3 sentences.
'%s', '%s'

Make sure EACH snippet is critical to answering the question/instruction. You will be penalized if your proposed question/instruction concerns only one or two snippets. Format your proposal as:

Question/Instruction: 
Snippet 1:
Snippet 2: 
Snippet 3:"""%(doc1, doc2)
    return prompt

def template_D_4(doc1, doc2):
    prompt = """Below are two documents. Select 4 sentences that are most pertinent to the content of the documents. Then generate a single question or instruction that can ONLY be answered or responded to using ALL 4 sentences.
'%s', '%s'

Make sure EACH snippet is critical to answering the question/instruction. You will be penalized if your proposed question/instruction concerns only one or two snippets. Format your proposal as:

Question/Instruction: 
Snippet 1:
Snippet 2: 
Snippet 3:
Snippet 4:"""%(doc1, doc2)
    return prompt

def template_E_4(cluster):
    prompt = """The documents below are ordered by relevance to a given query, with the first document being most relevant."""
    for doc in cluster:
        prompt += '\n\n'
        prompt += '\''
        prompt += doc
        prompt += '\''

    prompt += "\n\nGiven that the documents are ordered from most to least useful to answering the query, what could be the query X?\nX:"
    return prompt

def template_E_6(*args):
    prompt = ""
    for doc in args:
        prompt += '\''
        prompt += doc
        prompt += "\'\n\n"
    prompt += """Select two sentences from the above documents. Generate a query to either compare or contrast the information identified. Format your answer as:

Sentence 1:
Sentence 2:
Query:"""
    return prompt

# save directly
def template_E_5(cluster):
    prompt = "Documents:\n"
    n = random.choice(CHOICES)
    sents = [sent for doc in cluster for sent in doc.split('\n')]
    snips = random.sample(sents, min(n, len(sents)))
    for doc in cluster:
        prompt += '\''
        prompt += doc
        prompt += '\'\n\n'

    prompt += "Snippets:\n"
    for snip in snips:
        prompt += '\''
        prompt += snip
        prompt += '\'\n'
    prompt = prompt + "\nAbove is a series of snippets extracted from several documents given above. How many of the above documents do these snippets come from? Provide your answer as a single number." # -2 to remove last comma and space
    return prompt, len(snips)

# save directly
def template_E_2_long(doc1, doc2):
    doc1_split = doc1.split('\n')
    doc2_split = doc2.split('\n')
    try:
        doc1_snips = random.sample(doc1_split, random.choice(range(2,min(4,len(doc1_split)))))
    except: 
        doc1_snips = doc1_split[0]
    try:
        doc2_snips = random.sample(doc2_split, random.choice(range(2,min(4,len(doc2_split)))))
    except: 
        doc2_snips = doc2_split[0]
    
    # modify doc1
    for i, pos in enumerate(random.sample(range(len(doc1_split)), min(len(doc2_snips), len(doc1_split)))):
        doc1_split.insert(pos, doc2_snips[i])
    prompt1 = "\n".join(doc1_split) + "\n\nIn the document above, there are 2-4 snippets that do not belong. Which are they?"

    # modify doc2
    for i, pos in enumerate(random.sample(range(len(doc2_split)), min(len(doc1_snips), len(doc2_split)))):
        doc2_split.insert(pos, doc1_snips[i])
    prompt2 = "\n".join(doc2_split) + "\n\nIn the document above, there are 2-5 snippets that do not belong. Which are they?"
    
    return prompt1, prompt2, doc2_snips, doc1_snips

# save directly
def template_E_2_short(doc1, doc2):
    doc1_split = doc1.split('\n')
    doc2_split = doc2.split('\n')
    try:
        doc1_snips = random.sample(doc1_split, random.choice(range(2,min(4,len(doc1_split)))))
    except: 
        doc1_snips = doc1_split[0]
    try:
        doc2_snips = random.sample(doc2_split, random.choice(range(2,min(4,len(doc2_split)))))
    except: 
        doc2_snips = doc2_split[0]
    
    # modify doc1
    for i, pos in enumerate(random.sample(range(len(doc1_split)), min(len(doc2_snips), len(doc1_split)))):
        doc1_split.insert(pos, doc2_snips[i])
    prompt1 = "\n".join(doc1_split) + "\n\nIn the document above, there are a few snippets that do not belong. How many such misplaced snippets are there?"

    # modify doc2
    for i, pos in enumerate(random.sample(range(len(doc2_split)), min(len(doc1_snips), len(doc2_split)))):
        doc2_split.insert(pos, doc1_snips[i])
    prompt2 = "\n".join(doc2_split) + "\n\nIn the document above, there are a few snippets that do not belong. How many such misplaced snippets are there?"

    return prompt1, prompt2, len(doc2_snips), len(doc1_snips)

# summarization task
def template_S_1(source_docs, target_summary):
    # format input and output appropriately
    docs = source_docs.replace("<mask>", "").replace("<s>", "").replace("<doc-sep>","\n\n").replace("</s>","").strip()
    summ = target_summary.replace("<mask>", "").replace("<s>", "").replace("<doc-sep>","").replace("</s>","")

    if random.random() > 0.5:
        prefix = random.choice(SUMMARIZ_PREFIXES) 
        prompt = prefix + docs
    else: 
        suffix = random.choice(SUMMARIZ_SUFFIXES)
        prompt = docs + suffix 
    return prompt, summ

# long range task
def template_R_1(doc1, doc2):
    doc1_split = doc1.split('\n')
    doc2_split = doc2.split('\n')
    snippets = doc1_split + doc2_split
    random.shuffle(snippets) # randomly reorder the snippets
    prompt = "\n".join(snippets)
    prompt += "\n\n"

    prompt += r"""Two documents have been scrambled up randomly into the text above, which is composed of segments separated by \n. Reorder these \n-separated segments to reformulate the two original documents. Your output should be the two original documents enclosed in single quotes and separated with a comma. For instance, if the two documents' contents were "text in one document" and "text in another document" your output would be:
'text in one document', 'text in another document'""" # use r in front of string so that \n shows up as text instead of newline
    return prompt


# Function to generate and save instruction tuning examples of form S1
def generate_and_save_S1_instructions(
        input_dir, # input dir: greedy_entity_pyramid_pyramid_rouge_03_05
        output_dir,
):
    tokenizer = AutoTokenizer.from_pretrained('biu-nlp/QAmden')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prompt_dir = os.path.join(output_dir, "prompts")
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)

    cluster_id_dir = os.path.join(output_dir, "cluster_ids")
    if not os.path.exists(cluster_id_dir):
        os.makedirs(cluster_id_dir)
    
    ans_dir = os.path.join(output_dir, "answers")
    if not os.path.exists(ans_dir):
        os.makedirs(ans_dir)

    for split in ["train", "test", "valid"]:
        num_prompts = 0

        # Access all the primera summarization data .pt files
        all_files_in_split = [
            f
            for f in os.listdir(os.path.join(input_dir, split))
            if f.endswith(".pt")
        ]
        all_files_in_split = all_files_in_split[::-1]

        for file_idx, file_name in enumerate(tqdm(all_files_in_split)):
            print(file_name)

            all_prompts = []
            answers = []
            instr_cluster_ids = []

            all_examples_in_file = torch.load(os.path.join(input_dir, split, file_name))

            for example in all_examples_in_file:
                source_docs = tokenizer.decode(example['src'])
                target_summ = tokenizer.decode(example['tgt'])
                prompt, answer = template_S_1(source_docs, target_summ)
                all_prompts.append(prompt)
                answers.append(answer)
                instr_cluster_ids.append(1000000)
                num_prompts += 1
            
            prompt_file_name = file_name[file_name.index("newshead")+9:]
            torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
            print("Finished saving prompt slice %d at %s"%(file_idx, prompt_dir+"/"+prompt_file_name))

            cluster_id_file_name = prompt_file_name
            torch.save(instr_cluster_ids, os.path.join(cluster_id_dir, cluster_id_file_name))
            print("Finished saving prompt slice %d at %s"%(file_idx, cluster_id_dir+"/"+cluster_id_file_name))

            ans_file_name = prompt_file_name
            torch.save(answers, os.path.join(ans_dir, ans_file_name))
            print("Finished saving answers slice %d at %s"%(file_idx, ans_dir+"/"+ans_file_name))

            print(f"TOTAL NUMBER OF PROMPTS (as of slice {file_idx}): {num_prompts}")

        print("done!")   
            

# Function to generate and save instruction tuning prompts/examples
def generate_and_save_instructions(
    input_dir,
    output_dir,
    model_name,
    instruction_format,
    use_existing_prompts,
    given_prompt_dir,
    start_instr_file_idx
):
    # select specified generator model
    if model_name=="llama2-chat-7b":
        # VLLM use
        generator_model = LLM(
            model="meta-llama/Llama-2-7b-chat-hf", 
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            )
    
    elif model_name=="mistral-7b-instruct-v2":
        # VLLM use
        generator_model = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.2", 
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            )
    
    # ensure output dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Access all the newshead file addresses (train/valid/test split .pt files)
    all_files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(".pt")
    ]
    all_files = all_files[::-1] 
    count = 0

    # For each data split (train/valid/test)
    for file_idx, file_name in enumerate(tqdm(all_files)):
        print(file_name)
        num_prompts=0

        # prepare directories to save outputs
        all_prompts = []
        instr_cluster_ids = []
        source_docs = []
        num_snips = []
        answers = []
        prompt_dir = os.path.join(output_dir, "prompts")
        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir)
        docs_dir = os.path.join(output_dir, "source_docs")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
        cluster_id_dir = os.path.join(output_dir, "cluster_ids")
        if not os.path.exists(cluster_id_dir):
            os.makedirs(cluster_id_dir)
        
        # set up for saving answers if needed
        if instruction_format=="E_2" or instruction_format=="E_5":
            num_snips_dir = os.path.join(output_dir, "num_snips")
            if not os.path.exists(num_snips_dir):
                os.makedirs(num_snips_dir)
        elif instruction_format=="R_1":
            ans_dir = os.path.join(output_dir, "answers")
            if not os.path.exists(ans_dir):
                os.makedirs(ans_dir)

        # if prompts haven't been created yet
        if not use_existing_prompts:
            if instruction_format in {"A_0", "A_1_0", "B_0_3", "B_1_3", "B_1_4"}:
                base_snippet_pairs_path = "./base_snippet_pairs_regen/" + file_name.replace('.pt', '.csv')
                selected_sent_pairs = pd.read_csv(base_snippet_pairs_path, index_col=None, converters={"articles": literal_eval,"answer": literal_eval, "cluster_id": literal_eval})

                all_source_docs = selected_sent_pairs['articles'].tolist()
                all_answers = selected_sent_pairs['answer'].tolist()
                all_cluster_ids = selected_sent_pairs['cluster_id'].tolist()
                file_idx = 0

                # for each document pair and selected snippet pair, process according to the specified instruction template(s)
                for docs, snippets, cluster_id in zip(all_source_docs, all_answers, all_cluster_ids):
                    if instruction_format=="A_0":
                        prompt = template_A_0(snippets[0], snippets[1])
                        source_docs.append([docs[0], docs[1]])
                    elif instruction_format=="A_1_0":
                        prompt = template_A_1_0(docs[0], docs[1], snippets[0], snippets[1])
                    elif instruction_format=="B_0_3":
                        prompt = template_B_0_3(snippets[0], snippets[1])
                        source_docs.append((docs[0], docs[1]))
                    elif instruction_format=="B_1_3":
                        prompt = template_B_1_3(docs[0], docs[1], snippets[0], snippets[1])
                    elif instruction_format=="B_1_4":
                        prompt = template_B_1_4(docs[0], docs[1], snippets[0], snippets[1])
                    all_prompts.append(llama_ify(prompt))
                    instr_cluster_ids.append(cluster_id)
                    num_prompts += 1
                    
                    # if too many prompts have been processed, save and dump to avoid overloading memory
                    if num_prompts > 4:
                        prompt_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                        torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
                        print("Finished saving prompt slice %d at %s"%(file_idx, prompt_dir+"/"+prompt_file_name))

                        cluster_id_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                        torch.save(instr_cluster_ids, os.path.join(cluster_id_dir, cluster_id_file_name))
                        print("Finished saving cluster ids slice %d at %s"%(file_idx, cluster_id_dir+"/"+cluster_id_file_name))

                        if instruction_format=="A_0" or instruction_format=="B_0_3":
                            source_docs_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                            torch.save(source_docs, os.path.join(docs_dir, source_docs_file_name))
                            print("Finished saving source documents slice %d at %s"%(file_idx, docs_dir+"/"+source_docs_file_name))
                            source_docs = []

                        print("TOTAL NUMBER OF PROMPTS:",num_prompts)
                        num_prompts = 0
                        all_prompts = []
                        instr_cluster_ids = []
                        file_idx += 1

            elif instruction_format in {"D_3", "D_4", "E_4", "E_5", "E_6"}:
                all_clusters = torch.load(os.path.join(input_dir, file_name))

                file_idx = 0
                for cluster in all_clusters:
                    print("On cluster",count)

                    if instruction_format=="E_4":
                        prompt = template_E_4(cluster)
                        all_prompts.append(llama_ify(prompt))
                        instr_cluster_ids.append(count)
                        num_prompts += 1
                    elif instruction_format=="E_5":
                        prompt, snip_count = template_E_5(cluster)
                        all_prompts.append(prompt)
                        num_snips.append(snip_count)
                        instr_cluster_ids.append(count)
                        num_prompts += 1
                    else:
                        doc_idx_combos = list(combinations(range(len(cluster)), 2))

                        for idx1, idx2 in doc_idx_combos:
                            if instruction_format=="D_3":
                                prompt = template_D_3(cluster[idx1], cluster[idx2])
                            elif instruction_format=="D_4":
                                prompt = template_D_4(cluster[idx1], cluster[idx2])
                            elif instruction_format=="E_6":
                                prompt = template_E_6(cluster[idx1], cluster[idx2])
                            all_prompts.append(llama_ify(prompt))
                        num_prompts += len(doc_idx_combos)
                        instr_cluster_ids += [count] * len(doc_idx_combos)
                    
                    # if too many prompts have been processed, save and dump to avoid overloading memory
                    if num_prompts > 4:
                        prompt_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                        torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
                        print("Finished saving prompt slice %d at %s"%(file_idx, prompt_dir+"/"+prompt_file_name))

                        cluster_id_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                        torch.save(instr_cluster_ids, os.path.join(cluster_id_dir, cluster_id_file_name))
                        print("Finished saving cluster ids slice %d at %s"%(file_idx, cluster_id_dir+"/"+cluster_id_file_name))

                        if instruction_format=="E_5":
                            num_snips_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                            torch.save(num_snips, os.path.join(num_snips_dir, num_snips_file_name))
                            print("Finished saving snippet counts slice %d at %s"%(file_idx, num_snips_dir+"/"+num_snips_file_name))
                            num_snips = []

                        print("TOTAL NUMBER OF PROMPTS:",num_prompts)
                        num_prompts = 0
                        all_prompts = []
                        instr_cluster_ids = []
                        file_idx += 1
                    
                    count += 1

            elif instruction_format in {"E_2", "R_1"}:
                # Take two documents from different clusters. In the longer one, every 1/5th of it, place a sentence from the other document. Ask model: which sentences do not belong?
                all_clusters = torch.load(os.path.join(input_dir, file_name))

                doc_pairs = list(set((doc1, doc2) for cluster1, cluster2 in zip(all_clusters, all_clusters[1:]) for doc1, doc2 in zip(cluster1, cluster2)))

                file_id = 0
                for doc1, doc2 in doc_pairs:
                    if instruction_format=="E_2":
                        if random.random() > 0.5: # do long answer with probability 1/2
                            prompt1, prompt2, answer1, answer2 = template_E_2_long(doc1, doc2)
                            num_snips.append(', '.join(f"'{snip}'" for snip in answer1)) # answer is the extractive snippets
                            num_snips.append(', '.join(f"'{snip}'" for snip in answer2))
                        else:
                            prompt1, prompt2, answer1, answer2 = template_E_2_short(doc1, doc2)
                            num_snips.append(answer1) # answer is a integer
                            num_snips.append(answer2)

                        all_prompts.append(prompt1)
                        all_prompts.append(prompt2)
                        instr_cluster_ids.append(1000000)
                        instr_cluster_ids.append(1000000)
                        num_prompts += 2
                    elif instruction_format=="R_1":
                        prompt = template_R_1(doc1, doc2)
                        answer = "'%s', '%s'"%(doc1, doc2)
                        all_prompts.append(prompt)
                        answers.append(answer)
                        instr_cluster_ids.append(1000000)
                        num_prompts += 1
                    
                    # if too many prompts have been processed, save and dump to avoid overloading memory
                    if num_prompts > 1999:
                        prompt_file_name = file_name[:-2]+"%d.pt"%(file_id)
                        torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
                        print("Finished saving prompt slice %d at %s"%(file_id, prompt_dir+"/"+prompt_file_name))

                        cluster_id_file_name = file_name[:-2]+"%d.pt"%(file_id)
                        torch.save(instr_cluster_ids, os.path.join(cluster_id_dir, cluster_id_file_name))
                        print("Finished saving cluster ids slice %d at %s"%(file_id, cluster_id_dir+"/"+cluster_id_file_name))

                        if instruction_format=="E_2":
                            num_snips_file_name = prompt_file_name
                            torch.save(num_snips, os.path.join(num_snips_dir, num_snips_file_name))
                            print("Finished saving snippet counts slice %d at %s"%(file_id, num_snips_dir+"/"+num_snips_file_name))
                            num_snips = []

                        elif instruction_format=="R_1":
                            ans_file_name = prompt_file_name
                            torch.save(answers, os.path.join(ans_dir, ans_file_name))
                            print("Finished saving R1 answers slice %d at %s"%(file_id, ans_dir+"/"+ans_file_name))
                            answers = []

                        print("TOTAL NUMBER OF PROMPTS:",num_prompts)
                        num_prompts = 0
                        all_prompts = []
                        instr_cluster_ids = []
                        file_id += 1
                
            
            torch.save(all_prompts, os.path.join(prompt_dir, file_name))
            print("Finished saving remaining prompts at %s"%(prompt_dir+"/"+file_name))

            torch.save(instr_cluster_ids, os.path.join(cluster_id_dir, file_name))
            print("Finished saving remaining cluster_ids at %s"%(cluster_id_dir+"/"+file_name))

            if instruction_format=="A_0" or instruction_format=="B_0_3":
                torch.save(source_docs, os.path.join(docs_dir, file_name))
                print("Finished saving remaining source documents at %s"%(docs_dir+"/"+file_name))

            if instruction_format=="E_5" or instruction_format=="E_2":
                torch.save(num_snips, os.path.join(num_snips_dir, file_name))
                print("Finished saving remaining snippet counts at %s"%(num_snips_dir+"/"+file_name))

            if instruction_format=="R_1":
                torch.save(answers, os.path.join(ans_dir, file_name))
                print("Finished saving remaining R1 answers at %s"%(ans_dir+"/"+file_name))

            print("TOTAL NUMBER OF PROMPTS:",num_prompts)
        
        ##################################################
        ########## Generate LLM-Created Prompts ##########
        ##################################################
        if instruction_format!="E_2" and instruction_format!="E_5" and instruction_format!="R_1" and instruction_format!="S_1":
            
            if use_existing_prompts:
                prompt_dir = given_prompt_dir

            # ensure output directory exists
            instr_dir = os.path.join(output_dir, "instructions")
            if not os.path.exists(instr_dir):
                os.makedirs(instr_dir)
        
            all_files = sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt")
            ])

            # For each data split (train/valid/test)
            for file_idx, pt_file in enumerate(tqdm(all_files)):
                                  
                if start_instr_file_idx!=-1 and file_idx < start_instr_file_idx:
                    continue
                
                elif (start_instr_file_idx==-1) or (start_instr_file_idx!=-1 and file_idx >= start_instr_file_idx): 

                    # write current filename to log (in case need to pause/resume)
                    f = open(os.path.join(output_dir,"num_instr_slices_generated_so_far.txt"), "w")
                    f.write("\nCurerntly generating instructions for prompt file: "+str(pt_file))
                    f.write("\nCorresponding file_idx: "+str(file_idx))
                    f.close()

                    ## USING PIPELINE
                    # process in files of 200 prompts
                    print("On slice", file_idx)
                    cleaned_instructions = []
                    prompt_slice = torch.load(os.path.join(prompt_dir, pt_file))
                    instructions = generator_model.generate(prompt_slice, SAMPLING_PARAMS)
                    cleaned_instructions = [instrxn.outputs[0].text.removeprefix(prompt) for instrxn, prompt in zip(instructions, prompt_slice)]
                    
                    instr_file_name = pt_file

                    torch.save(cleaned_instructions, os.path.join(instr_dir, instr_file_name))
                    print("Finished saving cleaned instructions part %d at %s"%(file_idx,  instr_dir+"/"+instr_file_name))

        print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./base_articles",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_instructions/style_",
    )
    parser.add_argument(
        '--use_existing_prompts', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--given_prompt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--given_date_time",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_instr_file_idx",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--instruction_format",
        choices=[
            "A_0",
            "A_1_0",
            "B_0_3",
            "B_1_3",
            "B_1_4",
            "D_3",
            "D_4",
            "E_2",
            "E_3",
            "E_4",
            "E_5",
            "E_6",
            "S_1",
            "R_1",
        ],
        default="A_0",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        choices=[
            "llama2-chat-7b",
            "llama2-chat-13b",
            "mistral-7b-instruct-v2",
        ],
        default="llama2-chat-7b",
        type=str,
    )
    args = parser.parse_args()
    print(args)

    if args.given_date_time==None:
        current_date_time = str(datetime.now().strftime("%Y-%m-%d_hr%H-min%M"))
    else: 
        current_date_time = args.given_date_time
    
    if args.instruction_format=="S_1":
        output_dir = args.output_dir+args.instruction_format+"/model_"+args.model_name+"/"+current_date_time
        generate_and_save_S1_instructions(
            args.input_dir,
            output_dir,
        )
    
    else:
        output_dir = args.output_dir+args.instruction_format+"/model_"+args.model_name+"/"+current_date_time
        generate_and_save_instructions(
            args.input_dir,
            output_dir,
            args.model_name,
            args.instruction_format,
            args.use_existing_prompts,
            args.given_prompt_dir,
            args.start_instr_file_idx,
        )



