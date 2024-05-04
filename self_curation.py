### CURATES THE DATA: For each of train/test/val splits, saves the data for each as a single json file with one json object per line.
### NOTE: NO NEED TO RUN THIS FOR TEMPLATES E_2 and E_5

import argparse
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import re
import random
from random import shuffle
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import json
from vllm import LLM, SamplingParams

SAMPLING_PARAMS = SamplingParams(max_tokens=512) # use default params, match OpenAI params

# Login to HF to access Meta LLAMA model
from huggingface_hub import login
login("")

# Summarization pipeline for turning extractive answers to abstractive
summarizer_model = pipeline("summarization", model="slauw87/bart_summarisation")
def abstractify(penultimate_answer):
    summarized_answer = summarizer_model(penultimate_answer, max_length=512)
    return summarized_answer[0]['summary_text']

# add system prompt for llama if needed
def llama_ify(prompt):#, tokenizer):
    res = """<<SYS>>
You are a helpful assistant.
<</SYS>>
[INST]
User:
%s
[/INST]
Assistant:"""%(prompt)
    return res

# define string templates to be used in self-curation
CURATION_PREFIX = """Below is an Instruction from a user and a candidate Response from an AI Assistant. The goal of this AI Assistant is to generate a Response that effectively addresses the user's Instruction and that, in order to answer, requires the ability to reason over multiple documents. The Response should be a targeted question, instruction, prompt, or task that requires the use of information from different positions in the provided texts.

Evaluate whether the Response is a good example of how an AI Assistant should respond to the user's Instruction. Assign a score to the Response using the following 5-point scale:

1: It means the Response is incomplete, vague, off-topic, or not exactly what the user asked for in the Instruction. Perhaps it provides an incomplete prompt. Or it can be answered without looking at source documents provided in the Instruction. Or some content seems missing, the opening sentence repeats user's question, or it contains is irrelevant to the source documents or snippets provided in the Instruction.
2: It means the Response addresses some of the asks from the user but does not directly address the user's Instruction. For example, the Response only leverages one of several source documents or snippets provided in the Instruction. Or the Response can be answered using only ONE source document or snippet, and thus does not effectively assess and require use of multi-document reasoning capabilities.
3: It means the Response is fair and addresses all the basic asks from the user. It is complete and self contained and is relevant to most of the documents or snippets provided, but not all. It may be somewhat helpful toward assessing an agent's multi-document reasoning capability but still has room for improvement.
4: It means the Response is good quality. Specifically, the Response can only be answered by performing reasoning across most of the documents or snippets provided in the Instruction. The provided documents or snippets include all the information required to answer the Response, i.e. no information beyond that provided in the Instruction is needed. The Response has minor room for improvement, e.g. more concise and focused.
5: It means the Response is perfect, i.e. it can only be answered with strong ability to extract and synthesize information across the documents or snippets provided in the Instruction. The Response utilizes ALL documents or snippets provided in the instruction. The provided documents or snippets include all the information required to answer the Response. It is well-written and effective toward the AI Assistant's goal and has no irrelevant content.

Assess the Response and assign a rating score using this scale. Respond with "Score: <rating>".

"""

def curation_prompt1(prompt,query):
    return CURATION_PREFIX+f"Instruction:\n{prompt}\n\nResponse:\n{query}"

# prompt adapted from Self Rewarding LMs paper
def curation_prompt2(prompt, query):
    return """Review the user’s request and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant to and somewhat completes the user’s request, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s request, but does not completely resolve the query or provide a direct answer that adheres to user-stated guidelines.
- Award a third point if the response answers the basic elements of the user’s request in a useful way.
- Grant a fourth point if the response clearly addresses the user’s request directly and comprehensively, and is well-written and specific, even if there is slight room for improvement in clarity, conciseness, or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s request, without extraneous information, and demonstrating a high-quality, effective response.

User: %s

<response>%s</response>

After examining the user’s instruction and the response:

- Issue your total score using the format: “Score: <total points>”
- Briefly justify your total score, up to 100 words.

When assigning points to the response, keep in mind that the response should be a targeted question, instruction, prompt, or task that requires the use of information across various parts of the text provided by the user request."""%(prompt, query) 

def curation_prompt3(prompt, query):
    return """Below is an Instruction from a user and a Response. The goal is for the Response to effectively address the user's Instruction and be a targeted question, instruction, prompt, or task that requires the use of information from different parts in the provided texts in order to answer.

Evaluate whether the Response accomplishes this using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the Response is incomplete or related to but not exactly what the user asked for in the Instruction. Perhaps it can be answered without looking at source documents provided in the Instruction, or the opening sentence repeats user's question, or it is irrelevant to the source documents or snippets provided in the Instruction. 
- Add 1 point if the Response addresses some of the asks in the Insturction. For example, the Response only leverages one of several source documents or snippets provided in the Instruction. Or the Response can be answered using only ONE source document or snippet, and thus does not require information from various parts of provided text in the Instruction.
- Add 1 point if the Response is fair and addresses all the basic asks from the user. It is complete and self contained and is relevant to most of the documents or snippets provided, but not all. It may be somewhat helpful toward assessing an agent's multi-document reasoning capability but still has room for improvement.
- Add 1 point if the Response is good quality and can only be answered by performing reasoning across various parts of the documents or snippets provided in the Instruction. The provided documents or snippets include all the information required to answer the Response. The Response has minor room for improvement, e.g. more concise and focused.
- Add 1 point if the Response is perfect, i.e. it can only be answered with strong ability to extract and synthesize information across the documents or snippets provided in the Instruction. The Response utilizes ALL documents or snippets provided in the instruction. The provided documents or snippets include all the information required to answer the Response. It is well-written and effective and has no irrelevant content.

User Instruction: %s

<response>%s</response>

After examining the user’s Instruction and the Response:

- Issue your total score using the format: “Score: <total points>”
- Briefly justify your total score, up to 100 words.

When assigning points to the response, keep in mind that the response should be a targeted question, instruction, prompt, or task that requires the use of information across various parts of the text provided by the user Instruction."""%(prompt, query)

REGEX = re.compile(r"[Ss]core:\s*(\d+)")

LENGTH_ENHANCEMENTS = ["\nRespond in 2 sentences.", "\nRespond briefly."]

# IF GENERATOR MODEL WAS LLAMA
SYS_PREFIX = "<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n[INST]\nUser:\n"
SYS_SUFFIX = "\n[/INST]\nAssistant:"

# TODO: UPDATE SYS PREFIX/SUFFIX AS NEEDED IF GENERATOR IS ****NOT**** LLAMA

# extract the generated instruction/query from the generator model's response
def get_query(instr_style: str,
              instr: str,
              ):

    if instr_style=="A_0" or instr_style=="A_1_0" or instr_style=="B_0_3" or instr_style=="B_1_3" or instr_style=="B_1_4":
        instr = instr.replace('"X"',"")
        pattern = r'"([^"]*?[^X"]|\S*?\?)"'
        try:
            query = re.findall(pattern, instr)[0]
        except:
            pattern = r'^([^!?.]*\?[^\S]*)'
            try:
                query = re.search(pattern, instr).group(1)
            except:
                pattern = r'\n\n(.*\?\n\n)'
                try:
                    query = re.search(pattern, instr).group(1)
                except: 
                    query = ""
        return query.strip()
    
    elif instr_style=="D_3" or instr_style=="D_4":
        snippets = []
        pattern = r'"([^"]*)"'
        query = ""
        for part in instr.split("\n"):
            if len(part)>0 and part[-1]==':':
                continue
            if "Question/Instruction:" in part:
                query = part.strip("Question/Instruction:")
            elif "uestion" in part or "nstruction" in part:
                try:
                    query = part[part.index(": ")+2:]
                except:
                    try: 
                        query = part[part.index(":")+1:]
                    except:
                        pass
            elif "?" in part:
                pattern = r'^([^!?.]*\?[^\S]*)'
                try:
                    query = re.search(pattern, part).group(1)
                except:
                    pass
            elif "1: " in part:
                try:
                    snip = part[part.index("1: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
            elif "2: " in part:
                try:
                    snip = part[part.index("2: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
            elif "3: " in part:
                try:
                    snip = part[part.index("3: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
            elif "4: " in part:
                try:
                    snip = part[part.index("4: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
        return query.strip(), snippets
    
    elif instr_style=="E_4":
        instr = instr.replace('"X"',"")
        query = ""
        pattern = r'"([^"]*?[^X"]|\S*?\?)"'
        try:
            query = re.findall(pattern, instr)[-1]
        except:
            pass
            pattern = r'^([^!?.]*\?[^\S]*)'
            try:
                query = re.search(pattern, instr).group(1)
            except:
                pass
        return query.strip()
    elif instr_style=="E_6":
        snippets = []
        pattern = r'"([^"]*)"'
        query = ""
        for part in instr.split("\n"):
            if len(part)>0 and part[-1]==':':
                continue
            if "Query:" in part:
                query = part.strip("Query:")
            elif "uery" in part or "nstruction" in part:
                try:
                    query = part[part.index(": ")+2:]
                except:
                    try: 
                        query = part[part.index(":")+1:]
                    except:
                        pass
            elif "?" in part:
                pattern = r'^([^!?.]*\?[^\S]*)'
                try:
                    query = re.search(pattern, part).group(1)
                except:
                    pass
            elif "1: " in part:
                try:
                    snip = part[part.index("1: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
            elif "2: " in part:
                try:
                    snip = part[part.index("2: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
        return query.strip(), snippets

# convert given query, documents, and answer into appropriate format for saving as instruction tuning data
def finalize_instr(instr_style: str,
                  prompt: str,
                  query: str, # pre-stripped when passed in!
                  docs_or_snippets=None # list of source docs
                  ):

    # remove prefix and suffix from instruction generation prompt (LLAMA)
    prompt = prompt.removeprefix(SYS_PREFIX)
    prompt = prompt.removesuffix(SYS_SUFFIX)

    ###########################################
    ######## Obtain Documents & Answers #######
    ###########################################
    if instr_style=="A_0":
        docs = list(docs_or_snippets) # a tuple instead of a string? (due to using old version of script for A_0?)
        prompt_prefix = """You are a search engine. A person queried something in detail and the most relevant snippets about the query are as follows.\nQuery: X\nSnippets: """
        prompt_suffix = """\nWhat could the detailed query X be? Answer with a plausible question or instruction.\nX:"""
        isolated_snippets_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '")

    elif instr_style=="A_1_0":
        # Access documents
        prompt_midfix = "\nYou are a search engine. A person queried something in detail about the documents above and the most relevant snippets about the query are as follows.\nQuery: X\nSnippets: "
        isolated_docs_str, snippets_str = prompt.split(prompt_midfix)
        docs = isolated_docs_str[1:-1].split("', '") # a list; use [1:-1] to remove start and end single-quote mark '

        prompt_suffix = "\n\nWhat could the detailed query X be? Answer with a plausible question or instruction.\nX:"
        isolated_snippets_str = snippets_str.removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '") 

    elif instr_style=="B_0_3":
        docs = list(docs_or_snippets) # a tuple instead of a string? (due to using old version of script for B_0_3?)
        prompt_prefix = """Instruction: X\nSnippets: """
        prompt_suffix = """\n\nWhat kind of instruction could these two snippets be the answer to? You must answer with a specific question that can ONLY be answered by utilizing information in both snippets. You will be penalized if the question concerns only one snippet. Format your answer as plain text. Say "Not sure" if you can't come up with a high-quality question.\nX:"""
        isolated_snippets_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '")

    elif instr_style=="B_1_3":
        isolated_docs_str, snippets_str = prompt.split("\n\nInstruction: X\nSnippets: ")
        docs = isolated_docs_str[1:-1].split("', '")

        prompt_suffix = """\n\nWhat kind of instruction could these two snippets be the answer to? You must answer with a specific question that can ONLY be answered by utilizing information in both snippets. You will be penalized if the question concerns only one snippet. Format your answer as plain text. Say "Not sure" if you can't come up with a high-quality question. \nRead the question again: What kind of instruction could these two snippets be the answer to? You must answer with a specific question that can ONLY be answered by utilizing information in both snippets. You will be penalized if the question concerns only one snippet. Format your answer as plain text. Say "Not sure" if you can't come up with a high-quality question.\nX:"""
        isolated_snippets_str = snippets_str.removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '")

    elif instr_style=="B_1_4":
        isolated_docs_str, snippets_str = prompt.split("\n\nInstruction: X\nSnippets: ")
        docs = isolated_docs_str[1:-1].split("', '")

        prompt_suffix = """\n\nWhat kind of instruction could these two snippets be the answer to? You must answer with a specific question that can ONLY be answered by utilizing information in both snippets. You will be penalized if the question concerns only one snippet. Format your answer as plain text. Say "Not sure" if you can't come up with a high-quality question.\nX:"""
        isolated_snippets_str = snippets_str.removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '")

    elif instr_style=="D_3":
        prompt_prefix = """Below are two documents. Select 3 sentences that are most pertinent to the content of the documents. Then generate a single question or instruction that can ONLY be answered or responded to using ALL 3 sentences.\n"""
        prompt_suffix = """\n\nMake sure EACH snippet is critical to answering the question/instruction. You will be penalized if your proposed question/instruction concerns only one or two snippets. Format your proposal as:\n\nQuestion/Instruction: \nSnippet 1:\nSnippet 2: \nSnippet 3:"""

        isolated_docs_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        docs = isolated_docs_str[1:-1].split("', '")
        snippets = docs_or_snippets

    elif instr_style=="D_4":
        prompt_prefix = """Below are two documents. Select 4 sentences that are most pertinent to the content of the documents. Then generate a single question or instruction that can ONLY be answered or responded to using ALL 4 sentences.\n"""
        prompt_suffix = """\n\nMake sure EACH snippet is critical to answering the question/instruction. You will be penalized if your proposed question/instruction concerns only one or two snippets. Format your proposal as:\n\nQuestion/Instruction: \nSnippet 1:\nSnippet 2: \nSnippet 3:\nSnippet 4:"""
        isolated_docs_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        docs = isolated_docs_str[1:-1].split("', '")
        snippets = docs_or_snippets

    elif instr_style=="E_4":
        prompt_prefix = """The documents below are ordered by relevance to a given query, with the first document being most relevant.\n\n"""
        prompt_suffix = """\n\nGiven that the documents are ordered from most to least useful to answering the query, what could be the query X?\nX:"""

        isolated_docs_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        docs = isolated_docs_str[1:-1].split("\'\n\n\'")
        docs_with_ids = [(idx+1, docs[idx]) for idx in range(len(docs))]

    elif instr_style=="E_6":
        prompt_suffix = """\n\nSelect two sentences from the above documents. Generate a query to either compare or contrast the information identified. Format your answer as:\n\nSentence 1:\nSentence 2:\nQuery:"""
        isolated_docs_str = prompt.removesuffix(prompt_suffix)
        docs = isolated_docs_str[1:-1].split("\'\n\n\'")
        snippets = docs_or_snippets

    if instr_style!="E_4":
        finalized_instr = "'"
        finalized_instr += docs[0]
        finalized_instr += "'\n\n'"
        finalized_instr += docs[1]
        finalized_instr += "'\n\n"
        finalized_instr += query # revisit: any prompt needed or nah?

        finalized_answer = " ".join(snippets)


    elif instr_style=="E_4":
        finalized_instr = "Query: "+query 
        shuffle(docs_with_ids) #(1, x) (2, y) (3, z)->1 (2, y) 2 (3, z) 3 (1, x)
        id_pairs = []
        for idx, (doc_id, doc) in enumerate(docs_with_ids):
            finalized_instr += "\n\n"
            finalized_instr += str(idx+1)
            finalized_instr += ": '"
            finalized_instr += doc 
            finalized_instr += "'"
            id_pairs.append((idx+1, doc_id))
        finalized_instr += "\n\nEach document above is identified by an ID number. Order the document IDs according to relevance to the given query above, such that the first ID corresponds to the most relevant document to the query. Your answer should be an ordered list of ID numbers."
        # get answers
        id_pairs.sort(key=lambda x:x[1]) 
        finalized_answer = ""
        for pair in id_pairs:
            finalized_answer += str(pair[0])
            finalized_answer += ", "
        finalized_answer = finalized_answer[:-2]
    
    return finalized_instr, finalized_answer


# not changed for processing multiple thresholds at one since no scoring needed  
def process_E2_E5(args):

    # output dir not needed since no scoring but use to standardize data format
    current_date_time = str(datetime.now().strftime("%Y-%m-%d_hr%H-min%M"))
    scored_by = "scorer_"+args.model_name
    output_dir = os.path.join(args.input_dir, "data_jsons", current_date_time, scored_by)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # access instruction generation prompts and outputs (i.e., the generated instructions)
    prompt_dir = os.path.join(args.input_dir,"prompts") # instructions
    num_snips_dir = os.path.join(args.input_dir,"num_snips") # target answers
    cluster_ids_dir = os.path.join(args.input_dir,"cluster_ids") # cluster ids

    # sort alphanumerically so that names match in order of reference 
    all_prompt_files_train = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_prompt_files_valid = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_prompt_files_test = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_prompt_files = [all_prompt_files_train, all_prompt_files_valid, all_prompt_files_test]

    all_ans_files_train = sorted(sorted([
                f
                for f in os.listdir(num_snips_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_ans_files_valid = sorted(sorted([
                f
                for f in os.listdir(num_snips_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_ans_files_test = sorted(sorted([
                f
                for f in os.listdir(num_snips_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_ans_files = [all_ans_files_train, all_ans_files_valid,all_ans_files_test]

    all_cluster_id_files_train = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_cluster_id_files_valid = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_cluster_id_files_test = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_cluster_id_files = [all_cluster_id_files_train, all_cluster_id_files_valid,all_cluster_id_files_test]

    # iterate over train/val/test splits
    for split_idx, (prompt_split_files, ans_split_files, cluster_id_files) in enumerate(tqdm(zip(all_prompt_files, all_ans_files, all_cluster_id_files))):

        # Create json file to save data for current split
        one_prompt_pt = prompt_split_files[0] # e.g., train.1.pt
        split_json_file_name = one_prompt_pt[:one_prompt_pt.index(".")] + ".json" # e.g., train.1.pt -> train.json
        data_json_path = os.path.join(output_dir, split_json_file_name)

        with open(data_json_path, "a") as json_file:
            print("########################")
            print("Working on",split_json_file_name)

            # score all the instructons for current split
            for file_idx, (prompt_pt, ans_pt, cluster_id_pt) in enumerate(tqdm(zip(prompt_split_files, ans_split_files, cluster_id_files))):

                prompt_slice = torch.load(os.path.join(prompt_dir, prompt_pt))
                ans_slice = torch.load(os.path.join(num_snips_dir, ans_pt))
                cluster_id_slice = torch.load(os.path.join(cluster_ids_dir, cluster_id_pt))

                for example_idx, (prompt, answer, cluster_id) in enumerate(zip(prompt_slice, ans_slice, cluster_id_slice)):
                    finalized_instr = prompt

                    # Save data to json for HF dataset creation
                    data = {"instruction": finalized_instr, "answer": answer, "cluster_id": cluster_id, "score": -1}
                    json.dump(data, json_file)
                    json_file.write('\n')

                print("Finished writing selected datapoints part %d to %s"%(file_idx, data_json_path))
        print("Finished saving selected instructions for %s!"%(split_json_file_name))

# same as above but for templates S1 & R1
def process_S1_R1(args):

    # output dir not needed since no scoring but use to standardize data format
    current_date_time = str(datetime.now().strftime("%Y-%m-%d_hr%H-min%M"))
    scored_by = "scorer_"+args.model_name
    output_dir = os.path.join(args.input_dir, "data_jsons", current_date_time, scored_by)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # access instruction generation prompts and outputs (i.e., the generated instructions)
    prompt_dir = os.path.join(args.input_dir,"prompts") # instructions
    ans_dir = os.path.join(args.input_dir,"answers") # target answers
    cluster_ids_dir = os.path.join(args.input_dir,"cluster_ids") # cluster ids

    # sort alphanumerically so that names match in order of reference 
    all_prompt_files_train = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_prompt_files_valid = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_prompt_files_test = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_prompt_files = [all_prompt_files_train, all_prompt_files_valid, all_prompt_files_test]

    all_ans_files_train = sorted(sorted([
                f
                for f in os.listdir(ans_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_ans_files_valid = sorted(sorted([
                f
                for f in os.listdir(ans_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_ans_files_test = sorted(sorted([
                f
                for f in os.listdir(ans_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_ans_files = [all_ans_files_train, all_ans_files_valid,all_ans_files_test]

    all_cluster_id_files_train = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_cluster_id_files_valid = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_cluster_id_files_test = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_cluster_id_files = [all_cluster_id_files_train, all_cluster_id_files_valid,all_cluster_id_files_test]

    # iterate over train/val/test splits
    for split_idx, (prompt_split_files, ans_split_files, cluster_id_files) in enumerate(tqdm(zip(all_prompt_files, all_ans_files, all_cluster_id_files))):

        # Create json file to save data for current split
        one_prompt_pt = prompt_split_files[0] # e.g., train.1.pt
        split_json_file_name = one_prompt_pt[:one_prompt_pt.index(".")] + ".json" # e.g., train.1.pt -> train.json
        data_json_path = os.path.join(output_dir, split_json_file_name)

        with open(data_json_path, "a") as json_file:
            print("########################")
            print("Working on",split_json_file_name)

            for file_idx, (prompt_pt, ans_pt, cluster_id_pt) in enumerate(tqdm(zip(prompt_split_files, ans_split_files, cluster_id_files))):

                prompt_slice = torch.load(os.path.join(prompt_dir, prompt_pt))
                ans_slice = torch.load(os.path.join(ans_dir, ans_pt))
                cluster_id_slice = torch.load(os.path.join(cluster_ids_dir, cluster_id_pt))

                for example_idx, (prompt, answer, cluster_id) in enumerate(zip(prompt_slice, ans_slice, cluster_id_slice)):

                    # Save data to json for HF dataset creation
                    data = {"instruction": prompt, "answer": answer, "cluster_id": cluster_id, "score": -1}
                    json.dump(data, json_file)
                    json_file.write('\n')

                print("Finished writing selected datapoints part %d to %s"%(file_idx, data_json_path))

        print("Finished saving selected instructions for %s!"%(split_json_file_name))
        

def main(args):

    # ensure output dir exists
    # enhancement_descrip = get_enhancement_descrip(args)
    if args.given_date_time==None:
        current_date_time = str(datetime.now().strftime("%Y-%m-%d_hr%H-min%M"))
    else: 
        current_date_time = args.given_date_time
    json_dir = os.path.join(args.input_dir, "data_jsons", current_date_time)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    scored_by = "scorer_"+args.model_name
    threshold_descrip_3 = "thresh_3"
    threshold_descrip_4 = "thresh_4"
    threshold_descrip_5 = "thresh_5" # however threshold 4.5 per paper spec
    output_dir_3 = os.path.join(json_dir,scored_by,f"cur_prompt_{args.curation_prompt}", threshold_descrip_3)
    output_dir_4 = os.path.join(json_dir,scored_by,f"cur_prompt_{args.curation_prompt}",threshold_descrip_4)
    output_dir_5 = os.path.join(json_dir,scored_by,f"cur_prompt_{args.curation_prompt}",threshold_descrip_5)
    if not os.path.exists(output_dir_3):
        os.makedirs(output_dir_3)
    if not os.path.exists(output_dir_4):
        os.makedirs(output_dir_4)
    if not os.path.exists(output_dir_5):
        os.makedirs(output_dir_5)

    # load selector model
    if args.model_name=="llama2-chat-7b":
        selector_model = LLM(
            model="meta-llama/Llama-2-7b-chat-hf", 
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            # tensor_parallel_size=2, # not used in lambda2 run!!!
            ) # vllm
        
    elif args.model_name=="llama2-chat-13b":
        selector_model = LLM(
            model="meta-llama/Llama-2-13b-chat-hf", 
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            tensor_parallel_size=2,
            ) # vllm

    elif args.model_name=="chatglm2-6b":
        selector_model = LLM(
            model="THUDM/chatglm2-6b", 
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=2,
            ) # vllm

    elif args.model_name=="chatglm3-6b":
        selector_model = LLM(
            model="THUDM/chatglm3-6b", 
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            ) 

    # access instruction generation prompts and outputs (i.e., the generated instructions)
    prompt_dir = os.path.join(args.input_dir,"prompts")
    instr_dir = os.path.join(args.input_dir,"instructions")
    cluster_ids_dir = os.path.join(args.input_dir,"cluster_ids") # cluster ids

    # sort alphanumerically so that names match in order of reference 
    all_prompt_files_train = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_prompt_files_valid = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_prompt_files_test = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_prompt_files = [all_prompt_files_train, all_prompt_files_valid, all_prompt_files_test]

    all_instr_files_train = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_instr_files_valid = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_instr_files_test = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_instr_files = [all_instr_files_train, all_instr_files_valid, all_instr_files_test]

    all_cluster_id_files_train = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_cluster_id_files_valid = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_cluster_id_files_test = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_cluster_id_files = [all_cluster_id_files_train, all_cluster_id_files_valid,all_cluster_id_files_test]

    # iterate over train/val/test splits
    for split_idx, (prompt_split_files, instr_split_files, cluster_id_files) in enumerate(tqdm(zip(all_prompt_files, all_instr_files, all_cluster_id_files))):

        if instr_split_files==[] or split_idx<args.start_split_idx:
            continue

        # Create json file to save data for current split
        one_prompt_pt = instr_split_files[0] # e.g., train.1.pt ## TODO: CHANGE BACK TO PROMPT_SPLIT_FILES
        print("FILENAMES:",prompt_split_files, instr_split_files, cluster_id_files)
        split_json_file_name = one_prompt_pt[:one_prompt_pt.index(".")] + ".json" # e.g., train.1.pt -> train.json
        data_json_path_3 = os.path.join(output_dir_3, split_json_file_name)
        data_json_path_4 = os.path.join(output_dir_4, split_json_file_name)
        data_json_path_5 = os.path.join(output_dir_5, split_json_file_name)

        print("########################")
        print("Working on",split_json_file_name)
        
        # score all the instructons for current split
        for file_idx, instr_pt in enumerate(tqdm(instr_split_files)):
            
            # write current filename to log (in case need to pause/resume)
            f = open(os.path.join(json_dir,scored_by,"num_slices_scored_so_far.txt"), "w")
            f.write("\nOn instruction/prompt file: "+str(instr_pt))
            f.write("\nCorresponding file_idx: "+str(file_idx))
            f.close()
            
            if args.start_file_idx <= file_idx < len(instr_split_files): ## TODO: change len(...) to desired number N if want to only process N instruction/prompt files

                prompt_slice = torch.load(os.path.join(prompt_dir, instr_pt))
                instr_slice = torch.load(os.path.join(instr_dir, instr_pt))
                cluster_id_slice = torch.load(os.path.join(cluster_ids_dir, instr_pt))

                if args.instruction_format=="A_0" or args.instruction_format=="B_0_3":
                    doc_dir = os.path.join(args.input_dir,"source_docs")
                    source_doc_slice = torch.load(os.path.join(doc_dir, instr_pt))

                for example_idx, (prompt, instr, cluster_id) in enumerate(zip(prompt_slice, instr_slice, cluster_id_slice)):
                    
                    if args.instruction_format!="D_3" and args.instruction_format!="D_4" and args.instruction_format!="E_6":
                        query = get_query(args.instruction_format, instr.strip())
                    else:
                        query, snippets = get_query(args.instruction_format, instr.strip())

                    # prepare scoring prompt
                    if query != "": # only if the query is NONempty, i.e. if a query was actually generated
                        if args.curation_prompt==1:
                            curation_input = curation_prompt1(prompt, query)
                        elif args.curation_prompt==2:
                            curation_input = curation_prompt2(prompt, query)
                        elif args.curation_prompt==3:
                            curation_input = curation_prompt3(prompt, query)

                        # obtain scoring output from selector model
                        response = selector_model.generate([curation_input], SAMPLING_PARAMS)[0].outputs[0].text # vllm

                        # extract score
                        score_matched = REGEX.search(response)
                        score = int(score_matched.group(1)) if score_matched else None

                        # select instruction if score passes threshold
                        if score and score >= 3: 
                            if args.instruction_format=="A_0" or args.instruction_format=="B_0_3":
                                finalized_instr, finalized_answer = finalize_instr(args.instruction_format, prompt, query, source_doc_slice[example_idx]) 
                            elif args.instruction_format=="D_3" or args.instruction_format=="D_4" or args.instruction_format=="E_6":
                                finalized_instr, finalized_answer = finalize_instr(args.instruction_format, prompt, query, snippets)
                            else: 
                                finalized_instr, finalized_answer = finalize_instr(args.instruction_format, prompt, query)

                            # Save data to json for HF dataset creation
                            if finalized_answer != "":
                                data = {"instruction": finalized_instr, "answer": finalized_answer, "cluster_id": cluster_id, "score": score}
                                with open(data_json_path_3, "a") as json_file:
                                    json.dump(data, json_file)
                                    json_file.write('\n')

                                # since score is at least 3, check if it's also at least 4
                                if score >= 4:
                                    with open(data_json_path_4, "a") as json_file:
                                        json.dump(data, json_file)
                                        json_file.write('\n')
                                    
                                    # since score at least 4, check if it's equal to 5
                                    if score >= 4.5: # use 4.5 per instruction backtranslation paper spec
                                        with open(data_json_path_5, "a") as json_file:
                                            json.dump(data, json_file)
                                            json_file.write('\n')

                print("Finished writing selected datapoints part %d to %s"%(file_idx, data_json_path_3)) # TODO: indent BACKWARD once if remove the "if args.start_file_idx <= file_idx < len(instr_split_files)"

        print("Finished saving selected instructions for %s!"%(split_json_file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of path to appropriate instructions/prompts folders from ./generated_instructions folder
    parser.add_argument("--input_dir", type=str) 
    parser.add_argument("--given_date_time", type=str, default=None) 
    parser.add_argument("--start_file_idx", type=int, default=0) 
    parser.add_argument("--start_split_idx", type=int, default=0) # 0=train, 1=val, 2=test
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

    # selector model choices
    parser.add_argument(
        "--model_name",
        choices=[
            "llama2-chat-7b",
            "llama2-chat-13b",
            # "llama2-7b",
            "chatglm2-6b",
            "chatglm3-6b",
        ],
        default="llama2-chat-7b",
        type=str,
    )

    parser.add_argument("--curation_prompt", type=int, default=1) 

    args = parser.parse_args()

    if args.instruction_format=="E_2" or args.instruction_format=="E_5":
        process_E2_E5(args)

    elif args.instruction_format=="S_1" or args.instruction_format=="R_1":
        process_S1_R1(args)

    else:
        main(args)