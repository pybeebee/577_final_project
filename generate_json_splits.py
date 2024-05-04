### FILE TO CONVERT CURATED DATA TO .JSON FILES FOR INPUT TO HUGGINGFACE DATASET LOADER
import argparse
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import re
import random
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import json
import sys
import numpy as np

LENGTH_ENHANCEMENTS = ["\nRespond in 2 sentences.", "\nRespond briefly."]
SPLIT_JSON_NAMES = ["train.json", "valid.json", "test.json"]

# get string representation of enhancement status for use in output directory path
def get_enhancement_descrip(args):
    descrip = "enhance_"
    if args.use_enhancement_1:
        descrip += "1"
    if args.use_enhancement_2:
        descrip += "2"
    if args.use_enhancement_3:
        descrip += "3"
    if args.use_enhancement_4:
        descrip += "4"
    if args.use_enhancement_5:
        descrip += "5"
    if args.use_enhancement_6:
        descrip += "6"
    if args.use_enhancement_7:
        descrip += "7"
    if args.use_enhancement_8:
        descrip += "8"
    if args.use_enhancement_9:
        descrip += "9"
    if not args.use_enhancement_1 and not args.use_enhancement_2 and not args.use_enhancement_3 and not args.use_enhancement_4 and not args.use_enhancement_5 and not args.use_enhancement_6 and not args.use_enhancement_7 and not args.use_enhancement_8 and not args.use_enhancement_9:
        descrip += "0"
    return descrip
  
# for any rows with the same string in "answer", keep only the one with the highest "score" entry. if all have the same score then just keep the first one and discard the others from the df.
def keep_highest_score(group):
    max_score_indices = group['score'].index[group['score'] == group['score'].max()]
    random_index = np.random.choice(max_score_indices)
    return group.loc[random_index]

# Summarization pipeline for turning extractive answers to abstractive
summarizer_model = pipeline("summarization", model="slauw87/bart_summarisation", device_map="auto",)
ABS_COUNT = 0
FAIL_COUNT = 0
def abstractify_str(penultimate_answer: str): # applies to string input
    try: 
        summarized_answer = summarizer_model(penultimate_answer)
        global ABS_COUNT
        ABS_COUNT+=1
        print(ABS_COUNT)
        return summarized_answer[0]['summary_text']
    except: 
        global FAIL_COUNT
        FAIL_COUNT+=1
        print("FAIL:",FAIL_COUNT)
        return penultimate_answer

def abstractify_df(df: pd.DataFrame, split_json, output_dir, data_type): # applies at dataframe level
    global ABS_COUNT
    global FAIL_COUNT
    ABS_COUNT = 0
    FAIL_COUNT = 0
    random_rows = df.sample(frac=0.5, random_state=0)
    random_rows['answer'] = random_rows['answer'].apply(abstractify_str)
    df.update(random_rows)
    
    if split_json=="train.json":
        file_name = "train.txt"
    elif split_json=="valid.json":
        file_name = "valid.txt"
    elif split_json=="test.json":
        file_name = "test.txt"
    f = open(os.path.join(output_dir, file_name), "a")
    f.write("ABS/FAIL COUNTS FOR %s: %d, %d\n"%(data_type, ABS_COUNT, FAIL_COUNT))
    print("ABS/FAIL COUNTS:", ABS_COUNT, FAIL_COUNT)
    return df

def main(args):
    # turn response for certain instruction types into sentence as opposed to integer answer
    def sentenceify(num_snips):
        if type(num_snips)==int:
            return f"There are {num_snips} misplaced sentences."
        return num_snips
    
    # function to add enhancement to a given instruction
    def enhance(instr):
        addition = ""

        if args.use_enhancement_1:
            if random.random() > 0.5: # 50% of the time
                instr += random.choice(LENGTH_ENHANCEMENTS)
        elif args.use_enhancement_3:
            if random.random() > 0.5:
                addition = "\nYou will need to look for the answer across multiple locations in the input. Do not ignore the middle of the input context." # Do not prioritize only the beginning and end.
        elif args.use_enhancement_4: 
            if random.random() > 0.5:
                addition = "\nTo come up with your response, first read once through all the documents in detail. Then, do a second pass to answer the query."
        elif args.use_enhancement_6: 
            if random.random() > 0.5:
                addition = "\nSkim the text before providing a response."
        elif args.use_enhancement_7: 
            if random.random() > 0.5:
                addition = "\nAs you come up with your answer, carefully consider whether each piece of information in the source documents is useful to generating a successful answer."
        elif args.use_enhancement_8: 
            if random.random() > 0.5:
                addition = "\nWhen coming up with your answer, did you consider information across all positions in the input? If not, do so to revise your answer before responding."
        elif args.use_enhancement_9: 
            if random.random() > 0.5:
                addition = "\nBefore providing your response, read through the provided documents again. Is there anything you would change about your response? If so, make this change before giving your final answer."
        return instr+addition

    # function to deduplicate examples in the dataframe and select a specified number of samples
    def process_df(data_df, num_to_choose, data_type):
        # deduplicate ones with same answer (keep only first/highest scoring such instruction)
        if data_type != "E4":
            deduped_df = data_df.groupby('answer').apply(keep_highest_score)
            deduped_df = deduped_df.reset_index(drop=True) # reset index col to default
        else: 
            deduped_df = data_df

        # filter for specific number of clusters, if specified
        if args.num_clusters != -1:
            subset_cluster_ids = sorted(deduped_df['cluster_id'].unique())[:args.num_clusters]
            cluster_filtered_df = deduped_df[deduped_df['cluster_id'].isin(subset_cluster_ids)]
        elif args.prop_clusters != 1: # ASSUME ONLY ONE OF NUM_CLUSTERS AND PROP_CLUSTERS IS CHANGED FROM DEFAULT
            num_clusters_to_use = deduped_df['cluster_id'].nunique() // 2
            subset_cluster_ids = sorted(deduped_df['cluster_id'].unique())[:num_clusters_to_use]
            cluster_filtered_df = deduped_df[deduped_df['cluster_id'].isin(subset_cluster_ids)]
        else:
            cluster_filtered_df = deduped_df

        # randomly select num_to_choose examples from the df
        filtered_df = cluster_filtered_df.sample(n=min(num_to_choose,cluster_filtered_df.shape[0]), random_state=0)
        
        # apply enhancement function to each instruction
        if args.use_enhancement_1:
            if data_type in {"ABE6"}:
                filtered_df['instruction'] = filtered_df['instruction'].apply(enhance)
        if args.use_enhancement_3:
            if data_type in {"ABE6", "D3", "D4"}:
                filtered_df['instruction'] = filtered_df['instruction'].apply(enhance)
        if args.use_enhancement_4 or args.use_enhancement_6 or args.use_enhancement_7 or args.use_enhancement_8 or args.use_enhancement_9:
            filtered_df['instruction'] = filtered_df['instruction'].apply(enhance)

        return filtered_df
    
    # same as above but processing specific to E2 and E5
    def process_df_E2_E5_S1_R1(data_df, num_to_choose):
        # deduplicate ones with same instruction (keep only first/highest scoring such instruction)
        deduped_df = data_df.groupby('instruction').apply(keep_highest_score)
        deduped_df = deduped_df.reset_index(drop=True) # reset index col to default

        # filter for specific number of clusters, if specified
        if args.num_clusters != -1:
            subset_cluster_ids = sorted(deduped_df['cluster_id'].unique())[:args.num_clusters]
            cluster_filtered_df = deduped_df[deduped_df['cluster_id'].isin(subset_cluster_ids)]
        elif args.prop_clusters != 1: # ASSUME ONLY ONE OF NUM_CLUSTERS AND PROP_CLUSTERS IS CHANGED FROM DEFAULT
            num_clusters_to_use = deduped_df['cluster_id'].nunique() // 2
            subset_cluster_ids = sorted(deduped_df['cluster_id'].unique())[:num_clusters_to_use]
            cluster_filtered_df = deduped_df[deduped_df['cluster_id'].isin(subset_cluster_ids)]
        else:
            cluster_filtered_df = deduped_df
        
        # select num_to_choose of them randomly
        filtered_df = cluster_filtered_df.sample(n=min(num_to_choose, cluster_filtered_df.shape[0]), random_state=0)

        # apply enhancement function to each instruction
        if not args.use_enhancement_3 and not args.use_enhancement_1:
            filtered_df['instruction'] = filtered_df['instruction'].apply(enhance)

        return filtered_df
    
    # create output directory
    enhancement_descrip = get_enhancement_descrip(args)
    if args.num_clusters != -1:
        cluster_prop_descrip = f"num_clusters_{args.num_clusters}"
    elif args.prop_clusters != 1: 
        cluster_prop_descrip = f"prop_clusters_{args.prop_clusters}"
    else:
        cluster_prop_descrip = ""
    if cluster_prop_descrip == "":
        output_dir = os.path.join(args.data_splits_dir, "scorer_"+args.curation_model_id, args.instr_type_proportions_id, "thresh_"+str(args.instr_thresh_num), enhancement_descrip, str(args.total_instr_num))
    else:
        output_dir = os.path.join(args.data_splits_dir, "scorer_"+args.curation_model_id, args.instr_type_proportions_id, "thresh_"+str(args.instr_thresh_num), enhancement_descrip, str(args.total_instr_num), cluster_prop_descrip)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write args to file for easy reference later on
    try:
        from shlex import quote as cmd_quote
    except ImportError:
        from pipes import quote as cmd_quote
    cmdline = " ".join(map(cmd_quote, sys.argv[1:]))
    args_file_path = os.path.join(output_dir, "args.txt")
    f = open(args_file_path, "a")
    f.write(cmdline)
    f.close()

    # address one data split at a time (train/test/val)
    for split_json in SPLIT_JSON_NAMES:

        print("##############################")
        print("Working on:",split_json)
        dfs_to_concatenate = []

        if split_json=="train.json":
            file_name = "train.txt"
            if args.skip_train:
                continue
        elif split_json=="valid.json":
            file_name = "valid.txt"
            if args.skip_valid:
                continue
        elif split_json=="test.json":
            file_name = "test.txt"
            if args.skip_test:
                continue

        if args.use_ABE6 != 0:
            num_to_choose = args.total_instr_num * args.use_ABE6

            if split_json != "train.json":
                num_to_choose *= 0.10 # val and train set size 10% of training data
            num_to_choose = round(num_to_choose)

            # read all of the json files of the form specified at file call
            A_0_df = None
            if args.A_0_json_path!="": # path to train.json, valid.json, test.json for A_0 (take from output of self_curation_revised2.py)
                A_0_json = os.path.join(args.A_0_json_path, "train.json")
                A_0_df_full = pd.read_json(A_0_json, lines=True)
                num_rows = A_0_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    A_0_df = A_0_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    A_0_df = A_0_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    A_0_df = A_0_df_full.iloc[train_rows+test_rows:]

            A_1_0_df = None
            if args.A_1_0_json_path!="":
                A_1_0_json = os.path.join(args.A_1_0_json_path, "train.json")
                A_1_0_df_full = pd.read_json(A_1_0_json, lines=True)
                num_rows = A_1_0_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    A_1_0_df = A_1_0_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    A_1_0_df = A_1_0_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    A_1_0_df = A_1_0_df_full.iloc[train_rows+test_rows:]

            B_1_3_df = None
            if args.B_1_3_json_path!="":
                B_1_3_json = os.path.join(args.B_1_3_json_path, "train.json")
                B_1_3_df_full = pd.read_json(B_1_3_json, lines=True)
                num_rows = B_1_3_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    B_1_3_df = B_1_3_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    B_1_3_df = B_1_3_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    B_1_3_df = B_1_3_df_full.iloc[train_rows+test_rows:]

            B_1_4_df = None
            if args.B_1_4_json_path!="":
                B_1_4_json = os.path.join(args.B_1_4_json_path, "train.json")
                B_1_4_df_full = pd.read_json(B_1_4_json, lines=True)
                num_rows = B_1_4_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    B_1_4_df = B_1_4_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    B_1_4_df = B_1_4_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    B_1_4_df = B_1_4_df_full.iloc[train_rows+test_rows:]

            E_6_df = None
            if args.E_6_json_path!="":
                E_6_json = os.path.join(args.E_6_json_path, "train.json")
                E_6_df_full = pd.read_json(E_6_json, lines=True)
                num_rows = E_6_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    E_6_df = E_6_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    E_6_df = E_6_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    E_6_df = E_6_df_full.iloc[train_rows+test_rows:]

            data_df = pd.concat([A_0_df, A_1_0_df, B_1_3_df, B_1_4_df, E_6_df], ignore_index=True) # B_0_3_df

            processed_df = abstractify_df(process_df(data_df, num_to_choose, "ABE6"), split_json, output_dir, "ABE6")
            processed_df['instr_type'] = "ABE6"
            dfs_to_concatenate.append(processed_df)
            instr_num_ABE6 = processed_df.shape[0]
            print("Total instruction num (use_ABE6): %d"%(instr_num_ABE6))
            f = open(os.path.join(output_dir, file_name), "a")
            f.write("\nTotal instruction num (use_ABE6): %d"%(instr_num_ABE6))

        if args.use_D3 != 0:
            num_to_choose = args.total_instr_num * args.use_D3

            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)

            # read all of the json files
            D_3_json = os.path.join(args.D_3_json_path, "train.json")
            D_3_df_full = pd.read_json(D_3_json, lines=True)
            num_rows = D_3_df_full.shape[0]
            test_rows = round(0.2*num_rows)
            train_rows = round(0.6*num_rows)
            if split_json=="train.json":
                D_3_df = D_3_df_full.iloc[:train_rows]
            elif split_json=="valid.json":
                D_3_df = D_3_df_full.iloc[train_rows:train_rows+test_rows]
            elif split_json=="test.json":
                D_3_df = D_3_df_full.iloc[train_rows+test_rows:]

            processed_df = abstractify_df(process_df(D_3_df, num_to_choose, "D3"), split_json, output_dir, "D3")
            processed_df['instr_type'] = "D_3"
            dfs_to_concatenate.append(processed_df)
            instr_num_D3 = processed_df.shape[0]
            print("Total instruction num (use_D3): %d"%(instr_num_D3))
            f = open(os.path.join(output_dir, file_name), "a")
            f.write("\nTotal instruction num (use_D3): %d"%(instr_num_D3))

        if args.use_D4 != 0:
            num_to_choose = args.total_instr_num * args.use_D4

            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)

            # read all of the json files
            D_4_json = os.path.join(args.D_4_json_path, "train.json")
            D_4_df_full = pd.read_json(D_4_json, lines=True)
            num_rows = D_4_df_full.shape[0]
            test_rows = round(0.2*num_rows)
            train_rows = round(0.6*num_rows)
            if split_json=="train.json":
                D_4_df = D_4_df_full.iloc[:train_rows]
            elif split_json=="valid.json":
                D_4_df = D_4_df_full.iloc[train_rows:train_rows+test_rows]
            elif split_json=="test.json":
                D_4_df = D_4_df_full.iloc[train_rows+test_rows:]

            processed_df = abstractify_df(process_df(D_4_df, num_to_choose, "D4"), split_json, output_dir, "D4")
            processed_df['instr_type'] = "D_4"
            dfs_to_concatenate.append(processed_df)
            instr_num_D4 = processed_df.shape[0]
            print("Total instruction num (use_D4): %d"%(instr_num_D4))
            f = open(os.path.join(output_dir, file_name), "a")
            f.write("\nTotal instruction num (use_D4): %d"%(instr_num_D4))

        if args.use_E4 != 0:
            num_to_choose = args.total_instr_num * args.use_E4

            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)
            E_4_json_path = args.E_4_json_path
            E_4_json = os.path.join(E_4_json_path, split_json)
            if split_json=="valid.json":
                E_4_json = os.path.join(E_4_json_path, "train.json")
            E_4_df_full = pd.read_json(E_4_json, lines=True)
            num_rows = E_4_df_full.shape[0]
            train_rows = round(0.8*num_rows)
            
            if split_json=="train.json":
                E_4_df = E_4_df_full.iloc[:train_rows]
            elif split_json=="valid.json":
                E_4_df = E_4_df_full.iloc[train_rows:]
            elif split_json=="test.json":
                E_4_df = E_4_df_full

            processed_df = process_df(E_4_df, num_to_choose, "E4")
            processed_df['instr_type'] = "E_4"
            dfs_to_concatenate.append(processed_df)
            instr_num_E4 = processed_df.shape[0]
            print("Total instruction num (use_E4): %d"%(instr_num_E4))
            f = open(os.path.join(output_dir, file_name), "a")
            f.write("\nTotal instruction num (use_E4): %d"%(instr_num_E4))

        if args.use_E2 != 0:
            num_to_choose = args.total_instr_num * args.use_E2
            
            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)
            
            E_2_json_path = args.E_2_json_path
            E_2_json = os.path.join(E_2_json_path, split_json)
            E_2_df = pd.read_json(E_2_json, lines=True)

            if args.E2_sentence_form_answer:
                E_2_df['answer'] = E_2_df['answer'].apply(sentenceify)
            
            processed_df = process_df_E2_E5_S1_R1(E_2_df, num_to_choose)
            processed_df['instr_type'] = "E_2"
            dfs_to_concatenate.append(processed_df)
            instr_num_E2 = processed_df.shape[0]
            short_count = processed_df['answer'].apply(lambda x: isinstance(x, int)).sum() # how many E2 examples have a short integer answer?
            long_count = processed_df['answer'].apply(lambda x: isinstance(x, str)).sum() # how many E2 examples have a longer string answer?
            print("Total instruction num (use_E2): %d"%(instr_num_E2))
            print("Total instruction num (E2_short): %d"%(short_count))
            print("Total instruction num (E2_long): %d"%(long_count))
            f = open(os.path.join(output_dir, file_name), "a")
            f.write("\nTotal instruction num (use_E2): %d"%(instr_num_E2))
            f.write("\nTotal instruction num (E2_short): %d"%(short_count))
            f.write("\nTotal instruction num (E2_long): %d"%(long_count))

        if args.use_E5 != 0:
            num_to_choose = args.total_instr_num * args.use_E5

            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)

            E_5_json_path = args.E_5_json_path
            E_5_json = os.path.join(E_5_json_path, split_json)
            E_5_df = pd.read_json(E_5_json, lines=True)

            processed_df = process_df_E2_E5_S1_R1(E_5_df, num_to_choose)
            processed_df['instr_type'] = "E_5"
            dfs_to_concatenate.append(processed_df)
            instr_num_E5 = processed_df.shape[0]
            print("Total instruction num (use_E5): %d"%(instr_num_E5))
            f = open(os.path.join(output_dir, file_name), "a")
            f.write("\nTotal instruction num (use_E5): %d"%(instr_num_E5))

        if args.use_S1 != 0:
            num_to_choose = args.total_instr_num * args.use_S1

            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)

            S_1_json_path = args.S_1_json_path
            S_1_json = os.path.join(S_1_json_path, split_json)
            S_1_df = pd.read_json(S_1_json, lines=True)

            processed_df = process_df_E2_E5_S1_R1(S_1_df, num_to_choose)
            processed_df['instr_type'] = "S_1"
            dfs_to_concatenate.append(processed_df)
            instr_num_S1 = processed_df.shape[0]
            print("Total instruction num (use_S1): %d"%(instr_num_S1))
            f = open(os.path.join(output_dir, file_name), "a")
            f.write("\nTotal instruction num (use_S1): %d"%(instr_num_S1))

        if args.use_R1 != 0:
            num_to_choose = args.total_instr_num * args.use_R1

            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)

            R_1_json_path = args.R_1_json_path
            R_1_json = os.path.join(R_1_json_path, split_json)
            R_1_df = pd.read_json(R_1_json, lines=True)

            processed_df = process_df_E2_E5_S1_R1(R_1_df, num_to_choose)
            processed_df['instr_type'] = "R_1"
            dfs_to_concatenate.append(processed_df)
            instr_num_R1 = processed_df.shape[0]
            print("Total instruction num (use_R1): %d"%(instr_num_R1))
            f = open(os.path.join(output_dir, file_name), "a")
            f.write("\nTotal instruction num (use_R1): %d"%(instr_num_R1))


        split_data_df = pd.concat(dfs_to_concatenate, ignore_index=True)
        split_data_file_path = os.path.join(output_dir, split_json)
        split_data_df.to_json(split_data_file_path, orient='records', lines=True)
        print("################################")
        print("Finished saving instructions for %s to %s"%(split_json, split_data_file_path))
        print("Total number of instructions (%s):"%(split_json), split_data_df.shape[0])
        
        f = open(os.path.join(output_dir, file_name), "a")
        f.write("\nTotal number of instructions (%s): %d"%(split_json, split_data_df.shape[0]))
        f.write("\nTotal number of clusters represented (%s): %d"%(split_json, split_data_df['cluster_id'].nunique()))
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_splits_dir", 
        type=str,
        default="./data_splits_mistral"
    ) #e.g. "./data_splits"
    parser.add_argument(
        "--total_instr_num",
        type=int,
        default=25000,
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--prop_clusters", # ASSUME ONLY ONE OF NUM_CLUSTERS AND PROP_CLUSTERS IS CHANGED FROM DEFAULT
        type=float,
        default=1,
    )
    parser.add_argument(
        "--instr_type_proportions_id", # id describing the types of instructions, their relative proportions, and selection thresholds; keep tabs of this in personal notes file for reference.
        type=str,
        default="",
    )
    parser.add_argument(
        "--instr_thresh_num", # id describing the threshold used to select instructions
        type=int,
        default=3,
    )
    parser.add_argument(
        "--curation_model_id", # id describing what model was used to score and filter these instructions being inputted
        type=str,
    )
    
    parser.add_argument(
        "--use_ABE6",
        type=float, # e.g., 0.33 = 33% of data is of this instruction type
        default=0,
    )
    parser.add_argument(
        "--use_D3",
        type=float, # e.g., 0.33 = 33% of data is of this instruction type
        default=0,
    )
    parser.add_argument(
        "--use_D4",
        type=float, # e.g., 0.33 = 33% of data is of this instruction type
        default=0,
    )
    parser.add_argument(
        "--use_E2",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--use_E5",
        type=float,
        default=0,
    )
    # parser.add_argument(
    #     "--use_E3",
    #     type=float,
    #     default=0,
    # )
    parser.add_argument(
        "--use_E4",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--use_S1",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--use_R1",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--A_0_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--A_1_0_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--B_0_3_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--B_1_3_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--B_1_4_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--D_3_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--D_4_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--E_6_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--E_2_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--E_5_json_path",
        type=str,
        default="",
    )
    # parser.add_argument(
    #     "--E_3_json_path",
    #     type=str,
    #     default="",
    # )
    parser.add_argument(
        "--E_4_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--S_1_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--R_1_json_path",
        type=str,
        default="",
    )
    # parser.add_argument(
    #     "--",
    #     type=str,
    #     default="",
    # )


    # enhance instructions? 
    parser.add_argument(
        '--use_enhancement_1', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_2', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_3', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_4', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_5', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_6', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_7', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_8', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_9', 
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--E2_sentence_form_answer', 
        action='store_true',
        default=False,
    )

    # only do portion of split files?
    parser.add_argument(
        '--skip_train', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--skip_valid', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--skip_test', 
        action='store_true',
        default=False,
    )

    args = parser.parse_args()

    main(args)







