### FILE TO SAVE PAIRS OF SNIPPETS FROM PAIRS OF ARTICLES IN NEWSHEAD DATASET
from transformers import pipeline
import torch
import os
from datetime import datetime
import pdb
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import random
import argparse
from multiprocessing import Pool
import numpy as np
from itertools import combinations
from tqdm import tqdm
import re
import pandas as pd 

# Login to HF to access Meta LLAMA model
from huggingface_hub import login
login("")

# function to align a given snippet with all other snippets in the context documents and select the highest-alignment other snippet to pair in instruction generation
def align(snippet_idx, snippet_emb, all_embs, all_sents_in_cluster, sim_threshold=0.8):
    
    # Remove reference snippet from full list of snippets & embeddings
    del all_sents_in_cluster[snippet_idx]

    # Align reference snippet with other snippets
    # docs: https://huggingface.co/sentence-transformers/msmarco-distilbert-cos-v5
    cos_sims = util.cos_sim(snippet_emb, all_embs)[0].cpu().numpy()

    # Identify index of highest-alignment snippet that is not too similar (and not same snippet)
    mask = cos_sims < sim_threshold
    try:
        max_sim_idx = np.nonzero(mask)[0][np.argmax(cos_sims[mask])]
    except: # if no cosine similarities are less than threshold
        max_sim_idx = -1
    
    # If length of second snippet not great
    if len(all_sents_in_cluster[max_sim_idx][0].split())<=10:
        return -1

    return max_sim_idx


# Function to generate and save instruction tuning prompts/examples
def generate_and_save_snippet_pairs(
    input_dir,
    output_dir,
):
    # Instantiate sentence embedding model
    emb_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
            
    # access all the newshead file addresses (train/valid/test split .pt files)
    all_files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(".pt")
    ]
    all_files = all_files[::-1] # reverse order of files

    # for each data split (train/valid/test)
    for file_idx, file_name in enumerate(tqdm(all_files)):
        print(file_name)

        # load the data which is a list of lists of document Strings(?)
        all_clusters = torch.load(os.path.join(input_dir, file_name))
        all_source_docs = []
        all_answers = []

        count = 0
        for cluster in all_clusters:
            print("on cluster",count)

            # Access all segments separated by \n in the dataset and embed them
            all_sents_in_cluster = [(sentence, emb_model.encode(sentence), document) for document in cluster for sentence in document.split('\n')]
            all_embs = emb_model.encode(list(zip(*all_sents_in_cluster))[0])

            selected_sent_pairs = []

            # Select specified number of snippet pairs for instruction generation
            for i in range(len(all_sents_in_cluster)//2):
                
                # Randomly select a snippet index
                snippet1_idx = random.choice(range(len(all_sents_in_cluster)))
                snippet1 = all_sents_in_cluster[snippet1_idx]
                snippet1_emb = all_embs[snippet1_idx]

                # Don't use snippets that are too short (length <10 words)
                if len(snippet1[0].split())<=10:
                    pass

                else:
                    # Identify highest-alignment snippet that is not too similar
                    all_embs = np.delete(all_embs, snippet1_idx, axis=0)
                    snippet2_idx = align(snippet1_idx, snippet1_emb, all_embs, all_sents_in_cluster, sim_threshold=0.7)
                    
                    # If there is no such snippet then do nothing
                    if snippet2_idx == -1:
                        pass
                    # Otherwise, add the pair to the snippet pair pool
                    else:
                        snippet2 = all_sents_in_cluster[snippet2_idx]

                        selected_sent_pairs.append([(snippet1[0],snippet1[2]), (snippet2[0], snippet2[2])])

                        # Remove the second snippet from the pool
                        del all_sents_in_cluster[snippet2_idx]
                        all_embs = np.delete(all_embs, snippet2_idx, axis=0)

            # Save snippet pairs to list
            for [(snippet1, doc1),(snippet2, doc2)] in selected_sent_pairs:
                all_source_docs.append([doc1, doc2])
                all_answers.append([snippet1, snippet2])
            
            count += 1

        print("finished selecting snippets for",file_name)
        
        # ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # CSV Saving Notes
        ## All the CSV files in the dataset should have the same organization and in particular the same datatypes for the columns.
        ## Columns: articles, answer
        ## articles in all_source_docs, answer in all_answers
        
        data = {"articles": all_source_docs, "answer": all_answers}
        snippet_data = pd.DataFrame(data)
        snippet_data_csv_name = os.path.join(output_dir,file_name.replace('.pt', '.csv'))
        snippet_data.to_csv(snippet_data_csv_name, index=False)
        print("Finished saving snippet pair+source docs csv at %s"%(snippet_data_csv_name))

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
        default="./base_snippet_pairs",
    )
    
    args = parser.parse_args()
    print(args)

    current_date_time = str(datetime.now().strftime("%Y-%m-%d_hr%H-min%M"))
    
    generate_and_save_snippet_pairs(
        args.input_dir,
        args.output_dir,
    )







