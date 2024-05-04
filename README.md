# Instruction Tuning to Improve Multi-Document Processing Capabilities of LLMs
### CPSC 577 Final Project: Spring 2024
#### Kaili Liu & Richard Luo
-----

## Getting Started

After cloning the repository, create a conda environment for Python 3 using the `requirements.txt` file:
```
conda create --name <env_name> --file requirements.txt
```
Activate the conda environment by running:
```
conda activate <env_name>
```
where `<env_name>` is your name of choice for the conda environment.

## Using this repo
The results for this project can be reproduced using the following steps:
1. Download NewSHead data
2. Extract the NewSHead articles
3. Generate pairs of snippets for each cluster in the dataset
4. Use the snippets to generate instruction generation prompts and run inference on a generator model of choice to obtain instructions
5. Curate the instructions using a selector model of choice
6. Prepare the curated for input as a Dataset to HuggingFace Transformers
7. Use the generated dataset to instruction-tune LongT5-Base (or FlanT5-Base)
8. Evaluate the instruction-tuned model on a dataset of choice

We walk through each of these steps below.

## Step 1. Download NewSHead data

Download the cleaned NewSHead dataset from [here](https://storage.googleapis.com/primer_summ/newshead_data.tar.gz) and unzip the .tar.gz file in the base directory:
```
tar -xvf newshead_data.tar.gz
```
## Step 2. Extract the NewSHead articles

After Step 1 is complete, run `base_articles_extraction.py` to extract the articles for use in Step 3. 
```
python base_articles_extraction.py
```
Be sure to create the `./base_articles` directory first!

## Step 3. Generate pairs of snippets for each cluster in the dataset

Run:
```
python save_snippet_pairs.py
```
## Step 4. Use the snippets to generate instruction generation prompts and run inference on a generator model of choice to obtain instructions

This will be done using `generate_instructions.py`. If you use LLAMA2, make sure to paste your HuggingFace authentication token in line 25 of the file before running. You may need to request access via HuggingFace in advance if you have not done so previously.

This file can be run with several combinations of arguments — for basic usage, you can do:
```
python generate_instructions.py --instruction_format="A_1_0" --model_name="llama2-chat-7b"
```
This will use LLAMA2-Chat-7B as the generator model as in the final project report and prompt it to produce candidate instruction data according to template format `A_1_0` (which corresponds to A.1 in the project report).

The resulting instructions will be saved to a folder path of the form:
```
./generated_instructions/style_A_1_0/.../instructions
```
And the prompts used to obtain these instructions will be saved to:
```
./generated_instructions/style_A_1_0/.../prompts
```
For a full list of possible models and instruction templates, see the choices listed under the `--model_name` and `--instruction_format` arguments in line 457 and 438, respectively.

## Step 5. Curate the instructions using a selector model of choice

To do this, we will use `self_curation.py`. Continuing the example above, the command to do so is:
```
python self_curation.py --input_dir="./generated_instructions/style_B_1_4/..." --instruction_format="A_1_0" --model_name="chatglm2-6b"
```
As before, a full list of instruction formats and selector models can be seen in the end of the file.

This will dump the curated instructions into a series of .json files for use with HuggingFace Transformers.

## Step 6. Prepare the curated for input as a Dataset to HuggingFace Transformers

Continuing the example, to generate a dataset of 25000 examples all originating from template type `A_1_0` with scoring threshold 4, run:
```
python generate_json_splits.py --total_instr_num=25000 --instr_type_proportions_id=<data_id> --instr_thresh_num=4 --use_ABDE6=1  --A_1_0_json_path="./generated_instructions/style_A_1_0/.../data_jsons/scorer_chatglm2-6b/thresh_4"
```
Here, `<data_id>` a string identifier you wish to use to reference this dataset. If you wish to add an instruction enhancement as specified in the project report, you can add an additional argument such as `--use_enhancement_1`. For a full list of possible enhancements, see the end of the file.

## Step 7. Use the generated dataset to instruction-tune LongT5-Base

Now our data is ready and we can start an instruction tuning run. To do this for the ongoing example using similar parameters as in the project report, run:
```
python instr_tune_longt5b.py --json_dir="data_splits/<data_id>/thresh_4/enhance_0/25000" --wandb_project_name=<proj_name> --output_dir=<out_dir> --lr=0.001 --lr_scheduler_type="constant" --warmup_steps=0 --save_steps=50 --eval_steps=50 --num_train_epochs=3 --fp16 --train_num_samples=25000 --per_batch_size=2 --grad_acc_steps=64
```
Here, `<proj_name>` and `<out_dir>` can be set as preferred.

As before, to see the full list of possible arguments and how they are used, feel free to take a closer look at the file.

## Step 8. Evaluate the instruction-tuned model on a dataset of choice

Evaluation through 0-shot prompting on MultiNews or HotpotQA can be run using:
```
python run_mch.py --model_dir=<model_dir>
```
where `<model_dir>` is the directory to the model checkpoint you’d like to use. Evaluation on ZeroScrolls can be run using the [official ZeroScrolls code](https://github.com/tau-nlp/zero_scrolls).
