# Evaluation functions for ZeroScrolls, copied from official repo
import re
import string
from collections import Counter
from unidecode import unidecode

################################################
############## COMPUTE F1 SCORE ################
################################################
def normalize_answer_f1(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return unidecode(white_space_fix(remove_articles(remove_punc(lower(s)))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer_f1(prediction).split()
    ground_truth_tokens = normalize_answer_f1(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def compute_f1(predictions, references):
    f1 = 0
    for prediction, ground_truths in zip(predictions, references):
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return 100.0 * f1 / len(predictions)

def compute_f1_instr_tune(predictions, references):
    f1 = 0
    total = 0
    for prediction, ground_truth in zip(predictions, references):
        total += 1
        f1 += f1_score(prediction, ground_truth)
    return 100.0 * f1 / total

################################################
############## COMPUTE EM SCORE ################
################################################
def exact_match_score(prediction, ground_truth):
    return normalize_answer_f1(prediction) == normalize_answer_f1(ground_truth)

def compute_em(predictions, references):
    em = 0
    for prediction, ground_truths in zip(predictions, references):
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    return 100.0 * em / len(predictions)

def compute_em_instr_tune(predictions, references):
    em = 0
    total = 0
    for prediction, ground_truth in zip(predictions, references):
        total += 1
        em += exact_match_score(prediction, ground_truth)
    return 100.0 * em / total

################################################
############ COMPUTE ROUGE SCORE ###############
################################################
import multiprocessing
import nltk
from rouge_score import rouge_scorer
from multiprocessing import Pool

# def rouge_score(predictions, references, rouge_types=None, use_stemmer=True):
#     if rouge_types is None:
#         rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

#     scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
#     with Pool() as p:
#         scores = p.starmap(scorer.score, zip(references, predictions))

#     result = {}
#     for key in scores[0]:
#         result[key] = list(score[key] for score in scores)

#     return result

# def compute_rouge(predictions, references):
#     rouge = 0
#     for prediction, ground_truths in zip(predictions, references):
#         rouge += metric_max_over_ground_truths(rouge_score, prediction, ground_truths)
#     return 100.0 * rouge / len(predictions)

# # Copied from https://github.com/huggingface/transformers/blob/3977b58437b8ce1ea1da6e31747d888efec2419b/examples/pytorch/summarization/run_summarization.py#L520
# def postprocess_text(text):
    # rougeLSum expects newline after each sentence
    # return "\n".join(nltk.sent_tokenize(text))



################################################
########### COMPUTE EXP SIMILARITY #############
################################################
PATTERN = re.compile(r'\d+\.?\d*%')

def find_percentage(s):
    match = PATTERN.search(s)
    if match is None:
        return None
    return match.group(0)

def to_int(s):
    percentage_string = find_percentage(s)
    if percentage_string is None:
        return None
    percentage_string = percentage_string.replace("%", "")
    percentage = float(percentage_string)
    return percentage

def exp_similarity_score(prediction, ground_truth):
    ground_truth_percentage = to_int(ground_truth)
    pred_percentage = to_int(str(prediction))

    if ground_truth_percentage is None:
        # raise ValueError(f"ground_truth_percentage is None: {ground_truth_percentage}")
        ground_truth_percentage=0

    if pred_percentage is None:
        return 0.0

    return 0.5 ** (abs(ground_truth_percentage - pred_percentage) / 10)

def compute_exp_similarity(predictions, references):
    exp_similarity = 0
    for prediction, ground_truths in zip(predictions, references):
        exp_similarity += metric_max_over_ground_truths(exp_similarity_score, prediction, ground_truths)
    return 100 * exp_similarity / len(predictions)


################################################
########### COMPUTE CONCORDANCE IDX ############
################################################
from lifelines.utils import concordance_index

def keep_integers_commas_spaces(input_string):
    cleaned_string = re.sub(r'[^0-9\s,]', '', str(input_string))
    return cleaned_string

def normalize_answer_ci(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    normalized_list = keep_integers_commas_spaces(s).replace(",", " ").strip(string.punctuation).split()
    try:
        normalized_list = [int(remove_punc(x).strip()) for x in normalized_list]
    except ValueError:
        return []
    return normalized_list

def concordant_index_score(prediction, ground_truth):
    normalized_prediction = normalize_answer_ci(prediction)
    normalized_ground_truth = normalize_answer_ci(ground_truth)
    if sorted(normalized_ground_truth) != sorted(normalized_prediction):
        return 0.0

    pred_order = summ_id_per_location_to_pos_of_id(normalized_prediction)
    gold_order = summ_id_per_location_to_pos_of_id(normalized_ground_truth)

    return concordance_index(gold_order, pred_order)

def summ_id_per_location_to_pos_of_id(id_per_location):
    order = [-1] * len(id_per_location)
    for i, id_ in enumerate(id_per_location, 1):
        order[id_ - 1] = i
    return order

def compute_concordance_index(predictions, references):
    concordant_index = 0
    for prediction, ground_truths in zip(predictions, references):
        concordant_index += metric_max_over_ground_truths(concordant_index_score, prediction, ground_truths)
    return 100.0 * concordant_index / len(predictions)

################################################
############## COMPUTE ACCURACY ################
################################################
PATTERN_ACCURACY = re.compile(r'\b[A-D]\b')

def find_answer(s):
    match = PATTERN_ACCURACY.search(s)
    if match is None:
        return None
    return match.group()

def accuracy_score(prediction, ground_truth):
    letter_ground_truth = find_answer(ground_truth)
    assert letter_ground_truth in ["A", "B", "C", "D"], f"Invalid ground truth: {ground_truth}"
    letter_prediction = find_answer(str(prediction))
    return letter_prediction == letter_ground_truth

def compute_accuracy(predictions, references):
    accuracy = 0
    for prediction, ground_truths in zip(predictions, references):
        accuracy += metric_max_over_ground_truths(accuracy_score, prediction, ground_truths)
    return 100.0 * accuracy / len(predictions)
