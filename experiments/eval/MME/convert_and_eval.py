import os
import json
import argparse
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from starter.env_getter import get_env
# Environment Variables

DATA = get_env('DATA')
CKPT_DIR = get_env('CKPT')
DATA_DIR = get_env('DATA')

# Argument Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, required=False)
args = parser.parse_args()

# Evaluation Type Dictionary
eval_type_dict = {
    "Perception": ["existence", "count", "position", "color"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}



def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        # print(qa_path)
        if 'eval_tool' in qa_path:
            continue
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                GT[(category, file, question)] = answer.lower()
    return GT

def collect_answers(answers, GT,use_top_1 = True):
    collected_results = defaultdict(dict)

    category_cnt = {}
    for ind, answer in enumerate(answers):
        if type(answer['question_id']) == list:
            answer['question_id'] = answer['question_id'][0]
        if 'prompt' in answer and type(answer['prompt']) == list:
            answer['prompt'] = answer['prompt'][0]
        category, file = answer['question_id'].split('/')[-2:]
        file = file.split('.')[0] + '.txt'
        if 'prompt' in answer:
            question = answer['prompt']
        else:
            raise NotImplementedError
        if 'Answer the question using a single word or phrase.' in question:
            question = question.replace('Answer the question using a single word or phrase.', '').strip()
        if 'Please answer yes or no.' not in question:
            question = question + ' Please answer yes or no.'
            if (category, file, question) not in GT:
                question = question.replace(' Please answer yes or no.', '  Please answer yes or no.')
        # print(GT)
        gt_ans = GT[(category, file, question)]
        if use_top_1:
            naive_ans = 'yes' if answer['naive'].get('yes', 0) > answer['naive'].get('no', 0) else 'no'
        else:
            naive_ans = answer['text'].lower().strip()
        
        collected_results[category][file, question] = (gt_ans, naive_ans)

        
        if category not in category_cnt:
            category_cnt[category] = 0
        else:
            category_cnt[category] += 1

    return collected_results

def divide_chunks(l, n=2):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def parse_pred_ans(pred_ans):
    """Parse predicted answer into standardized form."""
    pred_ans = pred_ans.lower()
    if pred_ans.startswith("yes"):
        return "yes"
    elif pred_ans.startswith("no"):
        return "no"
    return "other"

def compute_metric(gts, preds):
    """Compute accuracy, precision, recall, and F1-score."""
    acc = accuracy_score(gts, preds)
    precision = precision_score(gts, preds, pos_label='yes')
    recall = recall_score(gts, preds, pos_label='yes')
    return acc, precision, recall

def process_results(collected_results):
    all_res = {}
    for eval_type, task_name_list in eval_type_dict.items():
        print("===========", eval_type, "===========")
        scores = 0
        res_dict = {}
        for task_name in task_name_list:
            if task_name in collected_results:
                results = collected_results[task_name]
                lines = ["\t".join([file, q, gt, pred]) for (file, q), (gt, pred) in results.items()]
                chunk_lines = list(divide_chunks(lines, 2))  # One image corresponds to two questions
                img_num = len(chunk_lines)
                acc_plus_correct_num = 0
                gts = []
                preds = []
                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    img_correct_num = 0
                    for img_item in img_items:
                        _, _, gt_ans, pred_ans = img_item.split("\t")
                        pred_ans = parse_pred_ans(pred_ans)
                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                    if img_correct_num == 2:
                        acc_plus_correct_num += 1
                acc = accuracy_score(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                score = (acc * 100 + acc_plus * 100)
                scores += score
                res_dict[task_name] = score
        print('Total Score', scores)
        for k, v in res_dict.items():
            print(f"{k}: Score: {v:.2f}")
            all_res[k] = v

        print()
        all_res[eval_type] = scores
    return all_res



if __name__ == "__main__":
    data_path = f'{DATA_DIR}/MME_Benchmark'
    GT = get_gt(data_path)

    sum_scores = {}
    count_scores = {}
    experiment_name = 'llavav1.5-7b-test-v1'
    
    use_top_1 = True
    
    for i in range(1, 2):
        experiment = experiment_name + '-' + str(i)
        method = 'pm'
        answers = [json.loads(line) for line in open(os.path.join(f'{CKPT_DIR}/MME/llava/{method}/{experiment_name}', f'{experiment}.jsonl'))]

        collect_res = collect_answers(answers, GT, use_top_1=use_top_1)

        all_res = process_results(collect_res)
        print(all_res)

        for key, value in all_res.items():
            if key in sum_scores:
                sum_scores[key] += value
                count_scores[key] += 1
            else:
                sum_scores[key] = value
                count_scores[key] = 1

    # Calculate the average for each key
    print(count_scores)
    avg_res = {key: sum_scores[key] / count_scores[key] for key in sum_scores}
    for k, v in avg_res.items():
        # print(f"{k}: Score: {v:.2f}")
        print(f"{v:.2f}")
    


