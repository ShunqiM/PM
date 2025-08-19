import os
import json
import argparse
import csv
from tqdm import tqdm
from starter.env_getter import get_env

CKPT_DIR = get_env('CKPT')

parser = argparse.ArgumentParser()
parser.add_argument("--gt_files")
parser.add_argument("--gen_files")
args = parser.parse_args()
flag = False
total_true = 0
cnt = 0

# Run loop for datasets and splits
for data in ['aokvqa', 'gqa', 'coco']:
    print('*' * 20 + f'Dataset {data}' + '*' * 20)
    for split in ['adversarial', 'popular', 'random']:
        print('*' * 20 + f'Split {split}' + '*' * 20)

        # Initialize variables for averaging
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        avg_accuracy = 0
        num_seeds = 0

        for seed in range(55, 56):
        
            if not os.path.exists(args.gen_files):
                print('Results file not found', args.gen_files)
                continue

            # Set up CSV file path in the same directory as gen_files
            csv_file_path = os.path.join(os.path.dirname(args.gen_files), 'evaluation_results.csv')
            if not flag:  # Create CSV file and write headers when processing the first seed
                flag = True
                with open(csv_file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Data', 'Split', 'Seed Range', 'F1', 'Accuracy', 'Precision (%)', 'Recall (%)'])

            # Open and read files
            gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]
            gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]

            # Initialize counters
            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0
            total_questions = len(gt_files)

            # Process each question-answer pair
            for index, line in enumerate(gt_files):
                idx = line["question_id"]
                
                gt_answer = line["label"]
                assert idx == gen_files[index]["question_id"]
                logits_score = gen_files[index]["logits_score"]
                gen_answer = 'yes' if logits_score[0] > logits_score[1] else 'no'
                gt_answer = gt_answer.lower().strip()

                if gt_answer == 'yes':
                    if 'yes' in gen_answer:
                        true_pos += 1
                    else:
                        false_neg += 1
                elif gt_answer == 'no':
                    if 'no' in gen_answer:
                        true_neg += 1
                    else:
                        false_pos += 1
                row = gen_files[index]
                cnt += 1
                

            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_pos + true_neg) / total_questions if total_questions > 0 else 0

            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            avg_accuracy += accuracy
            num_seeds += 1

            total_true += true_pos + true_neg
            print(total_true)


        # Average calculations after all seeds -- We care doing determinstic sampling so this is not needed
        if num_seeds > 0:
            avg_precision /= num_seeds
            avg_recall /= num_seeds
            avg_f1 /= num_seeds
            avg_accuracy /= num_seeds

            avg_f1_percent = round(avg_f1 * 100, 2)
            avg_accuracy_percent = round(avg_accuracy * 100, 2)
            # Write results to CSV file after processing all seeds for a dataset and split
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([data, split, '55-59', avg_f1_percent, avg_accuracy_percent, avg_precision, avg_recall])

        print(f'Average Precision for {data} {split}: {avg_precision:.4f}')
        print(f'Average Recall for {data} {split}: {avg_recall:.4f}')
        print(f'Average F1 for {data} {split}: {avg_f1:.4f}')
        print(f'Average Accuracy for {data} {split}: {avg_accuracy:.4f}')
