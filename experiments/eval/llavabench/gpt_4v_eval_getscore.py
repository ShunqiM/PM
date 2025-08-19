import argparse
import json
import os


def parse_scores(output_file):
    # Load the JSON file
    with open(output_file, 'r') as f:
        gpt_answer_records = json.load(f)

    # Initialize accumulators
    avg_hal_score_1 = 0
    avg_hal_score_2 = 0
    avg_det_score_1 = 0
    avg_det_score_2 = 0
    num_count = 0

    # Parse the scores
    for question_id, evaluation in gpt_answer_records.items():
        evaluation = evaluation.replace('**', '')
        print('*' * 20)
        try:
            # Extract scores for Accuracy
            hal_scores = evaluation.split("Accuracy: ")[-1].split("\n")[0].strip().split(" ")
            # print(evaluation.split("Accuracy: ")[-1].split("\n")[0])
            print('h', hal_scores)
            hal_score_1, hal_score_2 = map(int, hal_scores)
            # Extract scores for Detailedness
            det_scores = evaluation.split("Detailedness: ")[-1].split("\n")[0].strip().split(" ")
            print('d', det_scores)
            det_score_1, det_score_2 = map(int, det_scores)

            if hal_scores[1] == '0' and det_scores[1] == '0':
                print('skipping error generation')
                continue

            # Accumulate scores
            avg_hal_score_1 += hal_score_1
            avg_hal_score_2 += hal_score_2
            avg_det_score_1 += det_score_1
            avg_det_score_2 += det_score_2
            num_count += 1
        except (IndexError, ValueError) as e:
            print(f"Skipping malformed evaluation for question_id: {question_id}")
            print(question_id, evaluation)
            print(e)
            # exit()

    # Calculate averages
    if num_count > 0:
        print('Total Valid', num_count)
        avg_hal_score_1 /= num_count
        avg_hal_score_2 /= num_count
        avg_det_score_1 /= num_count
        avg_det_score_2 /= num_count
    else:
        print("No valid evaluations found.")

    # Print results
    print(f"Average Accuracy Scores: Relative Score: {avg_hal_score_2*100/avg_hal_score_1:.2f}, Assistant 1: {avg_hal_score_1*10:.2f}, Assistant 2: {avg_hal_score_2*10:.2f}")
    print(f"Average Detailedness Scores: {avg_det_score_2*100/avg_det_score_1:.2f},  Assistant 1: {avg_det_score_1*10:.2f}, Assistant 2: {avg_det_score_2*10:.2f}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse and summarize evaluation scores.')
    parser.add_argument('-o', '--output', required=True, help='Path to the output JSON file')
    args = parser.parse_args()

    # Validate file path
    if not os.path.isfile(args.output):
        print(f"File not found: {args.output}")
    else:
        parse_scores(args.output)