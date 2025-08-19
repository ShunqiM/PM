import base64
import requests
from PIL import Image
from io import BytesIO



import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import json

import openai
from openai import OpenAI
import time

NUM_SECONDS_TO_SLEEP = 0.5

client = OpenAI()


# Prompt based on OPERA https://arxiv.org/html/2311.17911v2

GPT_JUDGE_PROMPT = '''
You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinations should be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not countas necessary details.
Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgement.

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy: <Scores of the two answers>
Reason:

Detailedness: <Scores of the two answers>
Reason: 
'''


# OpenAI API Key
API_KEY = "YOUR_API_KEY"

def call_api(prompt, image_path):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json().keys())
    if 'error' in response.json().keys():
        print(response)
    # exit()
    return response.json()


def get_eval(content: str, max_tokens: int, image_path):

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    while True:
        try:
            completion = client.chat.completions.create(
                model='gpt-4o',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': [
                        {
                        "type": "text",
                        "text": prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        }
                    ],
                }],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            break
        except openai.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return completion.choices[0].message.content


def get_gpt4v_answer(prompt, image_path):
    while 1:
        try:
            res = call_api(prompt, image_path)
            if "choices" in res.keys():
                return res["choices"][0]["message"]["content"]
            else:
                assert False
        except Exception as e:
            print("retry")
            # pass
    # return call_api(prompt, image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=["./data/qa90_gpt4_answer.jsonl", "output/llava_bench/qwen-VL-Chat/qwen_naive_seed53_default.jsonl"])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('-i', '--image_path')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))

    if os.path.isfile(os.path.expanduser(args.output)) and os.path.getsize(args.output) > 0:
        cur_reviews = json.loads(args.output)
    else:
        cur_reviews = []


    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    gpt_answer_records = {}
    assistant_answer_records = {}
    avg_hal_score_1 = 0
    avg_hal_score_2 = 0
    avg_det_score_1 = 0
    avg_det_score_2 = 0
    num_count = 0

    handles = []
    idx = 0
    for ques_js, model_response_1, model_response_2 in tqdm(zip(f_q, f_ans1, f_ans2)):
        ques = json.loads(ques_js)
        ans1 = json.loads(model_response_1)
        ans2 = json.loads(model_response_2)

        inst = image_to_context[ques['image']]

        image_path = inst['image']
        image_path = os.path.join(args.image_path, image_path)
        question_id = ans1['question_id']
        prompt = GPT_JUDGE_PROMPT.format(ans1["text"], ans2["text"])
        if question_id in cur_reviews:
            gpt_answer = gpt_answer_records[question_id]
        else:
            gpt_answer = get_eval(prompt, args.max_tokens, image_path)
        gpt_answer_records[question_id] = gpt_answer

        # dump metric file
        with open(args.output, "w") as f:
            json.dump(gpt_answer_records, f)