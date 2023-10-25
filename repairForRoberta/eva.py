# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=int(label)
    return predictions

def calculate_scores(answers, predictions):
    TP, TN, FP, FN = 0, 0, 0, 0
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()

        if answers[key] == 1 and predictions[key] == 1:
            TP += 1
        elif answers[key] == 0 and predictions[key] == 0:
            TN += 1
        elif answers[key] == 0 and predictions[key] == 1:
            FP += 1
        elif answers[key] == 1 and predictions[key] == 0:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    scores = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    scores = calculate_scores(answers, predictions)

    print("Accuracy:", scores['Accuracy'])
    print("Precision:", scores['Precision'])
    print("Recall:", scores['Recall'])
    print("F1 Score:", scores['F1 Score'])

if __name__ == '__main__':
    main()
