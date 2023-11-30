from LoadData import read_json_objs
from collections import Counter
from TrainPredict import create_output_file

def majorityVoting(*args):
    objs = [read_json_objs(arg) for arg in args]
    final_labels = []
    for i in range(len(objs[0])):
        labels = list(map(lambda x: x['label'], [y[i] for y in objs]))
        final_label = sorted(Counter(labels).items(), key=lambda x: x[1])[-1][0]
        final_labels.append(final_label)
    return create_output_file(final_labels)
        
majorityVoting('../ALTA2023_submission/test_submissions/answer.json', '../ALTA2023_submission/test_submissions/DeBERTa9885/answer.json', '../ALTA2023_submission/test_submissions/Feature9845/answer.json')