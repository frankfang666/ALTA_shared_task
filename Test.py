from LoadData import read_json_objs
from FeatureCopy import RemoveSpecialCHs

# example_objs = read_json_objs('../Research_Dataset/M4/arxiv_chatGPT.jsonl')
# print(read_json_objs('../Research_Dataset/M4/arxiv_chatGPT.jsonl')[0]['human_text'])
# print(read_json_objs('../Research_Dataset/M4/arxiv_chatGPT.jsonl')[0]['machine_text'])

# print(len(example_objs))

train_objs = read_json_objs('./alta2023_public_data/training.json')
test_objs = read_json_objs('./alta2023_public_data/test_data.json')

word_lens = []
for train_obj in train_objs:
    text = train_obj['text']
    word_lens.append(len(RemoveSpecialCHs(text)))
    if len(RemoveSpecialCHs(text)) == 1:
        print(text)
print(min(word_lens))