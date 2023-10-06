from LoadData import read_json_objs
import os

path = '../Research_Dataset/M4/'
print(os.listdir(path))
print(read_json_objs(path + 'arxiv_chatGPT.jsonl')[0].keys())
# print(read_json_objs(path + 'arxiv_bloomz.jsonl')[0].keys())
print(read_json_objs(path + 'arxiv_cohere.jsonl')[0].keys())
print(read_json_objs(path + 'arxiv_davinci.jsonl')[0].keys())
# print(read_json_objs(path + 'arxiv_dolly.jsonl')[0].keys())
print(read_json_objs(path + 'arxiv_flant5.jsonl')[0].keys())