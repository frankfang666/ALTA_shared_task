from LoadData import read_json_objs

example_objs = read_json_objs('../Research_Dataset/M4/arxiv_chatGPT.jsonl')
# print(read_json_objs('../Research_Dataset/M4/arxiv_chatGPT.jsonl')[0]['human_text'])
# print(read_json_objs('../Research_Dataset/M4/arxiv_chatGPT.jsonl')[0]['machine_text'])

print(len(example_objs))