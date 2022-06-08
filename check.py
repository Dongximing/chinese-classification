number = 0
validation_example = []
validation_label = []
with open('/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/dev.txt') as f1:
    validation_lines = f1.readlines()

for line in validation_lines:
    if number == 100:
        break
    example, label = line.split("\t")
    print(example)
    validation_example.append(example)
    validation_label.append(int(label))
    number+=1