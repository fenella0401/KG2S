import pandas as pd
import numpy as np
import pickle as pkl

with open('all_dataset.pkl', 'rb') as f:
    all_dataset = pkl.load(f)

entity2id = all_dataset['entity2id']
relation2id = all_dataset['relation2id']
train2id = all_dataset['train2id']

entityid = all_dataset['entityid']
relationid = all_dataset['relationid']

with open('entity2id.txt', 'w') as f:
    f.write(str(entityid))
    f.write('\n')

entity2id_sort = sorted(entity2id.items(), key=lambda x: x[1], reverse=False)
with open('entity2id.txt', 'a') as f:
    for each in entity2id_sort:
        f.write(str(each[0]))
        f.write('\t')
        f.write(str(each[1]))
        f.write('\n')

print('save entity2id success')

with open('relation2id.txt', 'w') as f:
    f.write(str(relationid))
    f.write('\n')

relation2id_sort = sorted(relation2id.items(), key=lambda x: x[1], reverse=False)
with open('relation2id.txt', 'a') as f:
    for each in relation2id_sort:
        f.write(str(each[0]))
        f.write('\t')
        f.write(str(each[1]))
        f.write('\n')

print('save relation2id success')

print(len(train2id))
train2id_new = {}
for each in train2id:
    if each[0] != each[1]:
        train2id_new[str(each)] = 0
print('clean success')

train2id_new_list = list(train2id_new.keys())
with open('train2id.txt', 'w') as f:
    f.write(str(len(train2id_new_list)))
    f.write('\n')

print(len(train2id_new_list))

with open('train2id.txt', 'a') as f:
    for each in train2id_new_list:
        tmp = each.lstrip('[').rstrip(']').split(',')
        f.write(tmp[0].strip())
        f.write('\t')
        f.write(tmp[1].strip())
        f.write('\t')
        f.write(tmp[2].strip())
        f.write('\n')

print('save train2id success')