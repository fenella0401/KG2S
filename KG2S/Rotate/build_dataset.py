import numpy as np
import pandas as pd
import re
import json
import requests
from multiprocessing import Process, Pool
import multiprocessing as mp
import time
import sys
import pickle as pkl
import pkuseg

KGS = {'MedicalKG': '../../datasets/KGs/MedicalKG.spo'}
spo_files = ['MedicalKG']
spo_file_paths = [KGS.get(f, f) for f in spo_files]

entity2id = {}
relation2id = {}
train2id = []

entityid = 0
relationid = 0

rel_allow = ['症状','治疗方式','检查方式','常用药','并发症','就诊科室']

for spo_path in spo_file_paths:
    print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
    with open(spo_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                subj, pred, obje = line.strip().split("\t")
                if pred in rel_allow and subj != obje:
                    if subj not in entity2id.keys():
                        entity2id[subj] = entityid
                        entityid += 1
                    if obje not in entity2id.keys():
                        entity2id[obje] = entityid
                        entityid += 1
                    if pred not in relation2id.keys():
                        relation2id[pred] = relationid
                        relationid += 1
                    train2id.append([entity2id[subj], entity2id[obje], relation2id[pred]])
            except:
                if len(line.strip().split("\t")) > 3:
                    triple = line.strip().split("\t")
                    subj = triple[0]
                    pred = triple[1]
                    obje = ''.join(triple[2:])
                    if pred in rel_allow and subj != obje:
                        if subj not in entity2id.keys():
                            entity2id[subj] = entityid
                            entityid += 1
                        if obje not in entity2id.keys():
                            entity2id[obje] = entityid
                            entityid += 1
                        if pred not in relation2id.keys():
                            relation2id[pred] = relationid
                            relationid += 1
                        train2id.append([entity2id[subj], entity2id[obje], relation2id[pred]])
                else:
                    print("[KnowledgeGraph] Bad spo:", line)
            if entityid % 1000 == 0:
                print('entitynum: {}'.format(entityid))

all_dataset = {'entityid': entityid, 'relationid': relationid, 'entity2id': entity2id, 'relation2id': relation2id, 'train2id': train2id}
with open('all_dataset.pkl', 'wb') as f:
    pkl.dump(all_dataset, f)

print('get static KG success')
print(all_dataset['entityid'], all_dataset['relationid'])