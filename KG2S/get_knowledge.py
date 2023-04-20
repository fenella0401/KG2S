import numpy as np
import pandas as pd
import re
import pkuseg
import json
import requests
from multiprocessing import Process, Pool
import multiprocessing as mp
import time
import sys
from bert_serving.client import BertClient
import pickle as pkl
import argparse


# 使用bert-service获得句子向量化表示，并使用cosine计算句子相似度
class SimilarModel:
    def __init__(self):
        # ip默认为本地模式，如果bert服务部署在其他服务器上，修改为对应ip
        self.bert_client = BertClient()

    def close_bert(self):
        self.bert_client.close()

    def get_sentence_vec(self,sentence):
        '''
        根据bert获取句子向量
        :param sentence:
        :return:
        '''
        return self.bert_client.encode([sentence])[0]

    def cos_similar(self,sen_a_vec, sen_b_vec):
        '''
        计算两个句子的余弦相似度
        :param sen_a_vec:
        :param sen_b_vec:
        :return:
        '''
        vector_a = np.mat(sen_a_vec)
        vector_b = np.mat(sen_b_vec)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        return cos


# 静态知识的快速查询图构建
class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['CnDbpedia']
    """

    def __init__(self, spo_files, vocab, lookdown=False):
        self.lookdown = lookdown
        self.KGS = {'CnDbpedia': '../datasets/KGs/CnDbpedia.spo', 'OwnThink': '../datasets/KGs/OwnThink.spo', 'MedicalKG': '../datasets/KGs/MedicalKG.spo', 'Covid19KG': '../datasets/KGs/Covid19KG.spo'}
        self.spo_file_paths = [self.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.lookdown_table = self._create_lookdown_table()
        self.tokenizer = pkuseg.pkuseg(model_name="medicine", postag=False, user_dict=vocab)
        
    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        if len(line.strip().split("\t")) > 3:
                            triple = line.strip().split("\t")
                            subj = triple[0]
                            pred = triple[1]
                            obje = ''.join(triple[2:])
                        else:
                            print("[KnowledgeGraph] Bad spo:", line)
                    value = pred + obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def _create_lookdown_table(self):
        lookdown_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        if len(line.strip().split("\t")) > 3:
                            triple = line.strip().split("\t")
                            subj = triple[0]
                            pred = triple[1]
                            obje = ''.join(triple[2:])
                        else:
                            print("[KnowledgeGraph] Bad spo:", line)
                    value = subj + pred
                    if obje in lookdown_table.keys():
                        lookdown_table[obje].add(value)
                    else:
                        lookdown_table[obje] = set([value])
        return lookdown_table

# 静态知识抽取
def add_knowledge(graph, bert_client, sent, output_file, is_test):
    all_knowledge = {}
    all_entities = []
    if is_test == True:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(sent+'\n')
    split_sent = graph.tokenizer.cut(sent)
    if is_test == True:
        with open(output_file, 'a', encoding='utf-8') as f:
            for each in split_sent:
                f.write(each+'\t')
            f.write('\n')
    split_sent = list(set(split_sent))
    sen_emb = bert_client.get_sentence_vec(sent)
    for token in split_sent:
        know_sent = []
        entities = list(graph.lookup_table.get(token, []))
        if len(entities) != 0:
            all_entities.append(token)
        for each in entities:
            know_emb = bert_client.get_sentence_vec(token+each)
            cos_sim = bert_client.cos_similar(sen_emb, know_emb)
            know_sent.append((cos_sim, token+each))
            if is_test == True:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(token)
                    f.write(each)
                    f.write('\n')
        if graph.lookdown == True:
            entities = list(graph.lookdown_table.get(token, []))
            if len(entities) != 0:
                all_entities.append(token)
            for each in entities:
                know_emb = bert_client.get_sentence_vec(each+token)
                cos_sim = bert_client.cos_similar(sen_emb, know_emb)
                know_sent.append((cos_sim, each+token))
                if is_test == True:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(each)
                        f.write(token)
                        f.write('\n')
        all_knowledge[token] = know_sent
    if is_test == True:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n')
    
    #all_knowledge = '。'.join(all_knowledge)
    
    return all_knowledge, list(set(all_entities))

def get_knowledge(params):
    p_id, graph, bert_client, sentences, output_file, is_test = params
    knowledge = []
    entities = []
    sentences_num = len(sentences)
    for line_id, line in enumerate(sentences):
        if line_id % 100 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        know_tmp, ent_tmp = add_knowledge(graph, bert_client, line, output_file, is_test)
        knowledge.append(know_tmp)
        entities.append(ent_tmp)
    #random.shuffle(knowledge)
    return knowledge, entities

def get_spoKGs(graph, bert_client, sentences, workers_num):
    sentence_num = len(sentences)
    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
    start = time.time()
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append((i, graph, bert_client, sentences[i*sentence_per_block: (i+1)*sentence_per_block], '', False))
        pool = Pool(workers_num)
        res = pool.map(get_knowledge, params)
        pool.close()
        pool.join()
        dataset = res
    else:
        knowledge, entities = get_knowledge((0, graph, bert_client, sentences, '', False))
    end = time.time()
    running_time = end - start
    print('get_spoKGs running time:{}s'.format(running_time))
    return knowledge, entities

'''
# 清洗知识
def clean_know(knowledges, num):
    cleaned_know = []
    for line in knowledges:
        token_know = []
        line_know = []
        num_tmp = 0
        for token in line.keys():
            if len(line[token]) > 0:
                tmp = sorted(list(set(line[token])))
                while tmp[-1][0] > 0.92:
                    #print(tmp[-1])
                    line_know.append(tmp.pop()[1])
                    num_tmp += 1
                    if len(tmp) == 0:
                        break
                token_know.append(tmp)
        
        k_max = len(token_know)
        k = 0
        while num_tmp < num and k < k_max:
            k = 0
            for each in token_know:
                if len(each) > 0:
                    line_know.append(each.pop()[1])
                    num_tmp += 1
                else:
                    k += 1
        cleaned_know.append(line_know)
    return cleaned_know
'''


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path
    parser.add_argument("--path", default="", type=str,
                        help="Dataset for proccessed.")
    parser.add_argument("--dataset", default="train", type=str,
                        help="Dataset for proccessed.")
    
    args = parser.parse_args()

    dataset = pd.read_csv(args.path+args.dataset+'.tsv', encoding='utf-8', sep='\t')
    #kg = [OwnThink, 'MedicalKG', 'CnDbpedia']
    kg = ['Covid19KG', 'CnDbpedia']
    lookdown = True
    with open(args.path+'vocab.pkl','rb') as f:
        vocab = pkl.load(f)
    graph = KnowledgeGraph(kg, vocab, lookdown)
    bert_client = SimilarModel()

    sentences = dataset['text_a'].tolist()
    workers_num = 1
    #knowledge_num = 20
    knowledges, entities = get_spoKGs(graph, bert_client, sentences, workers_num)
    with open(args.path+args.dataset+'_know.pkl','wb') as f:
        pkl.dump(knowledges, f)
    with open(args.path+args.dataset+'_ent.pkl','wb') as f:
        pkl.dump(entities, f)
    bert_client.close_bert()
    print('抽取'+args.path+args.dataset+'知识成功')
    '''
    cleaned_know = clean_know(knowledges, knowledge_num)
    with open('data/'+args.dataset+'_know_cleaned.pkl','wb') as f:
        pkl.dump(cleaned_know, f)
    
    print('抽取'+args.dataset+'知识成功')
    '''

    '''
    # 读取数据集
    train = pd.read_csv('data/train.tsv', encoding='utf-8', sep='\t')
    dev = pd.read_csv('data/dev.tsv', encoding='utf-8', sep='\t')
    test = pd.read_csv('data/test.tsv', encoding='utf-8', sep='\t')

    # 知识查询图和相似度计算类实例化
    kg = ['CnDbpedia','Liuhuanyong']
    lookdown = True
    graph = KnowledgeGraph(kg, lookdown)
    bert_client = SimilarModel()

    # 抽取训练集知识
    sentences = train['text_a'].tolist()
    workers_num = 1
    knowledge_num = 20
    knowledges = {}
    knowledges = get_spoKGs(graph, bert_client, sentences, workers_num)
    knowledges = get_onlineKGs(sentences, knowledges, bert_client)
    cleaned_know = clean_know(knowledges, knowledge_num)
    with open('data/train_know.pkl','wb') as f:
        pkl.dump(cleaned_know, f)
    print('抽取训练集知识成功')
    
    # 抽取验证集知识
    sentences = dev['text_a'].tolist()
    workers_num = 1
    knowledge_num = 20
    knowledges = {}
    knowledges = get_spoKGs(graph, bert_client, sentences, workers_num)
    knowledges = get_onlineKGs(sentences, knowledges, bert_client)
    cleaned_know = clean_know(knowledges, knowledge_num)
    with open('data/dev_know.pkl','wb') as f:
        pkl.dump(cleaned_know, f)
    print('抽取验证集知识成功')

    # 抽取测试集知识
    sentences = test['text_a'].tolist()
    workers_num = 1
    knowledge_num = 20
    knowledges = {}
    knowledges = get_spoKGs(graph, bert_client, sentences, workers_num)
    knowledges = get_onlineKGs(sentences, knowledges, bert_client)
    cleaned_know = clean_know(knowledges, knowledge_num)
    with open('data/test_know.pkl','wb') as f:
        pkl.dump(cleaned_know, f)
    print('抽取测试集知识成功')
    '''

if __name__ == "__main__":
    main()