# -*- encoding:utf-8 -*-
import os
import random
import sys
import pickle as pkl
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
from sklearn.preprocessing import LabelEncoder
from torch._C import device
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from tqdm import notebook, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
import torch.nn as nn
import argparse
from multiprocessing import Process, Pool
from sklearn import metrics
import torch.nn.functional as F
import sklearn
import pkuseg
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Hierarchical_attention import Encoder
from act_fun import gelu

class DataPrecessForSingleSentence(object):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        # 创建多线程池
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, sentences, max_seq_len=30):
        """
        通过多线程（因为notebook中多进程使用存在一些问题）的方式对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        
        入参:
            sentences   : 传入的text。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。  
            
        """
        # 切词
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 获取定长序列及其mask
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments

    def trunate_and_pad(self, seq, max_seq_len):
        """
        1. 因为本类处理的是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        
        入参: 
            seq         : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度
        
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
           
        """
        # 对超长序列进行截断
        # 截断：1.取文本前510个字符; 2.取后510个字符; 3.前128个字符加后382个字符
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + seq + ['[SEP]']
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    # 调参 alpha=0.25, gamma=2为论文的最优结果
    # gamma占主导因素，gamma是用来控制难易分类样本的，当数据中难分类样本较多时，gamma可以设置的大一些
    # gamma增大时alpha要适当减小
    # alpha为0.5表示正负样本均等
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = Variable(torch.Tensor([alpha,1-alpha]))
            '''
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)'''
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class BertForSequenceClassification(nn.Module):
    def __init__(self, config, args, device, num_labels=2): # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert1 = BertModel.from_pretrained('../bert_model')
        self.bert2 = BertModel.from_pretrained('../bert_model')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.latent_dim = 768
        self.hidden_dim = 1024
        self.num_head1 = 6 # try
        self.num_head2 = 6
        self.num_head3 = 6
        self.layers_num1 = 1 # try
        self.layers_num2 = 1 # try
        self.layers_num3 = 1
        self.encoder1 = Encoder(self.latent_dim, self.num_head1, args.dropout, self.hidden_dim, self.layers_num1)
        self.encoder2 = Encoder(self.latent_dim, self.num_head2, args.dropout, self.hidden_dim, self.layers_num2)
        self.encoder3 = Encoder(self.latent_dim, self.num_head3, args.dropout, self.hidden_dim, self.layers_num3, True)
        self.output_layer_1 = nn.Linear(self.latent_dim*2, self.latent_dim*2)
        self.output_layer_2 = nn.Linear(self.latent_dim*2, num_labels)
        self.output_layer_3 = nn.Linear(self.latent_dim, self.latent_dim)
        self.output_layer_4 = nn.Linear(self.latent_dim, num_labels)
        #self.apply(self.init_bert_weights)\
        #nn.init.xavier_normal_(self.classifier.weight)
        self.pooling = args.pooling
        #self.dropout2 = nn.Dropout(args.dropout)
        #self.LeakyReLU = nn.LeakyReLU(0.1)
        self.device = device
        self.know_num = args.know_num

    def forward_once1(self, input_ids, token_type_ids=None, attention_mask=None): 
        # 12层encoder
        # token embedding 和 position embedding是学习得到的 (512, 768) 的lookup表
        last_layer, pooled_output = self.bert1(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        last_layer = self.dropout(last_layer)
        pooled_output = self.dropout(pooled_output)
        
        return last_layer, pooled_output

    def forward_once2(self, input_ids, token_type_ids=None, attention_mask=None): 
        # 12层encoder
        # token embedding 和 position embedding是学习得到的 (512, 768) 的lookup表
        _, pooled_output = self.bert2(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, batch_seqs_k, batch_seq_masks_k, batch_seq_segments_k, batch_know_emb, labels=None):
        # forward pass of information [[],[]]
        information_all, information_cls = self.forward_once1(batch_seqs, batch_seq_masks, batch_seq_segments)
        # forward pass of knowledge [[[],[]],[[],[]]] 8*20*768
        knowledge = torch.zeros(len(batch_seqs_k), self.know_num, self.latent_dim)
        knowledge = knowledge.to(self.device)
        for i in range(len(batch_seqs_k)):
            tmp = self.forward_once2(batch_seqs_k[i], batch_seq_masks_k[i], batch_seq_segments_k[i])
            knowledge[i] = tmp
        # knowledge graph embedding
        kg_emb = batch_know_emb
        
        kg_emb = self.encoder1(kg_emb)

        knowledge = self.encoder2(knowledge, kg_emb)
        #knowldege可以加入position区分每个triple
        
        information_all, attn = self.encoder3(information_all, knowledge)
        #knowldege可以加入position区分每个triple
        
        if self.pooling == "mean":
            output = torch.mean(information_all, dim=1)
        elif self.pooling == "max":
            output = torch.max(information_all, dim=1)[0]
        elif self.pooling == "last":
            output = information_all[:, -1, :]
        else:
            output = information_all[:, 0, :]

        output = torch.relu(self.output_layer_3(output))
        logits = self.output_layer_4(output)
        
        if labels is not None:
            #loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(2)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits, attn
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

def get_know_triplets(all_know, know_num, know_min, know_max, is_test):
    cleaned_know = []
    for line in all_know:
        token_know = []
        line_know = []
        num_tmp = 0
        for token in line.keys():
            if len(line[token]) > 0:
                tmp = sorted(list(set(line[token])))
                while tmp[-1][0] > know_max:
                    line_know.append(tmp.pop()[1])
                    num_tmp += 1
                    if len(tmp) == 0:
                        break
                token_know.append(tmp)
        
        k_max = len(token_know)
        k = 0
        while num_tmp < know_num and k < k_max:
            k = 0
            for each in token_know:
                if len(each) > 0 and each[-1][0] > know_min:
                    line_know.append(each.pop()[1])
                    num_tmp += 1
                else:
                    k += 1
        if len(line_know) < know_num:
            line_know.extend(['' for i in range(know_num-len(line_know))])
        
        cleaned_know.append(line_know)
    
    if is_test == 'test':
        with open('/home/fenella/project/KGbert/dataset/data20211210/test_know_cleaned.pkl','wb') as f:
            pkl.dump(cleaned_know, f)
    elif is_test == 'covid':
        with open('/home/fenella/project/KGbert/dataset/infodemic2019/covid_know_cleaned.pkl','wb') as f:
            pkl.dump(cleaned_know, f)
    
    '''
    # KG2S/B-S
    for line in all_know:
        line_know = []
        k = 0
        l = 0
        while k < know_num and l < len(line.keys()):
            l = 0
            for token in line.keys():
                if len(line[token]) > 0:
                    line_know.append(line[token].pop()[1])
                    k += 1
                else:
                    l += 1
        if len(line_know) < know_num:
            line_know.extend(['' for i in range(know_num-len(line_know))])
        cleaned_know.append(line_know)
    
    if is_test == 'test':
        with open('/home/fenella/project/KGbert/dataset/data20211210/test_know_cleaned_ablation.pkl','wb') as f:
            pkl.dump(cleaned_know, f)'''

    '''
    # KG2S/B-F
    for line in all_know:
        line_know = []
        for each in line.keys():
            if len(line[each]) > 0:
                line_know.extend(line[each])
        line_know = sorted(list(set(line_know)), reverse=True)
        line_know = [each[1] for each in line_know if each[0] > know_min]
        if len(line_know) > know_num:
            cleaned_know.append(line_know[:20])
        else:
            line_know.extend(['' for i in range(know_num-len(line_know))])
            cleaned_know.append(line_know)
    
    if is_test == 'test':
        with open('/home/fenella/project/KGbert/dataset/data20211210/test_know_cleaned_ablation.pkl','wb') as f:
            pkl.dump(cleaned_know, f)'''

    '''
    # KG2S/B-F
    for line in all_know:
        line_know = []
        for each in line.keys():
            if len(line[each]) > 0:
                line_know.extend(line[each])
        line_know = [each[1] for each in line_know]
        #random.shuffle(line_know)
        if len(line_know) > know_num:
            cleaned_know.append(line_know[:know_num])
        else:
            line_know.extend(['' for i in range(know_num-len(line_know))])
            cleaned_know.append(line_know)
    
    if is_test == 'test':
        with open('/home/fenella/project/KGbert/dataset/data20211210/test_know_cleaned_ablation.pkl','wb') as f:
            pkl.dump(cleaned_know, f)'''
    
    return cleaned_know

def get_know_emb(all_ent, entity2id, ent_embedding, ent_num):
    all_know_emb = []
    for each in all_ent:
        entity = []
        for e in each:
            if e in entity2id.keys():
                entity.append(ent_embedding[entity2id[e]].tolist())
            if len(entity) == ent_num:
                break
        if len(entity) < ent_num:
            entity.extend([np.zeros(768).tolist() for i in range(ent_num-len(entity))])
        all_know_emb.append(entity)
    return all_know_emb


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path
    parser.add_argument("--output_model_path", default="./models/classifier_model.pth", type=str,
                        help="Path of the output model.")
    parser.add_argument("--output_lossfig_path", default="./models/loss.png", type=str,
                        help="Path of the output model.")

    # Model options.
    # 调参 batch_size 8 16 32 (16 32 是Bert原文推荐的fine-tune参数)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--seq_length_sen", type=int, default=32,
                        help="Sequence length.") # try
    parser.add_argument("--seq_length_know", type=int, default=32,
                        help="Sequence length.") # try
    parser.add_argument("--know_num", type=int, default=15,
                        help="Num of selected knowledge.") # try
    parser.add_argument("--know_min", type=int, default=0.5,
                        help="The minux similarity of selected knowledge.") # try
    parser.add_argument("--know_max", type=int, default=0.95,
                        help="The max similarity of selected knowledge.") # try
    parser.add_argument("--ent_num", type=int, default=5,
                        help="Num of selected entities.") # try

    # Optimizer options.
    # 调参 learning rate 5e-5, 4e-5, 3e-5, 2e-5 (来自Bert原文推荐) 0.001
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    # 调参 dropout 0.1~0.3 (0.1为Bert推荐参数) 一般dropout都是0.5
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device use.")

    args = parser.parse_args()

    def set_seed(seed=7):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    set_seed(args.seed)
    
    # 读取数据
    train = pd.read_csv('../datasets/train.tsv', encoding='utf-8', sep='\t')
    dev = pd.read_csv('../datasets/dev.tsv', encoding='utf-8', sep='\t')
    test = pd.read_csv('../datasets/test.tsv', encoding='utf-8', sep='\t')
    covid = pd.read_csv('../datasets/covid.tsv', encoding='utf-8', sep='\t')
    with open('../datasets/train_know.pkl','rb') as f:
        train_know = pkl.load(f)
    with open('../datasets/dev_know.pkl','rb') as f:
        dev_know = pkl.load(f)
    with open('../datasets/test_know.pkl','rb') as f:
        test_know = pkl.load(f)
    with open('../datasets/covid_know.pkl','rb') as f:
        covid_know = pkl.load(f)
    with open('../datasets/train_ent.pkl','rb') as f:
        train_ent = pkl.load(f)
    with open('../datasets/dev_ent.pkl','rb') as f:
        dev_ent = pkl.load(f)
    with open('../datasets/test_ent.pkl','rb') as f:
        test_ent = pkl.load(f)
    with open('../datasets/covid_ent.pkl','rb') as f:
        covid_ent = pkl.load(f)
    # ent embeddings
    with open('../datasets/jz_ent_embeddings_384', 'rb') as f:
        jz_ent_embeddings = pkl.load(f)
    jz_entity2id = {}
    with open('../datasets/jz_entity2id.txt', 'r') as f:
        for each in f:
            tmp = each.split('\t')
            if len(tmp) > 1:
                jz_entity2id[tmp[0]] = int(tmp[1])
    with open('../datasets/covid_ent_embeddings_384', 'rb') as f:
        covid_ent_embeddings = pkl.load(f)
    covid_entity2id = {}
    with open('../datasets/covid_entity2id.txt', 'r') as f:
        for each in f:
            tmp = each.split('\t')
            if len(tmp) > 1:
                covid_entity2id[tmp[0]] = int(tmp[1])

    # Load bert vocabulary and tokenizer
    bert_config = BertConfig('../bert_model/bert_config.json') # try biobert
    BERT_MODEL_PATH = '../bert_model'
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=False)

    # 产生输入数据
    processor = DataPrecessForSingleSentence(bert_tokenizer=bert_tokenizer)
    
    # train dataset
    print("preparing train dataset")
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=train['text_a'].tolist(), max_seq_len=args.seq_length_sen)
    seqs_k, seq_masks_k, seq_segments_k = [], [], []
    cleand_know = get_know_triplets(all_know=train_know, know_num=args.know_num, know_min=args.know_min, know_max=args.know_max, is_test='train')
    for each in cleand_know:
        seqs_k_tmp, seq_masks_k_tmp, seq_segments_k_tmp = processor.get_input(
            sentences=each[:args.know_num], max_seq_len=args.seq_length_know)
        seqs_k.append(seqs_k_tmp)
        seq_masks_k.append(seq_masks_k_tmp)
        seq_segments_k.append(seq_segments_k_tmp)
    know_emb = get_know_emb(train_ent, jz_entity2id, jz_ent_embeddings, args.ent_num)
    labels = train['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_seqs_k = torch.tensor(seqs_k, dtype=torch.long)
    t_seq_masks_k = torch.tensor(seq_masks_k, dtype = torch.long)
    t_seq_segments_k = torch.tensor(seq_segments_k, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    t_know_emb = torch.tensor(know_emb, dtype = torch.float)
    train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_seqs_k, t_seq_masks_k, t_seq_segments_k, t_know_emb, t_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloder = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=args.batch_size)

    # dev dataset
    print("preparing dev dataset")
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=dev['text_a'].tolist(), max_seq_len=args.seq_length_sen)
    seqs_k, seq_masks_k, seq_segments_k = [], [], []
    cleand_know = get_know_triplets(all_know=dev_know, know_num=args.know_num, know_min=args.know_min, know_max=args.know_max, is_test='dev')
    for each in cleand_know:
        seqs_k_tmp, seq_masks_k_tmp, seq_segments_k_tmp = processor.get_input(
            sentences=each[:args.know_num], max_seq_len=args.seq_length_know)
        seqs_k.append(seqs_k_tmp)
        seq_masks_k.append(seq_masks_k_tmp)
        seq_segments_k.append(seq_segments_k_tmp)
    know_emb = get_know_emb(dev_ent, jz_entity2id, jz_ent_embeddings, args.ent_num)
    labels = dev['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_seqs_k = torch.tensor(seqs_k, dtype=torch.long)
    t_seq_masks_k = torch.tensor(seq_masks_k, dtype = torch.long)
    t_seq_segments_k = torch.tensor(seq_segments_k, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    t_know_emb = torch.tensor(know_emb, dtype = torch.float)
    dev_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_seqs_k, t_seq_masks_k, t_seq_segments_k, t_know_emb, t_labels)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloder = DataLoader(dataset=dev_data, sampler=dev_sampler, batch_size=args.batch_size)

    # test dataset
    print("preparing test dataset")
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=test['text_a'].tolist(), max_seq_len=args.seq_length_sen)
    seqs_k, seq_masks_k, seq_segments_k = [], [], []
    cleand_know = get_know_triplets(all_know=test_know, know_num=args.know_num, know_min=args.know_min, know_max=args.know_max, is_test='test')
    for each in cleand_know:
        seqs_k_tmp, seq_masks_k_tmp, seq_segments_k_tmp = processor.get_input(
            sentences=each[:args.know_num], max_seq_len=args.seq_length_know)
        seqs_k.append(seqs_k_tmp)
        seq_masks_k.append(seq_masks_k_tmp)
        seq_segments_k.append(seq_segments_k_tmp)
    know_emb = get_know_emb(test_ent, jz_entity2id, jz_ent_embeddings, args.ent_num)
    labels = test['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_seqs_k = torch.tensor(seqs_k, dtype=torch.long)
    t_seq_masks_k = torch.tensor(seq_masks_k, dtype = torch.long)
    t_seq_segments_k = torch.tensor(seq_segments_k, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    t_know_emb = torch.tensor(know_emb, dtype = torch.float)
    test_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_seqs_k, t_seq_masks_k, t_seq_segments_k, t_know_emb, t_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloder = DataLoader(dataset=test_data, sampler=test_sampler, batch_size=args.batch_size)

    # covid dataset
    print("preparing covid dataset")
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=covid['text_a'].tolist(), max_seq_len=args.seq_length_sen)
    seqs_k, seq_masks_k, seq_segments_k = [], [], []
    cleand_know = get_know_triplets(all_know=covid_know, know_num=args.know_num, know_min=args.know_min, know_max=args.know_max, is_test='covid')
    for each in cleand_know:
        seqs_k_tmp, seq_masks_k_tmp, seq_segments_k_tmp = processor.get_input(
            sentences=each[:args.know_num], max_seq_len=args.seq_length_know)
        seqs_k.append(seqs_k_tmp)
        seq_masks_k.append(seq_masks_k_tmp)
        seq_segments_k.append(seq_segments_k_tmp)
    know_emb = get_know_emb(covid_ent, covid_entity2id, covid_ent_embeddings, args.ent_num)
    labels = covid['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_seqs_k = torch.tensor(seqs_k, dtype=torch.long)
    t_seq_masks_k = torch.tensor(seq_masks_k, dtype = torch.long)
    t_seq_segments_k = torch.tensor(seq_segments_k, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    t_know_emb = torch.tensor(know_emb, dtype = torch.float)
    covid_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_seqs_k, t_seq_masks_k, t_seq_segments_k, t_know_emb, t_labels)
    covid_sampler = SequentialSampler(covid_data)
    covid_dataloder = DataLoader(dataset=covid_data, sampler=covid_sampler, batch_size=args.batch_size)


    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build classification model
    model = BertForSequenceClassification(bert_config, args, device, 2)

    if device == 'cuda':
        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    model = model.to(device)

    # evaluation function
    def evaluate(args, is_test, metrics='f1'):
        if is_test == 'test':
            dataset = test_dataloder
            instances_num = test.shape[0]
            print("The number of evaluation instances: ", instances_num)
        elif is_test == 'covid':
            dataset = covid_dataloder
            instances_num = covid.shape[0]
            print("The number of evaluation instances: ", instances_num)
        else:
            dataset = dev_dataloder
            instances_num = dev.shape[0]
            print("The number of evaluation instances: ", instances_num)
        
        correct = 0
        model.eval()
        # Confusion matrix.
        confusion = torch.zeros(2, 2, dtype=torch.long)
        
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)

        attn_all = torch.zeros(instances_num, 4, args.seq_length_sen, args.know_num)
        attn_num = 0

        for i, batch_data in enumerate(dataset):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_seqs_k, batch_seq_masks_k, batch_seq_segments_k, batch_know_emb, batch_labels = batch_data        
            with torch.no_grad():
                logits, attn = model(
                    batch_seqs, batch_seq_masks, batch_seq_segments, batch_seqs_k, batch_seq_masks_k, batch_seq_segments_k, batch_know_emb, labels=None)
            pred = logits.softmax(dim=1).argmax(dim=1)
            gold = batch_labels
            labels = batch_labels.data.cpu().numpy()
            predic = pred.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()
            #if is_test != 'dev':
                #attn_all[attn_num:(attn_num+len(attn))] = attn
                #attn_num += len(attn)
            
        if is_test != 'dev':
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")
            if is_test == 'test':
                with open('/home/fenella/project/KGbert/dataset/data20211210/attn_ablation.pkl', 'wb') as f:
                    pkl.dump(attn_all, f)
            #else:
            #    with open('/home/fenella/project/KGbert/dataset/infodemic2019/attn.pkl', 'wb') as f:
            #        pkl.dump(attn_all, f)

        f1_avg = 0
        for i in range(confusion.size()[0]):
            if confusion[i,:].sum().item() == 0:
                p = 0
            else:
                p = confusion[i,i].item()/confusion[i,:].sum().item()
            if confusion[:,i].sum().item() == 0:
                r = 0
            else:
                r = confusion[i,i].item()/confusion[:,i].sum().item()
            if (p+r) == 0:
                f1 = 0
            else:
                f1 = 2*p*r / (p+r)
                f1_avg += f1
            if i == 1:
                label_1_f1 = f1
            print("Label {}: {:.4f}, {:.4f}, {:.4f}".format(i,p,r,f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/instances_num, correct, instances_num))
        test_auc = sklearn.metrics.roc_auc_score(labels_all, predict_all)
        print(test_auc)
        acc = sklearn.metrics.accuracy_score(labels_all, predict_all)
        report = sklearn.metrics.classification_report(labels_all, predict_all,digits=4)
        weighted_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='weighted')
        print(report)
        if metrics == 'Acc':
            return correct/instances_num
        elif metrics == 'f1':
            return weighted_f1
        else:
            return correct/instances_num

    # training phase
    print("Start training.")
    instances_num = train.shape[0]
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)


    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            0.01
        },
        {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }
    ]
    # 灾难性遗忘：在学习新知识的时候，预训练模型原有的知识会被遗忘
    # 使用比较低的学习率可以缓解灾难性遗忘问题、学习率逐层衰减
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup,
                        t_total=train_steps)

    # 存储每一个batch的loss
    all_loss = []
    total_loss = 0.0
    result = 0.0
    best_result = 0.0

    for epoch in range(1, args.epochs_num+1):
        model.train()
        for step, batch_data in enumerate(train_dataloder):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_seqs_k, batch_seq_masks_k, batch_seq_segments_k, batch_know_emb, batch_labels = batch_data
            # 对标签进行onehot编码
            loss = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_seqs_k, batch_seq_masks_k, batch_seq_segments_k, batch_know_emb, batch_labels)
            loss.backward()
            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.4f}".format(epoch, step+1, total_loss / 100))
                sys.stdout.flush()
                all_loss.append(total_loss)
                total_loss = 0.
            optimizer.step()
            optimizer.zero_grad()

        print("Start evaluation on dev dataset.")
        result = evaluate(args, 'dev')
        if result > best_result:
            best_result = result
            #torch.save(model, open(args.output_model_path,"wb"))
            #save_model(model, args.output_model_path)
            torch.save(model.state_dict(), args.output_model_path)

        print("Start evaluation on test dataset.")
        evaluate(args, 'test')
        print("Start evaluation on the covid dataset.")
        evaluate(args, 'covid')
    

    # Evaluation phase.
    print("Final evaluation on the test dataset.")
    model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, 'test')
    print("Final evaluation on the covid dataset.")
    evaluate(args, 'covid')

    plt.figure(figsize=(12,8))
    plt.plot(range(len(all_loss)), all_loss,'g.')
    plt.grid(True)
    plt.savefig(args.output_lossfig_path)

if __name__ == "__main__":
    main()