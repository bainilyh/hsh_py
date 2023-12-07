# _*_ coding:utf-8 _*_

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from random import sample
import torch.nn as nn
import torch.optim as optim
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from transformers import BertTokenizerFast, BertForMaskedLM
# from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

data_base_path = r"./data/cc0546df-0ef9-4b12-97a0-dd4eb08f2e0c.csv"


class TitleDataset(Dataset):
    def __init__(self, mode, label2id=None, test_number=500):
        super(TitleDataset, self).__init__()
        self.sentences = []
        self.labels = []
        self.label2id = label2id

        if mode == "train":
            with open(data_base_path, encoding='UTF-8') as f:
                for line in f:
                    line_list = line.rsplit(',', 1)
                    if len(line_list) < 2:
                        continue
                    category_num = line_list[1].strip('"|\n')
                    if category_num not in self.label2id:
                        continue
                    self.sentences.append(line_list[0].strip('"|\n'))
                    _label_id = self.label2id[category_num]
                    self.labels.append(_label_id)
        if mode == "test":
            with open(data_base_path, encoding='UTF-8') as f:
                _sentences = []
                for line in f:
                    _sentences.append(line)
            _sentences = sample(_sentences, test_number)
            for line in _sentences:
                line_list = line.rsplit(',', 1)
                if len(line_list) < 2:
                    continue
                category_num = line_list[1].strip('"|\n')
                if category_num not in self.label2id:
                    continue
                self.sentences.append(line_list[0])
                _label_id = self.label2id[category_num]
                self.labels.append(_label_id)

    def __getitem__(self, idx):
        sentences = self.sentences[idx]
        labels = self.labels[idx]
        return sentences, labels

    def __len__(self):
        return len(self.sentences)


class BertClassificationModel(nn.Module):
    def __init__(self, hidden_size=784, category_num=2):
        super(BertClassificationModel, self).__init__()
        model_name = './model/bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        # self.bert_config = BertConfig.from_pretrained(model_name, num_labels=category_num)
        # self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(model_name, 'pytorch_model.bin'), config=self.bert_config)
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = F.relu
        self.fc2 = nn.Linear(128, category_num)
        if torch.cuda.device_count() >= 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.ret = 0

    def forward(self, batch_sentences):
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=512,
                                             add_special_tokens=True,
                                             return_tensors='pt')
        input_ids = sentences_tokenizer['input_ids'].to(self.device)
        attention_mask = sentences_tokenizer['attention_mask'].to(self.device)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_out[0]
        if not self.ret:
            self.ret = 1
            print(last_hidden_state.shape)
        bert_cls_hidden_state = last_hidden_state[:, 0, :]  # batch_size, len, embedding_size
        fc_out = self.fc2(self.relu(self.fc1(bert_cls_hidden_state)))
        return fc_out


def label2id(base_path):
    label_2_id = dict()
    with open(base_path, encoding='UTF-8') as f:
        label_id = 0
        for line in f:
            line_list = line.rsplit(',', 1)
            if len(line_list) < 2:
                continue
            category_num = line_list[1].strip('"|\n')
            try:
                int(category_num)
            except:
                continue
            if category_num not in label_2_id:
                label_2_id[category_num] = label_id
                label_id += 1
    return label_2_id


def main():
    test_number = 500
    batch_size = 512
    label_ids = label2id(data_base_path)
    print(f'label_ids is {label_ids}')
    train_data = TitleDataset(label2id=label_ids, mode="train")
    test_data = TitleDataset(label2id=label_ids, mode="test", test_number=test_number)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print('training...')

    epoch_num = 100
    model = BertClassificationModel(hidden_size=768, category_num=len(label_ids))
    device_str = 'cuda' if torch.cuda.device_count() >= 1 else 'cpu'
    model.to(device_str)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("模型数据已经加载完成,现在开始模型训练。")
    for epoch in range(epoch_num):
        epoch_loss = 0
        epoch_acc = 0
        # for i, (data, labels) in tqdm(enumerate(train_loader, 0), mininterval=2, desc='----train', leave=False):
        for i, (data, labels) in tqdm(enumerate(train_loader, 0), mininterval=2, desc='----train', leave=False):
            output = model(data)
            optimizer.zero_grad()
            labels = labels.to(device_str)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print('batch:%d loss:%.5f' % (i, loss.item()))
            acc = (output.argmax(dim=1) == labels).sum().item()
            epoch_acc += acc
            epoch_loss += loss.item() * len(data)

        val_acc = 0
        val_loss = 0
        model.eval()
        for j, (data, labels) in enumerate(val_loader, 0):
            output = model(data)
            out = output.argmax(dim=1)
            labels = labels.to(device_str)
            loss = criterion(output, labels)
            val_loss += loss.item() * len(data)
            val_acc += (out == labels).sum().item()
        print(
            f'''Epochs: {epoch + 1} 
            | Train Loss: {epoch_loss / len(train_data): .3f} 
            | Train Accuracy: {epoch_acc / len(train_data): .3f} 
            | Val Loss: {val_loss / len(test_data): .3f} 
            | Val Accuracy: {val_acc / len(test_data): .3f}
            '''
        )


if __name__ == '__main__':
    main()
