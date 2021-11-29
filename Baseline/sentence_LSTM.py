### THIS CODE IS TAKEN FROM https://github.com/arthtalati/Deep-Learning-based-Authorship-Identification/blob/master/sentence_gru_lstm.ipynb
### Baseline Accuracies

import pandas as pd
import numpy as np
import os
import dill
import pandas as pd
import glob, csv
import nltk
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import re
import string
import seaborn as sns
from sklearn.preprocessing import StandardScaler

train_file_df = pd.read_csv("train.csv")
test_file_df = pd.read_csv("test.csv")

auth_sort = sorted(train_file_df['Author'].unique())
dictOfAuthors = { i : auth_sort[i] for i in range(0, len(auth_sort) ) }
swap_dict = {value:key for key, value in dictOfAuthors.items()}
train_file_df['Author_num'] = train_file_df['Author'].map(swap_dict)

auth_sort = sorted(test_dataframe['Author'].unique())
dictOfAuthors = { i : auth_sort[i] for i in range(0, len(auth_sort) ) }
swap_dict = {value:key for key, value in dictOfAuthors.items()}
test_dataframe['Author_num'] = test_dataframe['Author'].map(swap_dict)
test_file_df = test_file_df.drop(columns='Author')
train_file_df = train_file_df.drop(columns='Author')

list_to_choose_test = test_file_df.text.apply(lambda x : len(x)) > 0 
test_file_df_choosen = test_file_df[list_to_choose_test]

list_to_choose_train = train_file_df.text.apply(lambda x : len(x)) > 0 
train_file_df_choosen = train_file_df[list_to_choose_train]

train_file_df_choosen.to_csv('sentence_train.csv', index = False)
test_file_df_choosen.to_csv('sentence_test.csv', index = False)

import torch
from torchtext.legacy import data
TEXT = data.Field(sequential=True, tokenize="spacy", lower=True, include_lengths=True)
SCORE = data.Field(sequential=False, use_vocab=False)

datafields = [("text", TEXT),
              ("Author_num", SCORE)]

train = data.TabularDataset(
    path='sentence_train.csv', 
    format='csv',fields=datafields,skip_header = True)

val = data.TabularDataset(
    path='sentence_test.csv', 
    format='csv',fields=datafields,skip_header = True)

from torchtext import vocab
from torchtext.vocab import GloVe
TEXT.build_vocab(train, val, min_freq = 3, vectors=GloVe(name='6B', dim=100))

device = torch.device('cuda:0')
BATCH_SIZE = 64
train_iterator = data.BucketIterator(
    train, 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True,
    repeat=False, 
    shuffle=True,
    device = device)

val_iterator = data.BucketIterator(
    val, 
    batch_size = BATCH_SIZE,
    sort=False,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True,
    repeat=False, 
    shuffle=False,
    device = device)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib
import pandas as pd
import dill
import random
import torch.optim as optim
import matplotlib.pyplot as plt

class AuthorClassifier(nn.Module):

    def __init__(self, mode, output_size, hidden_size, vocab_size, embedding_length, word_embeddings):
      super(AuthorClassifier, self).__init__()

      if mode not in ['rnn', 'lstm', 'gru', 'bilstm']:
        raise ValueError("Choose a mode from - rnn / lstm / gru / bilstm")

      self.mode = mode
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.vocab_size = vocab_size
      self.embedding_length = embedding_length
      self.embedding = nn.Embedding(self.vocab_size,self.embedding_length)
      self.embedding.weight = nn.Parameter(word_embeddings,requires_grad = False)

 
      if self.mode == 'rnn':
        self.network = nn.RNN(self.embedding_length,self.hidden_size)
      elif self.mode == 'lstm':
        self.network = nn.LSTM(self.embedding_length,self.hidden_size)
      elif self.mode == 'gru':
        self.network = nn.GRU(self.embedding_length,self.hidden_size)
      elif self.mode == 'bilstm':
        self.network = nn.LSTM(self.embedding_length,self.hidden_size,bidirectional = True)
      self.fclayer = nn.Linear(self.hidden_size,self.output_size)
      

    def forward(self, text, text_lengths):
      text_embeddings = self.embedding(text)
      pack_sequence = nn.utils.rnn.pack_padded_sequence(text_embeddings,text_lengths.to('cpu'))

      if self.mode in ('lstm','bilstm'):
        a,(hidden,cell) = self.network(pack_sequence)
        if self.mode == 'bilstm':
          hidden = hidden[0,:,:]+ hidden[1,:,:]
      else:
        a,hidden = self.network(pack_sequence) 
      hidden = hidden.squeeze(0)
      pred = self.fclayer(hidden)
      return pred

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

def train_classifier(model, dataset_iterator, loss_function, optimizer, num_epochs, log = "runs", verbose = False, recurrent = True):
  writer = SummaryWriter(log_dir=log)
  model.train()
  step = 0
  f1score_train = []
  loss_train = []
  accuracy_train = []
  loss_total = []


  for epoch in range(num_epochs):
    correct = 0
    total = 0
    total_loss = 0
    f1 = 0
    f1_step = 0
    
    
    for batch in dataset_iterator:
      comment, comment_lengths = batch.text
      labels = batch.Author_num

      batch_size = len(labels)
      optimizer.zero_grad()
      output = model(comment, comment_lengths).squeeze(0)

      loss = loss_function(output, labels.long())
      loss.backward() 
      nn.utils.clip_grad_norm_(model.parameters(),0.5)
      optimizer.step()

      pred = torch.max(output.data,1).indices
      f1 += sklearn.metrics.f1_score((labels.cpu()).numpy(), (pred.cpu()).numpy(),average= 'macro')
      correct += (torch.sum(pred == labels)).item()
      total += len(labels)
      total_loss += loss.item()
      loss_total.append(total_loss/total)
      f1_step += 1

      if ((step % 100) == 0):
        writer.add_scalar("Loss/train", total_loss/total, step)
        writer.add_scalar("Acc/train", correct/total, step)
        writer.add_scalar("F1 Score/train", f1/f1_step, step)
        
      step = step+1

    f1score_train.append(f1/f1_step)
    accuracy_train.append(correct/total)
    loss_train.append(total_loss/total)

    print('---Training---',"Epoch: %s Acc: %s Loss: %s"%(epoch+1, correct/total, total_loss/total),'F1 Score:',f1/f1_step,)

  return loss_train,f1score_train,accuracy_train

def evaluate_classifier(model, dataset_iterator, loss_function, recurrent = True):
  model.eval()

  correct = 0
  total = 0
  total_loss = 0
  overall_pred = []
  overall_label = []
  accuracy_test = []
  loss_test = []

  f1_step = 0
  f1 = 0

  for batch in dataset_iterator:
    comment, comment_lengths = batch.text
    labels = batch.Author_num
    output = model(comment, comment_lengths).squeeze(0)
    loss = loss_function(output, labels.long())
    pred = torch.max(output.data,1).indices 
    correct += (torch.sum(pred == labels)).item()
    total += len(labels)
    total_loss += loss.item()
    ap = pred.cpu()
    a = np.asarray(ap)
    labels = labels.cpu()
    b = np.asarray(labels)
    f1_step += 1
    overall_pred.append(a)
    overall_label.append(b)

  overall_p= [val for sublist in overall_pred for val in sublist]
  overall_l = [val for sublist in overall_label for val in sublist]
  f1ss = sklearn.metrics.f1_score(overall_l,overall_p,average= 'macro')
  accuracy_test.append(correct/total)
  loss_test.append(total_loss/total)
  print("Validation statistics: Acc: %s Loss: %s"%(correct/total, total_loss/total),'F1 Score:',f1ss)
  return overall_pred,overall_label,accuracy_test,loss_test,f1ss

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

output_size = 50
hidden_size = 300
vocab_size = len(TEXT.vocab)
embedding_length = 100
word_embeddings = TEXT.vocab.vectors
num_epochs = 1
mode = 'lstm'

model = AuthorClassifier(mode, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model = model.to(device)


loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.25*1e-2)
log_dir = 'runs/lstm1'
final_acc_train_lstm = []
final_loss_train_lstm = []
final_loss_test_lstm = []
final_acc_test_lstm = []
final_f1score_train_lstm = []
final_f1score_test_lstm = []


for multi in range(20):
  loss_train,f1score_train,accuracy_train = train_classifier(model, train_iterator, loss_function, optimizer, log = log_dir, num_epochs = num_epochs)
  overall_pred,overall_label,accuracy_test,loss_test,f1ss = evaluate_classifier(model, val_iterator, loss_function)
  final_acc_train_lstm.append(accuracy_train[0])
  final_acc_test_lstm.append(accuracy_test[0])
  final_f1score_train_lstm.append(f1score_train[0])
  final_f1score_test_lstm.append(f1ss)
  final_loss_train_lstm.append(loss_train[0])
  final_loss_test_lstm.append(loss_test[0])


cf = np.zeros((50,50))

overall_pred = [val for sublist in overall_pred for val in sublist]

overall_label = [val for sublist in overall_label for val in sublist]

ziplist = list(zip(overall_label,overall_pred))
for coordinate in ziplist:
  cf[coordinate]+=1

plt.figure(figsize = (20,20))
ax = sns.heatmap(cf,annot=True)

import plotly.graph_objects as go
fig_accuracy = go.Figure()

fig_accuracy.add_trace(go.Scatter(
    y=final_acc_train_lstm,
    connectgaps=True, marker_color='rgba(128, 0, 0, 0.9)', name = 'Training accuracy lstm'))

fig_accuracy.add_trace(go.Scatter(
    y=final_acc_test_lstm,
    connectgaps=True, marker_color='rgba(255, 0, 0, 0.9)', name = 'Testing accuracy lstm'))

fig_accuracy.show()