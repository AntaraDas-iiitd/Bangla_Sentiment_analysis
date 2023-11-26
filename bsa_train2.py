from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from copy import deepcopy
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE=torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')

def make_model(model_path):
  bert_model=AutoModel.from_pretrained(model_path)
  return bert_model

def make_tokenizer(model_path):
  bert_tokenizer=AutoTokenizer.from_pretrained(model_path)
  return bert_tokenizer

def split_data(data):
  df_train ,df_test = train_test_split(data, test_size=0.15, random_state=101)
  return df_train ,df_test


def tokenize_data(data_df,tokenizer,label_dict):
  SENT_COL='Data'
  LAB_COL='Sentiment'
  sents=list(data_df[SENT_COL].values)
  labels=list(data_df[LAB_COL].values)
  tokenized=tokenizer(sents,truncation=True,padding=True,max_length=256)
  tokenized['input_ids']=torch.tensor([t for t in tokenized['input_ids']],dtype=torch.long,device=DEVICE)
  tokenized['attention_mask']=torch.tensor([t for t in tokenized['attention_mask']],dtype=torch.long,device=DEVICE)
  labels_num=[label_dict[lb] for lb in labels]    #convert the labels to numerical values, then convert that list to tensor
  tokenized['labels']=torch.tensor([lb for lb in labels_num],dtype=torch.long,device=DEVICE)
  return tokenized

def make_data_loader(processed_data,train):
  ts_data=TensorDataset(processed_data['input_ids'],processed_data['attention_mask'],processed_data['labels'])
  if train:
    return DataLoader(ts_data,shuffle=True,batch_size=44)
  else:
    return DataLoader(ts_data,shuffle=False,batch_size=16)

def make_optimizer(model):
  for param in model.bert_model.parameters():
    param.requires_grad=False
  list(model.bert_model.parameters())[-1].requires_grad=True
  optimizer = torch.optim.Adam(
      [
      {"params":model.lin_layer.parameters(), "lr":3e-4},
      {"params":model.bert_model.parameters(), "lr":2e-5}
      ])
  return optimizer


def make_criterion():
  criterion=nn.CrossEntropyLoss()
  return criterion

class MakeModel(nn.Module):
  def __init__(self,model,k):
    super(MakeModel,self).__init__()
    self.bert_model=model
    self.lin_layer=nn.Linear(768,k)


  def forward(self,input_ids,attention_mask):
    out_vect=self.bert_model(input_ids=input_ids,attention_mask=attention_mask)
    lin_op=self.lin_layer(out_vect.last_hidden_state[:,0,:])
    return F.softmax(lin_op)


def train_epoch(train_loader,model,optimizer,loss_fn):
  epoch_loss=0
  for step,batch in enumerate(train_loader):
    optimizer.zero_grad()
    batch=tuple(t.to(DEVICE) for t in batch)
    input_ids,attention_mask,labels=batch
    out_val=model(input_ids,attention_mask)
    #print(out_val)
    loss=loss_fn(out_val,labels)
    epoch_loss+=loss.item()
    loss.backward()
    optimizer.step()
  return epoch_loss/step

def val_epoch(val_loader,model,loss_fn):
  epoch_loss=0
  with torch.no_grad():
    for step,batch in enumerate(val_loader):
      # optimizer.zero_grad()
      batch=tuple(t.to(DEVICE) for t in batch)
      input_ids,attention_mask,labels=batch
      out_val=model(input_ids,attention_mask)
      #print(out_val)
      loss=loss_fn(out_val,labels)
      epoch_loss+=loss.item()
      # loss.backward()
      # optimizer.step()
    return epoch_loss/step

def test(test_loader,model):
    true_label=list()
    pred_label=list()
    with torch.no_grad():
        for step,batch in tqdm(enumerate(test_loader)):
            batch=tuple(t.to(DEVICE) for t in batch)
            input_id,attention_mask,labels=batch
            out_val=model(input_id,attention_mask)
            out_labels=torch.argmax(out_val,-1)
            true_label.extend(labels.detach().cpu().numpy().tolist())
            pred_label.extend(out_labels.detach().cpu().numpy().tolist())
        #print(round(accuracy_score(true_label,pred_label),3))
        target_names = ['Negative', 'Positive', 'Neutral']
        print(classification_report(true_label, pred_label, target_names=target_names))


def train_func(train_loader,val_loader,model):
  loss_fn=make_criterion()
  optimizer=make_optimizer(model)
  EPOCHS=50
  train_losses=list()
  val_losses=list()
  best_model, min_loss = None, 1000000
  for i in tqdm(range(EPOCHS)):
    train_loss=train_epoch(train_loader,model,optimizer,loss_fn)
    val_loss=val_epoch(val_loader,model,loss_fn)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss < min_loss:
      best_model = deepcopy(model)
      min_loss = val_loss
    print(f'epoch:{i}   train_loss: {round(train_loss,3)}  val_loss:{round(val_loss,3)}')
  return best_model,train_losses,val_losses



if __name__=="__main__":
    # get the model ready for training
    model_path='csebuetnlp/banglabert'        # set it with the pretrained bert model
    bert_model=make_model(model_path)
    bert_tokenizer=make_tokenizer(model_path)
    k=3     #fill it for number of labels in your data
    fin_model=MakeModel(bert_model,k)
    fin_model.to(DEVICE)
    # get the data ready
    df_train = pd.read_excel('/home/vikram/Antara/Bangla_Sentiment/train_v3.xlsx')
    train_data, val_data = split_data(df_train)
    uniq_lb = list(set(list(df_train['Sentiment'])))  #create the label_dict all the possible labels in the data
    label_dict = {lb:i for i,lb in enumerate(uniq_lb)}
    processed_train=tokenize_data(train_data,bert_tokenizer,label_dict)
    train_loader=make_data_loader(processed_train,True)
    processed_val=tokenize_data(val_data,bert_tokenizer,label_dict)
    val_loader=make_data_loader(processed_val,False)
    # train and save the model
    model_trained,train_losses,val_losses=train_func(train_loader,val_loader,fin_model)
    PATH = '/home/vikram/Antara/Bangla_Sentiment/bsenti_model3'
    torch.save(model_trained, f'{PATH}.pt')