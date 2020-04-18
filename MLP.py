#Utils librairies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

#Data loading
X=pd.read_csv("dataX.csv")
y=pd.read_csv("datay.csv")
print(X.shape)

#Distribution of target variable
y_columns=y.columns
x = y[y_columns[0]].value_counts().values
sns.barplot([0,1], x)
plt.title("Distribution of target variable")

# Data normalization
scaler = MinMaxScaler(feature_range = (0, 1))
X = scaler.fit_transform(X)

# Train test split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_val=y_val.iloc[:,:].values
print(len(y_val))

#Data augmentation
ran=RandomOverSampler()
X_tr,y_tr= ran.fit_resample(X_tr,y_tr)
print(X_tr.shape)
print(y_tr.shape)
x = np.array([len(y_tr[y_tr==0]),len(y_tr[y_tr==1])])
sns.barplot([0,1], x)
plt.title("Répartition de la variable cible ")

# Pytorch dataloader
class Dataset(Dataset):
    def __init__(self, X,y):
        self.X=torch.from_numpy(X).float()
        self.y=torch.tensor(y, dtype=torch.long)
        self.sample=len(y)

    def __len__(self):
        return self.sample

    def __getitem__(self, item):
        return self.X[item],self.y[item]

training_set=Dataset(X_tr,y_tr)
test_set=Dataset(X_val,y_val)
training_generator=DataLoader(training_set, batch_size=32, shuffle=True)
test_generator=DataLoader(test_set, batch_size=32, shuffle=True)

#Build network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1=nn.Linear(in_features=6,out_features=10)
        self.bn1=nn.BatchNorm1d(num_features=10)
        self.fc2=nn.Linear(in_features=10,out_features=10)
        self.bn2=nn.BatchNorm1d(num_features=10)
        self.fc3=nn.Linear(in_features=10,out_features=6)
        self.bn3=nn.BatchNorm1d(num_features=6)
        self.fc4=nn.Linear(in_features=6,out_features=2)
    def forward(self,x):
        x=F.tanh(self.bn1(self.fc1(x)))
        x=F.tanh(self.bn2(self.fc2(x)))
        x=F.tanh(self.bn3(self.fc3(x)))
        x=F.log_softmax((self.fc4(x)),dim=1)
        
        return x

Net=Network()
print(Net)

#Network Training
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9,weight_decay= 1e-6)
optimizer = torch.optim.Adam(Net.parameters(), lr=1e-5)
epochs=500
PATH="best_weigths"
test_interval=1
best_loss = 1e10
best_epoch = 0
Net.train()
num_iter_per_epoch = len(training_generator)
test_loss=[]
train_loss=[]
for epoch in range(epochs):
    epoch_loss=[]
    for iter, batch in enumerate(training_generator):
        x, label = batch
        x = Variable(x, requires_grad=True)
        optimizer.zero_grad()
        logits = Net(x)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss*len(label))
    tr_loss=sum(epoch_loss)/len(training_set)
    train_loss.append(tr_loss)
    if epoch % test_interval == 0:
        Net.eval()
        loss_ls = []
        for te_iter, te_batch in enumerate(test_generator):
            te_x, te_label = te_batch
            num_sample = len(te_label)
            with torch.no_grad():
                te_logits = Net(te_x)
            batch_loss= criterion(te_logits, torch.max(te_label, 1)[1])
            loss_ls.append(batch_loss * num_sample)
        te_loss = sum(loss_ls) / test_set.__len__()
        print("Epoch: {}/{} Train_loss:{:.2f} Test_loss:{:.2f}".format(
                epoch + 1,
                epochs,
                tr_loss,
                te_loss
                ))
        test_loss.append(te_loss)
        Net.train()
        if te_loss < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(Net.state_dict(), PATH)
                
# Training curves
fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
axs[0].plot(np.arange(epochs), train_loss)
axs[0].set_title("Train loss")
axs[1].plot(np.arange(epochs), test_loss)
axs[1].set_title("Test loss")

#Weights loading
Net.load_state_dict(torch.load(PATH))
Net.eval()

# Inference
x,target=test_set[:]
y_pred=Net(x)
y_pred=torch.argmax(y_pred, dim=1)
cm=confusion_matrix(target,y_pred)
PMC_accuracy=(cm[0][0]+cm[1][1])/sum(sum(cm))
PMC_roc=roc_auc_score(target,y_pred)
print("Acurracy score:",PMC_accuracy)
print("ROC score:",PMC_roc)
print("confusion matrix:")
print(cm)

#Comaparaison with other machine learning methods

classifiers=[RandomForestClassifier(),GradientBoostingClassifier(),
             AdaBoostClassifier(),DecisionTreeClassifier(),
             KNeighborsClassifier(),GaussianProcessClassifier()]
names=["RandomForestClassifier","GradientBoostingClassifier",
       "AdaBoostClassifier","DecisionTreeClassifier",
       "KNeighborsClassifier","GaussianProcessClassifier"]
Acuracy=[]
ROC=[]

for i in range(len(classifiers)):
    print(names[i]+":")
    classifier=classifiers[i]
    classifier.fit(X_tr,y_tr)
    y_pred=classifier.predict(X_val)
    cm=confusion_matrix(y_val,y_pred)
    accuracy=(cm[0][0]+cm[1][1])/sum(sum(cm))
    roc=roc_auc_score(y_val,y_pred)
    Acuracy.append(accuracy)
    ROC.append(roc)
    print("Acurracy score:",accuracy)
    print("ROC score:",roc)
    print("confusion matrix:")
    print(cm)
    print("="*20)
names.append("PMC")
Acuracy.append(PMC_accuracy)
ROC.append(PMC_roc)
d={"classifiers":names,"accuracy":Acuracy,"roc score":ROC}  
df=pd.DataFrame(data=d)    
print(df)

