#Util librairies
!pip install pytorchcv
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import seaborn as sn
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#Load dataset
train_data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
training_set = datasets.ImageFolder(root='../input/covid19xray/covid-19-dataset/train',
                                           transform=train_data_transform)

val_data_transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_set = datasets.ImageFolder(root='../input/covid19xray/covid-19-dataset/test',
                               transform=val_data_transform)

dataloaders = {'train':  torch.utils.data.DataLoader(training_set,
                                             batch_size=64, shuffle=True),
               'val':torch.utils.data.DataLoader(test_set,
                                             batch_size=64, shuffle=True)}

dataset_sizes = {'train': len(training_set) ,'val':len(test_set)}
class_names = training_set.classes  
print(class_names)


#data exploration
label_0=[0 for i in range(len(training_set)) if training_set[i][1] ==0 ]
label_1=[1 for i in range(len(training_set)) if training_set[i][1] ==1 ]
label_2=[2 for i in range(len(training_set)) if training_set[i][1] ==2]
plt.hist(label_0,histtype='bar',color='red',label=class_names[0])
plt.hist(label_1,histtype='bar',color='green',label=class_names[1])
plt.hist(label_2,histtype='bar',color='orange',label=class_names[2])
plt.legend(prop={'size': 10})
plt.title("Distribution of training set target variable ")

image_1,label_1=training_set[0]
image_2,label_2=training_set[1]
image_3,label_3=training_set[289]
image_4,label_4=training_set[289+1]
image_5,label_5=training_set[289+1095]
image_6,label_6=training_set[289+1095+1]
images=[image_1,image_2,image_3,image_4,image_5,image_6]
labels=[label_1,label_2,label_3,label_4,label_5,label_6]
plt.figure(figsize=(10,6))
for i in range(6):
    image=images[i]
    image=np.transpose(image,(1,2,0))
    plt.subplot(2,3,i+1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[labels[i]])


# Use transfert learning

#Util functions

def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses={'train':[],'val':[]}
    accuracy={'train':[],'val':[]}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for iter, batch in enumerate(dataloaders[phase]):
                inputs,labels=batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            losses[phase]=losses[phase]+[epoch_loss]
            accuracy[phase]=accuracy[phase]+[epoch_acc]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return {"model":model,'losses':losses,'accuracy':accuracy}


def tensorbord(epoch,results):
    epochs=np.arange(1,epoch+1)
    losses=results["losses"]
    accuracy=results["accuracy"]
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(epochs,losses["train"],label="train",color="orange")
    plt.plot(epochs,losses["val"],label="val",color="green")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train/Val losses")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs,accuracy["train"],label="train",color="orange")
    plt.plot(epochs,accuracy["val"],label="val",color="green")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Train/Val accuracy")
    plt.legend()
    return

def visualize_prediction(model, num_images=9):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12,11))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            scores = F.softmax(outputs,dim=1)
            cfs, _ = torch.max(scores, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                plt.subplot(3,3,images_so_far)
                image=inputs.cpu().data[j]
                image=np.transpose(image,(1,2,0))
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel("actual: "+str(class_names[labels[j]]),color="green")
                plt.title('predicted: {} {}%'.format(class_names[preds[j]],round(float(100*cfs[j]),1)),
                          color="orange")
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
    
def evaluate(model):
    model.eval()
    data=torch.utils.data.DataLoader(test_set,batch_size=len(test_set), shuffle=True)
    with torch.no_grad():
            for iter,batch in enumerate(data):
                images,labels=batch
                images=images.to(device)
                labels=labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                print(classification_report(labels.cpu(),  preds.cpu(), target_names=class_names))
                cm=confusion_matrix(labels.cpu(),preds.cpu())
                df_cm = pd.DataFrame(cm, index = class_names,columns = class_names)
                plt.figure(figsize = (10,5))
                sn.heatmap(df_cm, annot=True)
               
    return 


# First model resnet18
resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 3)
resnet=resnet.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001, weight_decay=0.0005)
#optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
results = train_model(resnet, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=100)
resnet=results["model"]
torch.save(resnet, "resnet-covid-19-model.pth")
torch.save(resnet.state_dict(),"resnet-covid-19-weigths.pth" )

tensorbord(100,results)

visualize_prediction(model=resnet, num_images=9)

evaluate(resnet)


#2nd model darknet53
darknet53 = ptcv_get_model("darknet53", pretrained=True)
darknet53.output=nn.Linear(1024, 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
darknet53=darknet53.to(device)
optimizer = torch.optim.Adam(darknet53.parameters(), lr=0.001, weight_decay=0.0005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

results_darknet53 = train_model(darknet53, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=100)
darknet53=results_darknet53["model"]
torch.save(darknet53, "darknet53-covid-19-model.pth")
torch.save(darknet53.state_dict(),"darknet53-covid-19-weigths.pth" )

tensorbord(100,results_darknet53)

visualize_prediction(model=darknet53, num_images=9)

evaluate(darknet53)


