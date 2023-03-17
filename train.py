# train.py
# author: ysl
import os
import traceback
import torch
import torch.nn as nn
from model import AlexNet
from dataloader import dataloader_train
from torch.utils.tensorboard import SummaryWriter
from utils import CreateShortCut2,GetShortCut2,path2filename,range_percent
from pysl import args_sys
from infer import main as infers


os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def main(rep=None,gpu=False):
    
    if gpu:
        device = torch.device('cuda')
        model1=torch.nn.DataParallel(AlexNet(num_classes=15), device_ids=[0])
    else:
        model1=AlexNet(num_classes=15)
        
    save_path='./pth/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    lr=0.01
    resume_epoch=0
    
    if gpu:
        model1.to(device)
        criterion=nn.CrossEntropyLoss().to(device)
    else:
        criterion=nn.CrossEntropyLoss()
    
    optimizer=torch.optim.SGD(model1.parameters(),
                              lr=lr,momentum=0.9,
                              weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)                         
                                
    try:
      if not rep or len(rep)==0:
        model_path=GetShortCut2(save_path+'latest_epoch.lnk')
      else:
        model_path=rep[0]
      print('Load from {} resume into epoch {}'.
                format(path2filename(model_path),int(model_path.split('_')[-1])+1))
      resume_epoch=int(model_path.split('_')[-1])+1
      save_info=torch.load(model_path)
      model1.load_state_dict(save_info['model'])
      optimizer.load_state_dict(save_info['optimizer'])
      # scheduler.load_state_dict(save_info['scheduler'])
    except:  
      print(traceback.print_exc())
      print('No model for resume')
        
    save_info={
    'optimizer':optimizer.state_dict(),
    'model':model1.state_dict(),
    # 'scheduler':scheduler.state_dict()
    }
    write=True
    writer=SummaryWriter()
    
    model1.train()
    
    epoch=200
    for epoch_idx in range(epoch):
        
        train_loss=0
        correct=0
        total=0
        
        print('\nepoch {}:'.format(epoch_idx+resume_epoch))
        R=range_percent(len(dataloader_train),'Train')
        for batch_idx,(inputs,targets) in enumerate(dataloader_train):
        
            ###
            if gpu:
                inputs=inputs.to(device)
                targets=targets.to(device)
            
            optimizer.zero_grad()
            outputs=model1(inputs)
            loss=criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            
            train_loss+=loss.item()
            _,predicted=outputs.max(1)
            total+=targets.size(0)
            correct+=predicted.eq(targets).sum().item()
            
            if write:# 绘图
                writer.add_scalar('loss/train', loss,batch_idx)
                writer.add_scalar('true/train', correct/(total+1),batch_idx)
                 
            R.update(batch_idx+1,new=' Loss: {:.3f}|Acc: {:.3f}%'.
                     format( train_loss/(batch_idx+1),100.*correct/total) )
        
        
        # scheduler.step()

        torch.save(save_info,save_path+'model_epoch_'+str(epoch_idx+resume_epoch))
        CreateShortCut2(save_path+'model_epoch_'+str(epoch_idx+resume_epoch),
                                       save_path+'latest_epoch.lnk')
        
        infers(GetShortCut2('./pth/latest_epoch.lnk'),(os.name!='nt'))
        
if __name__ == '__main__':
    main(args_sys(),(os.name!='nt'))

    
    
    
    
    
    
    
    
    
    