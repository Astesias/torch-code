# infer.py
# author: ysl
import os
import torch
import torch.nn as nn
from model import AlexNet
from dataloader import dataloader_test,index2types
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from utils import GetShortCut2,range_percent,path2filename,cv2_imread
from pysl import args_sys

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(2520)

def main(path,gpu=False):

    try:
        model_path=GetShortCut2(path)
    except:
        model_path=path
    
    if gpu:
        device = torch.device('cuda')
    
    
    model=torch.nn.DataParallel(AlexNet(num_classes=15), device_ids=[0])
    model.to(device)
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict =torch.load(model_path)['model'] #预训练模型路径
    for k, v in state_dict.items():
    	# 手动添加“module.”
        if 'module' not in k:
            k = 'module.'+k
        else:
        # 调换module和features的位置
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v

    model.load_state_dict(new_state_dict)
    criterion=nn.CrossEntropyLoss().to(device)

    model.eval()
    
    test_loss=0
    correct=0
    total=0
    
    write=True
    writer=SummaryWriter
    
    fp=open('err','a+')
    
    
    
    with torch.no_grad():
        print()
        R=range_percent(len(dataloader_test),'Test')
        for batch_idx,(inputs,targets) in enumerate(dataloader_test):
            inputs=inputs.to(device)
            targets=targets.to(device)
            
            outputs=model(inputs)
            loss=criterion(outputs,targets)
        
            if write:# 绘图
                writer.add_scalar('loss/test', loss,batch_idx)
                writer.add_scalar('true/test', correct/(total+1),batch_idx)
                # writer.add_histogram('para/weight', model.parameters()[0],batch_idx)
                # writer.add_histogram('para/weight', model.bias,batch_idx)
                pass
            
            test_loss+=loss.item()
            _,predicted=outputs.max(1)
            total+=targets.size(0)
            correct+=predicted.eq(targets).sum().item()
            
            if int(targets)!=int(predicted):
              fp.write(f'True {int(targets)} False {int(predicted)}\n')

            R.update(batch_idx,new=' Loss: {:.2f}|Acc: {:.2f}%'.format( test_loss/(batch_idx+1),100.*correct/total))
            
        print('\nCorrect {} Error {}'.format(correct,total-correct))
        
        fp.close()
          
if __name__ == '__main__':
    
    model_path='./pth/latest_epoch.lnk'
    if not args_sys():
      main(GetShortCut2(model_path),(os.name!='nt'))
    else:
      main(args_sys()[0],(os.name!='nt'))
    
    
    
    
    
    
    
    
    
    
    
