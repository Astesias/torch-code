# utils.py
# author: ysl
import os
import cv2
import sys
import numpy as np
# import win32com.client as client

class range_percent():
    def __init__(self,total,process_name='Process',obj="█",nonobj='░',ef=True):
        self.total = total
        self.process_name=process_name
        self.obj=obj
        self.nonobj=nonobj
        self.ef=ef
    def update(self,now,new=''):
        precent=now/self.total
        num=int(100*precent)
        sys.stdout.flush()
        print("\r\r\r", end="")
        print("{} {:>3}% |".format(self.process_name,num),self.obj*(num//3),self.nonobj*(33-num//3),'|{}/{}'.format(now,self.total),sep='', end=new)
        if now==self.total and self.ef:
          print()
          self.ef=0
        sys.stdout.flush()


# shell = client.Dispatch("WScript.Shell")
# def GetShortCut(shortcut):    
#     return shell.CreateShortCut(shortcut).Targetpath
# def createShortCut(filename, lnkname):
#     shortcut = shell.CreateShortCut(lnkname)    
#     shortcut.TargetPath = filename    
#     shortcut.save()
# def CreateShortCut(filename, lnkname):
#     createShortCut(os.path.abspath(filename), lnkname)
    
    
    
def cmd(command,log=False):
    import subprocess
    cmd=subprocess.getstatusoutput(command)
    if log:
        print(('Success' if not cmd[0] else 'Fail') + ' Command:\n   '+command)
        print(cmd[1].replace('Active code page: 65001',''))
    if cmd[0] and not log:
        raise Exception(f'cmd order {command} failed')
        
def CreateShortCut2(filename, lnkname):
    with open(lnkname,'w') as fp:
        fp.write(filename)
def GetShortCut2(shortcut):    
    return open(shortcut).read()
    
    
def path2filename(path):
    if type(path)!=type('str'):
        raise TypeError('path is a str,not {}'.format(type(path)))
    if path.rfind('\\')>path.rfind('/'):
        return path[path.rfind('\\')+1:]
    else:
        return path[path.rfind('/')+1:]
    

def cv2_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def easy_show_img(img,rate=1,name=' '):                 
    if type(img)==type(''):
        img=cv2_imread(img)
    if rate and rate!=1:
        h,w,d=img.shape
        img=cv2.resize(img,(int(rate*h),int(rate*w)))
        
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

