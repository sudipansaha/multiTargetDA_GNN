import os
import glob
import random
import rasterio
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
from preprocessInput import preprocessSentinel2DividedBy10000

# indices of sentinel-2 bands related to land
S2_BANDS_LD = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
S2_BANDS_RGB = [2, 3, 4] # B(2),G(3),R(4)


# util function for reading s2 data
def load_s2(path, imgTransform, s2_band): 
    bands_selected = s2_band
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    if not imgTransform:
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
    s2 = s2.astype(np.float32)
    return s2


# util function for reading s1 data
def load_s1(path, imgTransform):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    if not imgTransform:
        s1 /= 25
        s1 += 1
    s1 = s1.astype(np.float32)
    return s1
    

# util function for reading data from single sample
def load_sample(sample, labels, label_type, threshold, imgTransform, use_s1, use_s2, use_RGB, IGBP_s):

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_LD)
    # load only RGB   
    if use_RGB and use_s2==False:
        img = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_RGB)
        
    # load s1 data
    if use_s1:
        if use_s2 or use_RGB:
            img = np.concatenate((img, load_s1(sample["s1"], imgTransform)), axis=0)
        else:
            img = load_s1(sample["s1"], imgTransform)
            
    # load label
    lc = labels[sample["id"]]
    
    # covert label to IGBP simplified scheme
    if IGBP_s:
        cls1 = sum(lc[0:5]);
        cls2 = sum(lc[5:7]); 
        cls3 = sum(lc[7:9]);
        cls6 = lc[11] + lc[13];
        lc = np.asarray([cls1, cls2, cls3, lc[9], lc[10], cls6, lc[12], lc[14], lc[15], lc[16]])
        
    if label_type == "multi_label":
        lc_hot = (lc >= threshold).astype(np.float32)     
    else:
        loc = np.argmax(lc, axis=-1)
        lc_hot = np.zeros_like(lc).astype(np.float32)
        lc_hot[loc] = 1
             
    rt_sample = {'image': img, 'label': lc_hot, 'id': sample["id"]}
    
    if imgTransform is not None:
        rt_sample = imgTransform(rt_sample)
    
    return rt_sample


#  calculate number of input channels  
def get_ninputs(use_s1, use_s2, use_RGB):
    n_inputs = 0
    if use_s2:
        n_inputs += len(S2_BANDS_LD)
    if use_s1:
        n_inputs += 2
    if use_RGB and use_s2==False:
        n_inputs += 3
        
    return n_inputs


# class SEN12MS..............................
class SEN12MS(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""
    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val/test split and can be obtained from:
    # https://github.com/MSchmitt1984/SEN12MS/

    def __init__(self, path, ls_dir=None, imgTransform=None, 
                 label_type="multi_label", threshold=0.1, subset="train",
                 use_s2=False, use_s1=False, use_RGB=False, IGBP_s=True):
        """Initialize the dataset"""

        # inizialize
        super(SEN12MS, self).__init__()
        self.imgTransform = imgTransform
        self.threshold = threshold
        self.label_type = label_type

        # make sure input parameters are okay
        if not (use_s2 or use_s1 or use_RGB):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2, s1, RGB] to True!")
        self.use_s2 = use_s2
        self.use_s1 = use_s1
        self.use_RGB = use_RGB
        self.IGBP_s = IGBP_s
        
        assert subset in ["train", "val", "test"]
        assert label_type in ["multi_label", "single_label"] # new !!
        
        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2, use_RGB)

        # provide number of IGBP classes 
        if IGBP_s == True:
            self.n_classes = 10
        else:
            self.n_classes = 17 

        # make sure parent dir exists
        assert os.path.exists(path)
        assert os.path.exists(ls_dir)



#-------------------------------- import split lists--------------------------------
        if label_type == "multi_label" or label_type == "single_label":
            # find and index samples
            self.samples = []
            
            pathList = glob.glob(data_dir+'*.tif')
            sample_list = []
            for pathName in pathList:
                fileName = os.path.basename(pathName)
                sample_list.append(fileName)
            
            
            # if subset == "train":
            #     file =os.path.join(ls_dir, 'train_list.pkl')
            #     sample_list = pkl.load(open(file, "rb"))
            #     print(sample_list)
            #     x = 10/0
        
                

            
            #
          
            
            for s2_id in sample_list:
                
                s2_loc = os.path.join(path, s2_id)
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                
               
                self.samples.append({"s1": s1_loc, "s2": s2_loc, 
                                     "id": s2_id})
       

#----------------------------------------------------------------------               
        
        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the sen12ms subset", subset)
        
        # import lables as a dictionary
        label_file = os.path.join(ls_dir,'IGBP_probability_labels.pkl')

        a_file = open(label_file, "rb")
        self.labels = pkl.load(a_file)
        a_file.close()
        

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        labels = self.labels
        return load_sample(sample, labels, self.label_type, self.threshold, self.imgTransform, 
                           self.use_s1, self.use_s2, self.use_RGB, self.IGBP_s)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)




#%% data normalization

class Normalize(object):
    def __init__(self, bands_mean, bands_std):
        
        self.bands_s1_mean = bands_mean['s1_mean']
        self.bands_s1_std = bands_std['s1_std']

        self.bands_s2_mean = bands_mean['s2_mean']
        self.bands_s2_std = bands_std['s2_std']
        
        self.bands_RGB_mean = bands_mean['s2_mean'][0:3]
        self.bands_RGB_std = bands_std['s2_std'][0:3]
        
        self.bands_all_mean = self.bands_s2_mean + self.bands_s1_mean
        self.bands_all_std = self.bands_s2_std + self.bands_s1_std

    def __call__(self, rt_sample):

        img, label, sample_id = rt_sample['image'], rt_sample['label'], rt_sample['id']

        # different input channels
        if img.size()[0] == 12:
            for t, m, s in zip(img, self.bands_all_mean, self.bands_all_std):
                t.sub_(m).div_(s) 
        elif img.size()[0] == 10:
            for t, m, s in zip(img, self.bands_s2_mean, self.bands_s2_std):
                t.sub_(m).div_(s)          
        elif img.size()[0] == 5:
            for t, m, s in zip(img, 
                               self.bands_RGB_mean + self.bands_s1_mean,
                               self.bands_RGB_std + self.bands_s1_std):
                t.sub_(m).div_(s)                                
        elif img.size()[0] == 3:
            for t, m, s in zip(img, self.bands_RGB_mean, self.bands_RGB_std):
                t.sub_(m).div_(s)
        else:
            for t, m, s in zip(img, self.bands_s1_mean, self.bands_s1_std):
                t.sub_(m).div_(s)            
        
        return {'image':img, 'label':label, 'id':sample_id}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, rt_sample):
        
        img, label, sample_id = rt_sample['image'], rt_sample['label'], rt_sample['id']
        
        rt_sample = {'image': torch.tensor(img), 'label':label, 'id':sample_id}
        return rt_sample



def processDirectory(data_dir,list_dir,imgTransform,label_type,subset,season,maxImagePerClass,classesUsedForCreatedDataset):
    ds = SEN12MS(data_dir, list_dir, imgTransform, 
                 label_type, subset, use_s1=False, use_s2=False, use_RGB=True, IGBP_s=True)
    numImage = ds.__len__()
    for imageIter in range(numImage):
        imageItem = ds.__getitem__(imageIter)
        imageLabelOneHot = imageItem["label"]
        imageLabel = (np.argwhere(imageLabelOneHot)[0][0])+1
        if imageLabel not in classesUsedForCreatedDataset:
            continue
        
        imageRGB = (imageItem["image"]).numpy()
        imageRGB = np.transpose(imageRGB,axes=[1,2,0])
        imageRGB = preprocessSentinel2DividedBy10000(imageRGB)
        
        imageIDWithExtension = imageItem['id']
        imageID = (imageIDWithExtension.rsplit(".tif",1))[0]
        savePath = './multiseasonDataset/'+season+'/'+str(imageLabel)+'/'+imageID+'.png'
        
        filesInSavePath = os.listdir('./multiseasonDataset/'+season+'/'+str(imageLabel)+'/')
        
        fileCountInSavePath = len(filesInSavePath)
        if fileCountInSavePath==maxImagePerClass:
            continue
       
        cv2.imwrite(savePath,(imageRGB*255).astype('uint8'))
        
        f= open('./multiseasonDataset/'+season+'.txt',"a+")
        classNumberForTextFile = classesUsedForCreatedDataset.index(imageLabel)
        textToWrite = str(imageLabel)+'/'+imageID+'.png '+str(classNumberForTextFile)+'\n'
        f.write(textToWrite)
        f.close()



#%%...........................................................................
# DEBUG usage examples
if __name__ == "__main__":
    
    
    classesUsedForCreatedDataset =[1,2,3,4,6,7,10]
    maxImagePerClass = 1000
    
    f= open('./multiseasonDataset/'+'summer'+'.txt',"w+")
    f.close()
    f= open('./multiseasonDataset/'+'spring'+'.txt',"w+")
    f.close()
    f= open('./multiseasonDataset/'+'fall'+'.txt',"w+")
    f.close()
    f= open('./multiseasonDataset/'+'winter'+'.txt',"w+")
    f.close()
   
    
    # Summer
    dataDirRoot = "./ROIs1868_summer_s2/ROIs1868_summer/"    # SEN12MS dir
    seasonWithRoi = (dataDirRoot.rsplit("_s2",1))[0]
    season = seasonWithRoi.rsplit('_',1)[1]
    f= open('./multiseasonDataset/'+season+'.txt',"a+")
    f.close()
    list_dir = './labels_splits/'
    subFolders = [ f.path for f in os.scandir(dataDirRoot) if f.is_dir() ]
    for subdirIter in range(min(len(subFolders),60)):
        data_dir = subFolders[subdirIter]+'/'
        # define image transform
        imgTransform=transforms.Compose([ToTensor()])
        label_type="single_label"
        subset="train"
        processDirectory(data_dir,list_dir,imgTransform,label_type,subset,season,maxImagePerClass,classesUsedForCreatedDataset)
    
        
   
    
    
    ##Spring
    dataDirRoot = "./ROIs1158_spring_s2/ROIs1158_spring/"    # SEN12MS dir
    seasonWithRoi = (dataDirRoot.rsplit("_s2",1))[0]
    season = seasonWithRoi.rsplit('_',1)[1]
    f= open('./multiseasonDataset/'+season+'.txt',"a+")
    f.close()
    list_dir = './labels_splits/'
    subFolders = [ f.path for f in os.scandir(dataDirRoot) if f.is_dir() ]
    for subdirIter in range(min(len(subFolders),60)):
        data_dir = subFolders[subdirIter]+'/'
        # define image transform
        imgTransform=transforms.Compose([ToTensor()])
        label_type="single_label"
        subset="train"
        processDirectory(data_dir,list_dir,imgTransform,label_type,subset,season,maxImagePerClass,classesUsedForCreatedDataset)
    
       
    
    
    ##Fall
    dataDirRoot = "./ROIs1970_fall_s2/ROIs1970_fall/"    # SEN12MS dir
    seasonWithRoi = (dataDirRoot.rsplit("_s2",1))[0]
    season = seasonWithRoi.rsplit('_',1)[1]
    f= open('./multiseasonDataset/'+season+'.txt',"a+")
    f.close()
    list_dir = './labels_splits/'
    subFolders = [ f.path for f in os.scandir(dataDirRoot) if f.is_dir() ]
    for subdirIter in range(min(len(subFolders),80)):
        data_dir = subFolders[subdirIter]+'/'
        # define image transform
        imgTransform=transforms.Compose([ToTensor()])
        label_type="single_label"
        subset="train"
        processDirectory(data_dir,list_dir,imgTransform,label_type,subset,season,maxImagePerClass,classesUsedForCreatedDataset)
        
      
    
    
    ###Winter
    dataDirRoot = "./ROIs2017_winter_s2/ROIs2017_winter/"    # SEN12MS dir
    seasonWithRoi = (dataDirRoot.rsplit("_s2",1))[0]
    season = seasonWithRoi.rsplit('_',1)[1]
    f= open('./multiseasonDataset/'+season+'.txt',"a+")
    f.close()
    list_dir = './labels_splits/'
    subFolders = [ f.path for f in os.scandir(dataDirRoot) if f.is_dir() ]
    for subdirIter in range(min(len(subFolders),60)):
        data_dir = subFolders[subdirIter]+'/'
        # define image transform
        imgTransform=transforms.Compose([ToTensor()])
        label_type="single_label"
        subset="train"
        processDirectory(data_dir,list_dir,imgTransform,label_type,subset,season,maxImagePerClass,classesUsedForCreatedDataset)
    
   
    
    

    






