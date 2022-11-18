import matplotlib.pyplot as plt
import pydicom
import pydicom.data
import numpy as np
from torch import nn
import torch

# Full path of the DICOM file is passed in base
base = r"/home/alican/Documents/yoloV5/sample"
pass_dicom = "22580244_5530d5782fc89dd7_MG_R_ML_ANON.dcm"  # file name is 1-12.dcm
  
# enter DICOM image name for pattern
# result is a list of 1 element
filename = pydicom.data.data_manager.get_files(base, pass_dicom)[0]
  
ds = pydicom.dcmread(filename)
data1 = ds.pixel_array # normal
normalize = data1/data1.max()
data = nn.Sigmoid()((nn.Sigmoid()(torch.tensor(normalize))**45)*20000000) # normalized
print(data1.max())

# fig = plt.figure(figsize=(10,10))
# rows=1
# cols = 2

# fig.add_subplot(rows,cols, 1)
# plt.imshow(data, cmap=plt.cm.bone)  # set the color map to bone
# plt.title("normalized")

# fig.add_subplot(rows,cols,2)
# plt.imshow(data1,cmap=plt.cm.bone)
# plt.title("normal")

# plt.show()
