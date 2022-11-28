from connectedSegnet_elements import *
import torch
import torch.nn as nn
import torch.functional as F



class ConSegnetsModel(nn.Module):
    def __init__(self,in_channels):
        super(ConSegnetsModel,self).__init__()
        # ###############################
        # First Segnet Encoder Part
        # ###############################
        self.in_channels = in_channels
        self.encoderBlock1 = DoubleConv(in_channels,out_channels=64,encoder=True)
        self.encoderBlock2 = DoubleConv(64,128,encoder=True)
        self.encoderBlock3 = TripleConv(128,256,encoder=True)
        self.encoderBlock4 = TripleConv(256,512,encoder=True)
        self.encoderBlock5 = DoubleConv(512,512,encoder=True)

        # ###############################
        # First Segnet Decoder Part
        # ###############################

        self.decoderBlock1 = TripleConv(512,512)
        self.decoderBlock2 = TripleConv(512,256)
        self.decoderBlock3 = TripleConv(256,128)
        self.decoderBlock4 = DoubleConv(128,64)
        self.decoderBlock5 = nn.Conv2d(64,64,3,padding=1)

        # ###############################
        #For Second Segnet Encoder Part
        #  ##############################

        self.secondFirst = DoubleConv(64,64,encoder=True)
        self.secondEncoder2 = DoubleConv(128,128,encoder=True)
        self.secondEncoder3 = TripleConv(256,256,True)
        self.secondEncoder4 = TripleConv(512,512,True)
        self.secondEncoder5 = TripleConv(1024,1024,True)

        # ################################
        # Second Segnet Decoder Part 
        # ################################

        self.secondDecoder1 = TripleConv(1024,512)
        self.secondDecoder2 = TripleConv(512,256)
        self.secondDecoder3 = TripleConv(256,128)
        self.secondDecoder4 = DoubleConv(128,64)
        self.secondDecoder5 = DoubleConv(64,64)
        self.outconv=nn.Conv2d(64,1,kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,5)
        
       
        
        
   
       


        
        
        self.dilation = dilationConv(64,64)
        self.conv1x1 = conv1x1(in_channels=64,out_channels=1)
    def forward(self,input_image):
#FIRST SEGNET MODEL
        #ENCODER STAGE---->1
        dim_0 = input_image.size()
        out1=self.encoderBlock1(input_image)
        out1,indices_1= maxpooling(out1)
        #ENCODER STAGE----->2
        dim_1 = out1.size()
        out2=self.encoderBlock2(out1)
        out2,indices_2=maxpooling(out2)
        #ENCODER STAGE----->3
        dim_2 = out2.size()

        out3 = self.encoderBlock3(out2)
        out3,indices_3= maxpooling(out3)
        #ENCODER STAGE----->4
        dim_3 = out3.size()
        out4 = self.encoderBlock4(out3)
        out4,indices_4 = maxpooling(out4)
        #ENCODER STAGE----->5
        dim_4 = out4.size()
        out5  = self.encoderBlock5(out4)
        out5,indices_5=maxpooling(out5)
        dim_5 =out5.size()
        #DECODER STAGE----->5
        dec_out1 = unmaxpooling(out5,maxpool_indices=indices_5,dim=dim_4)
        dec_out1 = self.decoderBlock1(dec_out1)
        dec_d1=dec_out1.size()
        #DECODER STAGE----->4
        dec_out2 = unmaxpooling(dec_out1,maxpool_indices=indices_4,dim=dim_3)
        dec_out2 = self.decoderBlock2(dec_out2)
        dec_d2 = dec_out2.size()
        #DECODER STAGE----->3
        dec_out3 = unmaxpooling(dec_out2,maxpool_indices=indices_3,dim=dim_2)

        dec_out3 = self.decoderBlock3(dec_out3)
        dec_d3 = dec_out3.size()
        #DECODER STAGE----->2
        dec_out4 = unmaxpooling(dec_out3,maxpool_indices=indices_2,dim=dim_1)
        dec_out4 = self.decoderBlock4(dec_out4)
        dec_d4 = dec_out4.size()
        #DECODER STAGE----->1
        dec_out5 = unmaxpooling(dec_out4,maxpool_indices=indices_1,dim=dim_0)
        dec_out5 = self.decoderBlock5(dec_out5)
        dec_d5 = dec_out5.size()
#SECOND SEGNET MODEL
        #SECOND ENCODER STAGE---->1

        sec_en_out1 = self.secondFirst(dec_out5)
        sec_en_out1,indices_sec_1 = maxpooling(sec_en_out1)
        sec_en_out1 = cat(sec_en_out1,dec_out4)
        sec_dim_1 = sec_en_out1.size()

        #SECOND ENCODER STAGE---->2
        sec_en_out2 = self.secondEncoder2(sec_en_out1)
        sec_en_out2,indices_sec_2 = maxpooling(sec_en_out2)
        #concate output and third output of decoder
        sec_en_out2=cat(sec_en_out2,dec_out3)
        #finding dimension to upscaling
        sec_dim_2 = sec_en_out2.size()
        #SECOND ENCODER STAGE----->3
        sec_en_out3 = self.secondEncoder3(sec_en_out2)
        sec_en_out3 ,indices_sec_3 = maxpooling(sec_en_out3)
        sec_en_out3 = cat(sec_en_out3,dec_out2)
        sec_dim_3 = sec_en_out3.size()
        #SECOND ENCODER STAGE----->2
        sec_en_out4 = self.secondEncoder4(sec_en_out3)
        sec_en_out4,indices_sec_4 = maxpooling(sec_en_out4)
        sec_en_out4 = cat(sec_en_out4,dec_out1)
        sec_dim_4 = sec_en_out4.size()
        #SECOND ENCODER STAGE----->1
        sec_en_out5 = self.secondEncoder5(sec_en_out4)
        sec_en_out5,indices_sec_5 = maxpooling(sec_en_out5)
        sec_dim_5 = sec_en_out5.size()

        #SECOND DECODER STAGE----->5
        sec_dec_out1 = unmaxpooling(sec_en_out5,maxpool_indices=indices_sec_5,dim=sec_dim_4)
        sec_dec_out1 = self.secondDecoder1(sec_dec_out1)
        sec_dec_d1 = sec_dec_out1.size()
        #SECOND DECODER STAGE----->4
        sec_dec_out2 = unmaxpooling(sec_dec_out1,maxpool_indices=indices_sec_4,dim=sec_dim_3)
        sec_dec_out2 = self.secondDecoder2(sec_dec_out2)
        sec_dec_d2 = sec_dec_out2.size()
        #SECOND DECODER STAGE----->3
        sec_dec_out3 = unmaxpooling(sec_dec_out2,maxpool_indices=indices_sec_3,dim=sec_dim_2)
        sec_dec_out3 = self.secondDecoder3(sec_dec_out3)
        sec_dec_d3 = sec_dec_out3.size()
        #SECOND DECODER STAGE----->2
        sec_dec_out4 = unmaxpooling(sec_dec_out3,maxpool_indices=indices_sec_2,dim=sec_dim_1)
        sec_dec_out4 = self.secondDecoder4(sec_dec_out4)
        sec_dec_d4 = sec_dec_out4.size()
        #SECOND DECODER STAGE----->1
        sec_dec_out5 = unmaxpooling(sec_dec_out4,maxpool_indices=indices_sec_1,dim=dec_d5)
        sec_dec_out5=self.secondDecoder5(sec_dec_out5)
        sec_dec_d5 = sec_dec_out5.size()
        dilation_out= self.dilation(sec_dec_out5)
        out1 = self.conv1x1(dilation_out)


        sec_dec_out5 = self.avg(sec_dec_out5)
        sec_dec_out5 = sec_dec_out5.view(sec_dec_out5.size(0),-1)
        out2 = self.fc1(sec_dec_out5)
        out2 = self.fc2(out2)
        out2 = self.fc3(out2)
        




        return out1,out2




if __name__== "__main__":
    import torchvision.transforms as T
    from PIL import Image
    import cv2 as cv
    import numpy as np
    import pydicom as dicom 

    path = "/Users/okanegemen/yoloV5/INbreast Release 1.0/AllDICOMs/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm"

    dicom_img = dicom.dcmread(path)

    numpy_pixels = dicom_img.pixel_array
    img = np.resize(numpy_pixels,(600,600))
    img = np.array(img,dtype="float32")



    tensor = torch.from_numpy(img)
    tensor = tensor.float()
    tensor = torch.reshape(tensor,[1,1,600,600])
    #tensor = torch.view_as_real(tensor)
    model = ConSegnetsModel(1)

    output1,output2 = model(tensor)

    numpy_img = output1.cpu().detach().numpy()
    print(output2.size())
    print(torch.max(output2))

    numpy_img = np.resize(numpy_img,(600,600))


    image = Image.fromarray(numpy_img,'L')
    image.show()
