
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn


a = [0,1,2,3,4,5]





# print(model.parameters)

# conv = torch.nan 
# module_list = [modules for modules in model.children()]
# buffer = []
# temp = module_list[0]
# while True:
    
#     buffer.append(temp)
#     try:
#         childOrParent = next(iter(buffer[-1].children()))[0]
#         temp = childOrParent
        

#     except StopIteration:
#         conv = temp
#         break

    

model = models.GoogLeNet()

# conv = torch.nan 
# module_list = [modules for modules in model.children()]
# temp = module_list[1:]






# buffer2 = []
# module_list2 = [modules for modules in model.children()]
# temp2 = module_list2[-1]





# while True:
    
#     buffer2.append(temp2)
#     try:
#         childOrParent = next(iter(buffer2[-1].children()))[0]
#         temp2 = childOrParent
        

#     except:
#         linear = temp2
#     break

# try:

#     if len(linear)!=1:
#         linear[1] = nn.Identity()
# except:
#     linear = nn.Identity()
    
# print(linear)



def modifyFirstLayertakeBody(model):
        module_list = [modules for modules in model.children()]

        temp = module_list[0]
        body = module_list[1:-1]
        last = module_list[-1]
        buffer = []

        firstBlock = []

        count = 0

        while True:
            buffer.append(temp)
            try:
                childOrParent = next(iter(buffer[-1].children()))[0]
                temp = childOrParent

                
                count +=1
            except:
                
                if count==1:
                    firstBlock.append(buffer[0][0])
                    break
                    
                firstBlock.append(temp)
                break 


        oneOrTwoDim = 0

        length = 0

        try:
            length=len(firstBlock[0])

            if length!=0:
                oneOrTwoDim = 2
        except TypeError:
            
            oneOrTwoDim = 1





        lookFor = module_list[0]

        try:

            if len(lookFor)!=0:
                firstBody = []
                firstBody.append(lookFor[1:])
        except:
            firstBody = torch.nan
            
        print(firstBlock)

        try:
            if oneOrTwoDim == 1:




                firstBlock[0] = nn.Conv2d(256,
                                            firstBlock[0].out_channels,
                                            firstBlock[0].kernel_size,
                                            firstBlock[0].stride,
                                            firstBlock[0].padding,
                                            firstBlock[0].dilation,
                                            firstBlock[0].groups,
                                            bias=firstBlock[0].bias)

            else: 
                firstBlock[0][0] = nn.Conv2d(256,
                                                firstBlock[0][0].out_channels,
                                                firstBlock[0][0].kernel_size,
                                                firstBlock[0][0].stride,
                                                firstBlock[0][0].padding,
                                                firstBlock[0][0].dilation,
                                                firstBlock[0][0].groups,
                                                bias=firstBlock[0][0].bias)
                
        except:
            firstBlock[0].conv = nn.Conv2d(256,
                                            firstBlock[0].conv.out_channels,
                                            firstBlock[0].conv.kernel_size,
                                            firstBlock[0].conv.stride,
                                            firstBlock[0].conv.padding,
                                            firstBlock[0].conv.dilation,
                                            firstBlock[0].conv.groups,
                                            firstBlock[0].conv.bias
                                            )
        



                    

        return firstBlock,firstBody,body

def changeLastlayer(model):
    buffer2 = []
    module_list2 = [modules for modules in model.children()]
    temp2 = module_list2[-1]





    while True:
        
        buffer2.append(temp2)
        try:
            childOrParent = next(iter(buffer2[-1].children()))[0]
            temp2 = childOrParent
            

        except:
            linear = temp2
        break
    new_buffer = []
    try:
        
        if len(linear)!=1:
            new_buffer.append(linear[1])
            linear[1] = nn.Identity()
    except:
        new_buffer.append(linear)
        linear = torch.nan


    return new_buffer,linear



firstBlock,firstBody,body = modifyFirstLayertakeBody(model)
# module_list = [modules for modules in model.children()]

# temp = module_list[0]
# body = module_list[1:-1]
# last = module_list[-1]
# buffer = []

# firstBlock = []

# count = 0

# while True:
#     buffer.append(temp)
#     try:
#         childOrParent = next(iter(buffer[-1].children()))[0]
#         temp = childOrParent

        
#         count +=1
#     except:
        
#         if count==1:
#             firstBlock.append(buffer[0][0])
#             break
            
#         firstBlock.append(temp)
#         break 


# oneOrTwoDim = 0

# length = 0

# try:
#     length=len(firstBlock[0])

#     if length!=0:
#         oneOrTwoDim = 2
# except TypeError:
    
#     oneOrTwoDim = 1





# lookFor = module_list[0]

# try:

#     if len(lookFor)!=0:
#         firstBody = []
#         firstBody.append(lookFor[1:])
# except:
#     print("First Module have just one element")



# try:
#     if oneOrTwoDim == 1:




#         firstBlock[0] = nn.Conv2d(256,
#                                     firstBlock[0].out_channels,
#                                     firstBlock[0].kernel_size,
#                                     firstBlock[0].stride,
#                                     firstBlock[0].padding,
#                                     firstBlock[0].dilation,
#                                     firstBlock[0].groups,
#                                     bias=firstBlock[0].bias)

#     else: 
#         firstBlock[0][0] = nn.Conv2d(256,
#                                         firstBlock[0][0].out_channels,
#                                         firstBlock[0][0].kernel_size,
#                                         firstBlock[0][0].stride,
#                                         firstBlock[0][0].padding,
#                                         firstBlock[0][0].dilation,
#                                         firstBlock[0][0].groups,
#                                         bias=firstBlock[0][0].bias)
# except:
#     firstBlock[0].conv = nn.Conv2d(256,
#                                     firstBlock[0].conv.out_channels,
#                                     firstBlock[0].conv.kernel_size,
#                                     firstBlock[0].conv.stride,
#                                     firstBlock[0].conv.padding,
#                                     firstBlock[0].conv.dilation,
#                                     firstBlock[0].conv.groups,
#                                     firstBlock[0].conv.bias
#                                     )


            

# print(len(firstBlock))


                


            





  









