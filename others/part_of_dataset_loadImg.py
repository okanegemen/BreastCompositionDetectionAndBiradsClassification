# For crop breast image

# if config.MINIMIZE_IMAGE:
#     transform = T.Compose([
#                 T.PILToTensor()
#                 ])

# img = transform(image)
# _,H,W = img.size()

# ignore = config.IGNORE_SIDE_PIXELS
# temp = img[:,ignore:-ignore,ignore:-ignore]

# _,centerH,centerW = ndi.center_of_mass(temp.detach().cpu().numpy())
# centerH, centerW = int(centerH)+ignore,int(centerW)+ignore
# distance_to_sideR = W - centerW

# if image == "MLO_L":
#     img = img[:,centerH-int(H*0.25):centerH+int(H*0.4),:centerW+int(W*0.3)]
#     _,Hx,Wx = img.size()
#     transform = T.Compose([
#         T.Pad((0,0,int(Hx/1.75)-Wx,0)),
#         T.ToPILImage(),])
#     img = transform(img)

# elif view == "MLO_R":
#     img = img[:,centerH-int(H*0.25):centerH+int(H*0.4),centerW-distance_to_sideR -int(W*0.08):]
#     _,Hx,Wx = img.size()
#     transform = T.Compose([
#         T.Pad((int(Hx/1.75)-Wx,0,0,0)),
#         T.ToPILImage(),])
#     img = transform(img)

# elif view == "CC_L":
#     img = img[:,centerH-int(H*0.3):centerH+int(H*0.3),:centerW+int(W*0.3)]
#     _,Hx,Wx = img.size()
#     transform = T.Compose([
#         T.Pad((0,0,int(Hx/1.75)-Wx,0)),
#         T.ToPILImage(),])

#     img = transform(img)
# elif view == "CC_R":
#     img = img[:,centerH-int(H*0.3):centerH+int(H*0.3),centerW-distance_to_sideR-int(W*0.08):]
#     _,Hx,Wx = img.size()
#     transform = T.Compose([ 
#         T.Pad((int(Hx/1.75)-Wx,0,0,0)),
#         T.ToPILImage(),])
#     img = transform(img)

# else:
#     raise Exception(f"{view} is not an available option for View!")

# img = img[:,centerH-int(H*0.3):centerW+int(H*0.5),:]


            
            # if config.L1regularization:
            #     l1_lambda = 0.001
            #     l1_norm = sum(p.abs().sum()
            #                 for p in model.parameters())

            #     loss_train = loss_train + l1_lambda * l1_norm

            # if config.L2regularization:
            #     l2_lambda = 0.001
            #     l2_norm = sum(p.pow(2.0).sum()
            #                 for p in model.parameters())

            #     loss_train = loss_train + l2_lambda * l2_norm