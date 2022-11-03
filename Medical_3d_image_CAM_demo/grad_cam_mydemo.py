import numpy as np
from net.UNet_3d import UNet_3d,net
import cv2
import SimpleITK as sitk
import torch
import scipy.ndimage as ndimage
from skimage.transform import resize
from matplotlib import pyplot as plt

###---Load The model----###

# Now suppose you have your trained model and preprocessed ct_tensoruts ready   

ct_path = './data/train_ct/amos_0001.nii.gz'
model_path = './models/model.pth'
# read CT from file
ct = sitk.ReadImage(ct_path, sitk.sitkFloat32)
ct_array = sitk.GetArrayFromImage(ct)

# data preprocess
upper = 350
lower = -350
ct_array[ct_array > upper] = upper
ct_array[ct_array < lower] = lower
ct_array = (ct_array - lower)  / (upper-lower)
ct_tensor = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(0)
# print(ct_tensor.shape)  # (1,1,48,48,48)

ct_tensor = ct_tensor.cuda()

grad_model = net.cuda()

# use register_forward_hook() to gain the features map
class LayerActivations:
    features = None
    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        # 获取model.features中某一层的output
    
    def hook_fn(self, module, ct_tensorut, output):
        self.features = output.cpu()
 
    def remove(self): ## remove hook
        self.hook.remove()
 
# load model
grad_model = torch.nn.DataParallel(UNet_3d(training=False)).cuda()
grad_model.load_state_dict(torch.load(model_path))
grad_model.eval() 
 
# Instantiate, get the i_th layer (second argument) of each convolution
# conv_out = LayerActivations(grad_model.Conv2.conv,0) # train
conv_out = LayerActivations(grad_model.module.Conv2.conv,0) # test

output = grad_model(ct_tensor)
cam = conv_out.features # gain the ith output
# cam = output # gain the latest output
conv_out.remove # delete the hook

###---lAYER-Name--to-visualize--###
# Create a graph that outputs target convolution and output
print('cam.shape1',cam.shape)
cam = cam.cpu().detach().numpy().squeeze()
print('cam.shape2',cam.shape)
cam = cam[1]
print('cam.shape3',cam.shape)

capi=resize(cam,(480,480,480))
#print(capi.shape)
capi = np.maximum(capi,0)
heatmap = (capi - capi.min()) / (capi.max() - capi.min())
f, axarr = plt.subplots(3,3,figsize=(12,12))


f.suptitle('Weight-CAM')
axial_slice_count=25
coronal_slice_count=25
sagittal_slice_count=25
    
axial_ct_img=np.squeeze(ct_array[axial_slice_count, :,:])
axial_grad_cmap_img=np.squeeze(heatmap[axial_slice_count*10,:, :])

sagittal_ct_img=np.squeeze(ct_array[:,:,sagittal_slice_count])
sagittal_grad_cmap_img=np.squeeze(heatmap[:,:,sagittal_slice_count*10]) 

coronal_ct_img=np.squeeze(ct_array[:,coronal_slice_count,:])
coronal_grad_cmap_img=np.squeeze(heatmap[:,coronal_slice_count*10,:]) 


img_plot = axarr[0,0].imshow(axial_ct_img, cmap='gray')
axarr[0,0].axis('off')
axarr[0,0].set_title('axial CT')
    
# axial view
img_plot = axarr[0,1].imshow(axial_grad_cmap_img, cmap='jet')
axarr[0,1].axis('off')
axarr[0,1].set_title('Weight-CAM')

# Zoom in ten times to make the weight map smoother
axial_ct_img = ndimage.zoom(axial_ct_img,  (10,10), order=3)

# Overlay the weight map with the original image
axial_overlay=cv2.addWeighted(axial_ct_img,0.3, axial_grad_cmap_img, 0.6, 0)
    
img_plot = axarr[0,2].imshow(axial_overlay,cmap='jet')
axarr[0,2].axis('off')
axarr[0,2].set_title('Overlay')


# sagittal view
img_plot = axarr[1,0].imshow(sagittal_ct_img, cmap='gray')
axarr[1,0].axis('off')
axarr[1,0].set_title('sagittal CT')
    
img_plot = axarr[1,1].imshow(sagittal_grad_cmap_img, cmap='jet')
axarr[1,1].axis('off')
axarr[1,1].set_title('Weight-CAM')
    

sagittal_ct_img = ndimage.zoom(sagittal_ct_img,  (10,10), order=3)
sagittal_overlay=cv2.addWeighted(sagittal_ct_img,0.3,sagittal_grad_cmap_img, 0.6, 0)


img_plot = axarr[1,2].imshow(sagittal_overlay,cmap='jet')
axarr[1,2].axis('off')
axarr[1,2].set_title('Overlay')

# coronal view
img_plot = axarr[2,0].imshow(coronal_ct_img, cmap='gray')
axarr[2,0].axis('off')
axarr[2,0].set_title('coronal CT')
    
img_plot = axarr[2,1].imshow(coronal_grad_cmap_img, cmap='jet')
axarr[2,1].axis('off')
axarr[2,1].set_title('Weight-CAM')
    

coronal_ct_img = ndimage.zoom(coronal_ct_img,  (10,10), order=3)
Coronal_overlay=cv2.addWeighted(coronal_ct_img,0.3,coronal_grad_cmap_img, 0.6, 0)


img_plot = axarr[2,2].imshow(Coronal_overlay,cmap='jet')
axarr[2,2].axis('off')
axarr[2,2].set_title('Overlay')

# plt.colorbar(img_plot,shrink=0.5) # color bar if need
plt.show()