import pydicom
from PIL import Image
import numpy as np

path = "/home/alican/Downloads"
name = "f0e2bba9e9409b3532ca57e92676878d.dicom"

name = pydicom.data.data_manager.get_files(path, name)[0]

ds = pydicom.dcmread(name)

ds = np.round((ds.pixel_array/4095)*255)
image = Image.fromarray(ds.astype(np.uint8))

image.save(name.split(".")[0]+".png")