import os
import nibabel as nib
import matplotlib.pyplot as plt

img_path = "/Users/postgres/Desktop/phd/project/adenomyosis/data/SK40339275D3stimNormalpelvis.nii.gz"

img = nib.load(img_path)
data = img.get_fdata()
plt.imshow(data[:, :, 0], cmap='gray')
plt.show()

names = [""]