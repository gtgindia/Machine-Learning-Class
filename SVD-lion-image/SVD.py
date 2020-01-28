
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


ny_file = "lion.jpg"
img = Image.open(ny_file)
img.show()


# In[9]:


img.size


# In[10]:


red_band =img.getdata(band=0)


# In[11]:


img_mat = np.array(list(red_band), float) 
img_mat.size


# In[12]:


# get image shape
img_mat.shape = (img.size[1], img.size[0])
# conver to 1d-array to matrix
img_mat = np.matrix(img_mat)
img_mat


# In[13]:


fig, axs = plt.subplots(1, 2,figsize=(10,10))
axs[0].imshow(img)
axs[0].set_title('Original Image', size=16)
axs[1].imshow(img_mat)
axs[1].set_title(' "R" band image', size=16)
plt.tight_layout()
plt.savefig('Original_image_and_R_band_image_for_SVD.jpg',dpi=150)


# In[15]:


#Let us center and scale the data before applying SVD. This will help us put each variable in the same scale.
 
# scale the image matrix befor SVD
img_mat_scaled= (img_mat-img_mat.mean())/img_mat.std()


# In[16]:


# Perform SVD using np.linalg.svd
U, s, V = np.linalg.svd(img_mat_scaled) 


# In[17]:


U.shape


# In[18]:


V.shape


# In[51]:


s.shape


# In[20]:


# Compute Variance explained by each singular vector
var_explained = np.round(s**2/np.sum(s**2), decimals=3)


# In[21]:


var_explained[0:20]


# In[23]:


sns.barplot(x=list(range(1,21)),
            y=var_explained[0:20], color="dodgerblue")
plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.savefig('svd_scree_plot.png',dpi=150, figsize=(8,6))
plt.savefig("Line_Plot_with_Pandas_Python.jpg")


# In[57]:


var_explained[0:80].sum()/var_explained[::].sum()


# In[43]:


num_components = 5
reconst_img_5 = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) * np.matrix(V[:num_components, :])
plt.imshow(reconst_img_5)
plt.title('Reconstructed Image: 5 SVs', size=16)
plt.savefig('reconstructed_image_with_5_SVs.png',dpi=150, figsize=(8,6))


# In[39]:


num_components = 20
reconst_img_20 = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) * np.matrix(V[:num_components, :])
plt.imshow(reconst_img_20)
plt.title('Reconstructed Image: 20 SVs', size=16)
plt.savefig('reconstructed_image_with_20_SVs.png',dpi=150, figsize=(8,6))


# In[58]:


num_components = 79
reconst_img_79 = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) * np.matrix(V[:num_components, :])
plt.imshow(reconst_img_79)
plt.title('Reconstructed Image: 79 SVs', size=16)
plt.savefig('reconstructed_image_with_79_SVs.png',dpi=150, figsize=(8,6))


# In[59]:


num_components = 1000
reconst_img_1000 = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) * np.matrix(V[:num_components, :])
plt.imshow(reconst_img_1000)
plt.title('Reconstructed Image: 1000 SVs', size=16)
plt.savefig('reconstructed_image_with_1000_SVs.png',dpi=150, figsize=(8,6))


# In[60]:


fig, axs = plt.subplots(2, 2,figsize=(10,10))
axs[0, 0].imshow(reconst_img_5)
axs[0, 0].set_title('Reconstructed Image: 5 SVs', size=16)
axs[0, 1].imshow(reconst_img_20)
axs[0, 1].set_title('Reconstructed Image: 20 SVs', size=16)
axs[1, 0].imshow(reconst_img_79)
axs[1, 0].set_title('Reconstructed Image: 79 SVs', size=16)
axs[1, 1].imshow(reconst_img_1000)
axs[1, 1].set_title('Reconstructed Image: 1000 SVs', size=16)
plt.tight_layout()
plt.savefig('reconstructed_images_using_different_SVs.jpg',dpi=150)

