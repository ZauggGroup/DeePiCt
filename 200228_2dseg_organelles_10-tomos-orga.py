# TODO: Check requirements in here!
import os
import random 
import pandas as pd
import numpy as np
# h5py to read the data-set
import h5py
# tensorboard
from keras.callbacks import TensorBoard

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Cropping2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import datetime
import mrcfile
from PatchUtil import *


if tf.test.is_gpu_available(): #config.list_physical_devices('GPU')
    print("GPU is available 😩👌")
else:
    print("GPU is not available 💩")


# ## Data pre-processing

# In[3]:


basepath = '/home/mattausc/mattausc/data/200207_3d_segmentation/'

dataset_paths = [
    "180426_004_zstride-5_size-288.h5",
    "180426_005_zstride-5_size-288.h5",
    "180426_021_zstride-5_size-288.h5",
    "180426_024_zstride-5_size-288.h5",
    "180426_026_zstride-5_size-288.h5",
    "180426_027_zstride-5_size-288.h5",
    "180711_003_zstride-5_size-288.h5",
    "180711_004_zstride-5_size-288.h5",
    "181119_002_zstride-5_size-288.h5",
    "181119_030_zstride-5_size-288.h5"
]

np.random.shuffle(dataset_paths)


# In[4]:


datasets = []

for p in dataset_paths: 
    print(f"Reading {basepath}{p}...")
    with h5py.File(basepath + p, 'r') as f:
        features = f['features'][:]
        labels = f['labels'][:]
        sample_id = f.attrs["sample_id"]
       
        features = np.expand_dims(features, -1)
        labels = np.expand_dims(labels, -1)

        datasets.append([sample_id, features, labels])

all_ids, all_features, all_labels = zip(*datasets)
del datasets


# In[5]:


for dataset_id, dataset in zip(all_ids, all_features):
    print(dataset_id, end="\t")
    mean = dataset.mean()
    std = dataset.std()
    print(f"Before normalization: {mean: .2} +/-{std:.2}", end="\t")

    dataset -= mean
    dataset /= std
    print(f"After normalization: {dataset.mean(): .2} +/-{dataset.std():.2}")
    


# In[6]:


comb_idx = np.hstack([np.full(d.shape[0], i) for i, d in enumerate(all_features)])


# In[7]:


comb_features = np.vstack(all_features)
comb_labels = np.vstack(all_labels)


# In[8]:


# Free up mem when using low z-stride
del all_features, all_labels


# ### Notes
# Perform normalization sample-wise!

# In[9]:


#Each block of u-net architecture consist of two Convolution layers
# These two layers are written in a function to make our code clean
def conv2d_block(input_tensor, n_filters, kernel_size=3, dropout=.2):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               padding="same")(input_tensor)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
               padding="same")(x)
    x = Activation("relu")(x)
    #x = Dropout(dropout)(x)
    return x


# ### TODO: padding

# In[10]:


# The u-net architecture consists of contracting and expansive paths which
# shrink and expands the inout image respectivly. 
# Output image have the same size of input image
def get_unet(input_img, n_filters, target_shape):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*4, kernel_size=3) #The first block of U-net
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*8, kernel_size=3)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*16, kernel_size=3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = conv2d_block(p3, n_filters=n_filters*32, kernel_size=3)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*64, kernel_size=3)
     
    # expansive path
    u6 = Conv2DTranspose(n_filters*32, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*32, kernel_size=3)

    u7 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*16, kernel_size=3)

    u8 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*8, kernel_size=3)

    u9 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters*4, kernel_size=3)
    
    offsets = ((input_img.shape[1] - target_shape[1]) // 2, (input_img.shape[2] - target_shape[2]) // 2)
    #c9 = Cropping2D(offsets)(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# ## Training

# In[11]:


def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps)

def neg_dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -((2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps))


# In[13]:


cv_sets = np.unique(comb_idx)


# ### Cross validation

# In[ ]:



ts = datetime.datetime.strftime(datetime.datetime.now(), "%y%m%d-%H%M")
cv_results = []
lr_orga = 1e-4
bs = 8
epochs_orga = 20
filter_frac = 0 #0.5 # Fraction of all-zero slices to drop
k = 5


for cv_idx, cv_ids in enumerate(np.array_split(cv_sets, k)):
    ##### Cytoplasm ######    
    print(f"{f' CV fold {cv_idx} ':#^30}")

    input_img = Input((comb_features.shape[1], comb_features.shape[2], 1), name='img')
    # Data splitting
    train_ids = [sample_id for idx, sample_id in enumerate(all_ids) if not(idx in cv_ids)]
    train_features = comb_features[~np.isin(comb_idx, cv_ids)]
    train_labels = comb_labels[~np.isin(comb_idx, cv_ids)]
    train_labels[train_labels == 1] = 0
    train_labels[train_labels > 1] = 1

    # Filter out fraction of all-empty patches
    #drop_idx = np.array([np.any(slice) for slice in train_labels]) | (np.random.random(train_labels.shape[0]) > filter_frac)
    #train_features = train_features[drop_idx]
    #train_labels = train_labels[drop_idx]

    test_ids = [sample_id for idx, sample_id in enumerate(all_ids) if idx in cv_ids]
    test_features = comb_features[np.isin(comb_idx, cv_ids)]
    test_labels = comb_labels[np.isin(comb_idx, cv_ids)]
    test_labels[test_labels == 1] = 0
    test_labels[test_labels > 1] = 1

    print("Datasets for training:", *train_ids)
    print("Datasets for testing:", *test_ids)

    # Create model
    model = get_unet(input_img, n_filters=4, target_shape=train_labels.shape)
    model.compile(optimizer=Adam(learning_rate=lr_orga), loss=neg_dice_coefficient, metrics=[dice_coefficient, "binary_crossentropy"])

    # Saving the log and show it by tensorboard
    NAME=f"u-net_organelles_lr-{lr_orga:.0e}_{ts}_zstride-5_dice_CV-{cv_idx}"
    print("TensorBoard name:", NAME)

    tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
    # Fitting the model 
    results = model.fit(train_features, train_labels, batch_size=bs, epochs=epochs_orga, callbacks=[tensorboard],
                        validation_data=(test_features, test_labels))
    
    results.history["phase"] = "organelles"
    results.history["cv_fold"] = cv_idx
    results.history["train_ids"] = train_ids 
    results.history["val_ids"] = test_ids
    results.history["epoch"] = np.arange(epochs_orga)

    cv_results.append(results.history)


# In[32]:


test_ids


# In[16]:


df = []
for res in cv_results:
    df.append({
        "cv_fold":res["cv_fold"],
        "val_dice_coef":res["val_dice_coefficient"][-1],
        "val_ids":res["val_ids"],
        "train_dice_coef":res["dice_coefficient"][-1],
        "train_ids":res["train_ids"],
        "phase":res["phase"]
    })
    
cv_df = pd.DataFrame(df)


# In[17]:


cv_df


# In[20]:


cv_df.to_csv("200228_metrics.csv", index=False, header=True)


# Check out a few predictions...

# In[67]:


def predict_and_show(image, model, pad=48):
    mean = image.mean()
    std = image.std()

    image -= mean
    image /= std
    
    image_patches = np.expand_dims(into_patches(image, (288, 288), (5, 5)), -1)
    image_pred = model.predict(image_patches)
    rec = from_patches(image_pred[...,0], (5, 5), image.shape, pad=pad)
    
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(image[pad:-pad, pad:-pad], cmap="gray", interpolation="bilinear")
    plt.title("Input", loc="left", fontsize=20)
    plt.axis("off")

    plt.subplot(122)
    comp = np.stack([image[pad:-pad, pad:-pad]]*3, -1)
    comp -= comp.min()
    comp /= comp.max()

    comp[...,1] *= 1-rec*0.33
    comp[...,2] *= 1-rec*0.33
    plt.imshow(comp, interpolation="bilinear")
    plt.title("Prediction", loc="left", fontsize=20)
    plt.axis("off")
    plt.tight_layout()


# In[22]:


file = "/struct/mahamid/mattausc/processing/180713/027/etomo/bin4/027_df_sorted.rec"
with mrcfile.open(file) as mrc:
    transfer_feats = mrc.data.copy().astype(np.float32)[255]
    
predict_and_show(transfer_feats, model)


# In[23]:


file = "/struct/mahamid/mattausc/processing/180713/027/etomo/bin4/027_df_sorted.rec"
with mrcfile.open(file) as mrc:
    transfer_feats = mrc.data.copy().astype(np.float32)[250]
    
predict_and_show(transfer_feats, model)


# In[44]:


file = "/struct/mahamid/mattausc/processing/180426/021/etomo/bin4/021_df_sorted.rec"
with mrcfile.open(file) as mrc:
    transfer_feats = mrc.data.copy().astype(np.float32)


# In[68]:


predict_and_show(transfer_feats[480], model)


# In[46]:


tomo = transfer_feats


# In[47]:


tomo.shape


# In[50]:


patch_size = (288, 288)
pad = 48
z_count = 250
z_center = tomo.shape[0] // 2
z_idx = slice(z_center-(z_count // 2), z_center+(z_count // 2))

tomo = tomo[z_idx]

mean = tomo.mean()
std = tomo.std()

tomo -= mean
tomo /= std


# In[51]:


tomo_patches = np.expand_dims(into_patches_3d(tomo, patch_size, (5, 5)), -1) # Add channel dim


# In[52]:


tomo_pred = model.predict(tomo_patches)


# In[53]:


rec = from_patches_3d(tomo_pred[...,0], (5, 5), tomo.shape, pad=pad)


# In[54]:


rec.shape


# In[55]:


mrcfile.new("/home/mattausc/mattausc/data/200226_3d-predictions/180426_021_pred.mrc", data=rec.astype(np.float32), overwrite=True).close()


# In[56]:


500-250/2


# In[ ]:




