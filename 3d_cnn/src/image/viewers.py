from os.path import join

import h5py
import matplotlib.pyplot as plt


def view_slice(data, slice_id, fig_size=10, columns=1, labels=None):
    plt.figure(figsize=(fig_size, fig_size))
    if columns == 1:
        plt.subplot(221)
        plt.imshow(data[slice_id], cmap='gray')
        plt.title("Input Slice %i" % slice_id)
        plt.xlabel("x - dataset coords (z,y,x)")
        plt.ylabel("y - dataset coords (z,y,x)")
        plt.show()
    if columns == 2:
        plt.subplot(221)
        plt.imshow(data[slice_id], cmap='gray')
        plt.title("Input Slice %i" % slice_id)
        plt.xlabel("x - dataset coords (z,y,x)")
        plt.ylabel("y - dataset coords (z,y,x)")
        plt.subplot(222)
        plt.imshow(labels[slice_id], cmap='gray')
        plt.title("Labels Slice %i" % slice_id)
        plt.show()


def view_images_h5(data_path: str, img_range: tuple, cathegory="ribosomes"):
    with h5py.File(data_path, 'r') as f:
        raw_subtomo_names = list(f['volumes/raw'])
        img_init, img_end = img_range
        for subtomo_name in raw_subtomo_names[img_init:img_end]:
            raw_subtomo_h5_internal_path = join('volumes/raw', subtomo_name)
            data_subtomo = f[raw_subtomo_h5_internal_path][:]
            labels_subtomo_h5_internal_path = join('volumes/labels',
                                                   cathegory)
            labels_subtomo_h5_internal_path = join(
                labels_subtomo_h5_internal_path,
                subtomo_name)
            label = f[labels_subtomo_h5_internal_path][:]
            print(subtomo_name)
            view_slice(data_subtomo, 0, 3, 2, label)
            view_slice(data_subtomo, 32, 3, 2, label)
            view_slice(data_subtomo, 63, 3, 2, label)
