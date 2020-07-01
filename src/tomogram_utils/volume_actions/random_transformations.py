from os.path import join

import h5py
import numpy as np
from scipy.ndimage import interpolation
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

import python_utils_inferno as pyu
from constants import h5_internal_paths
from file_actions.readers import h5

"""
All these functions were taken from the package Inferno:
https://github.com/inferno-pytorch/
"""


# from .base import Transform

class Transform(object):
    """
    Base class for a Transform. The argument `apply_to` (list) specifies the
    indices of the tensors this transform will be applied to.
    The following methods are recognized (in order of descending priority):
        - `batch_function`: Applies to all tensors in a batch simultaneously
        - `tensor_function`: Applies to just __one__ tensor at a time.
        - `volume_function`: For 3D volumes, applies to just __one__ volume at
        a time.
        - `image_function`: For 2D or 3D volumes, applies to just __one__ image
        at a time.
    For example, if both `volume_function` and `image_function` are defined,
    this means that only the former will be called. If the inputs are therefore
    not 5D batch-tensors of 3D volumes, a `NotImplementedError` is raised.
    """

    def __init__(self, apply_to=None):
        """
        Parameters
        ----------
        apply_to : list or tuple
            Indices of tensors to apply this transform to. The indices are with respect
            to the list of arguments this object is called with.
        """
        self._random_variables = {}
        self._apply_to = list(apply_to) if apply_to is not None else None

    def build_random_variables(self, **kwargs):
        pass

    def clear_random_variables(self):
        self._random_variables = {}

    def get_random_variable(self, key, default=None, build=True,
                            **random_variable_building_kwargs):
        if key in self._random_variables:
            return self._random_variables.get(key, default)
        else:
            if not build:
                return default
            else:
                self.build_random_variables(**random_variable_building_kwargs)
                return self.get_random_variable(key, default, build=False)

    def set_random_variable(self, key, value):
        self._random_variables.update({key: value})

    def __call__(self, *tensors, **transform_function_kwargs):
        print("len(tensors)", len(tensors))
        print("tensors[0].shape", tensors[0].shape)
        # FIXME here this should be taken:
        tensors = tensors[0]
        tensors = pyu.to_iterable(tensors)  # this sets tensors = [tensors]
        print("len(tensors)", len(tensors))
        # Get the list of the indices of the tensors to which we're going to
        # apply the transform
        apply_to = list(
            range(len(tensors))) if self._apply_to is None else self._apply_to
        print(apply_to)
        # Flush random variables and assume they're built by image_function
        self.clear_random_variables()
        if hasattr(self, 'batch_function'):
            transformed = self.batch_function(tensors,
                                              **transform_function_kwargs)
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'tensor_function'):
            transformed = [
                self.tensor_function(tensor, **transform_function_kwargs)
                if tensor_index in apply_to else tensor
                for tensor_index, tensor in enumerate(tensors)]
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'volume_function'):
            # Loop over all tensors
            print("tensors shape", tensors[0].shape)
            transformed = [
                self._apply_volume_function(tensor, **transform_function_kwargs)
                if tensor_index in apply_to else tensor
                for tensor_index, tensor in
                enumerate(tensors)]
            transformed = np.array(transformed)
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'image_function'):
            # Loop over all tensors
            transformed = [
                self._apply_image_function(tensor, **transform_function_kwargs)
                if tensor_index in apply_to else tensor
                for tensor_index, tensor in enumerate(tensors)]
            return pyu.from_iterable(transformed)
        else:
            raise NotImplementedError

    # noinspection PyUnresolvedReferences
    def _apply_image_function(self, tensor, **transform_function_kwargs):
        assert pyu.has_callable_attr(self, 'image_function')
        # 2D case
        if tensor.ndim == 4:
            return np.array([np.array(
                [self.image_function(image, **transform_function_kwargs)
                 for image in channel_image])
                for channel_image in tensor])
        # 3D case
        elif tensor.ndim == 5:
            return np.array([np.array([np.array([self.image_function(image,
                                                                     **transform_function_kwargs)
                                                 for image in volume])
                                       for volume in channel_volume])
                             for channel_volume in tensor])
        elif tensor.ndim == 3:
            # Assume we have a 3D volume (signature zyx) and apply the image function
            # on all yx slices.
            return np.array(
                [self.image_function(image, **transform_function_kwargs)
                 for image in tensor])
        elif tensor.ndim == 2:
            # Assume we really do have an image.
            return self.image_function(tensor, **transform_function_kwargs)
        else:
            raise NotImplementedError

    # noinspection PyUnresolvedReferences
    def _apply_volume_function(self, tensor, **transform_function_kwargs):
        assert pyu.has_callable_attr(self, 'volume_function')
        # 3D case
        if tensor.ndim == 5:
            # tensor is bczyx
            # volume function is applied to zyx, i.e. loop over b and c
            # FIXME This loops one time too many
            print("we are in case tensor.ndim == 5")
            return np.array([np.array([np.array([self.volume_function(volume,
                                                                      **transform_function_kwargs)
                                                 for volume in channel_volume])
                                       for channel_volume in batch])
                             for batch in tensor])
        elif tensor.ndim == 4:
            # We're applying the volume function on a czyx tensor, i.e. we loop over c and apply
            # volume function to (zyx)
            return np.array(
                [self.volume_function(volume, **transform_function_kwargs)
                 for volume in tensor])
        elif tensor.ndim == 3:
            # We're applying the volume function on the volume itself
            return self.volume_function(tensor, **transform_function_kwargs)
        else:
            raise NotImplementedError


class RandomFlip3D(Transform):
    def __init__(self, **super_kwargs):
        super(RandomFlip3D, self).__init__(**super_kwargs)

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('flip_lr', np.random.uniform() > 0.5)
        self.set_random_variable('flip_ud', np.random.uniform() > 0.5)
        self.set_random_variable('flip_z', np.random.uniform() > 0.5)

    def volume_function(self, volume):
        if self.get_random_variable('flip_lr'):
            volume = volume[:, :, ::-1]
        if self.get_random_variable('flip_ud'):
            volume = volume[:, ::-1, :]
        if self.get_random_variable('flip_z'):
            volume = volume[::-1, :, :]
        return volume


class RandomRot3D(Transform):
    def __init__(self, rot_range=30, p=0.5, **super_kwargs):
        super(RandomRot3D, self).__init__(**super_kwargs)
        self.rot_range = rot_range
        self.p = p

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('angle_z', np.random.uniform(-self.rot_range,
                                                              self.rot_range))
        p_rotate = np.random.uniform(low=0, high=1, size=1)[0]
        self.set_random_variable('p_rotate', p_rotate)

    def volume_function(self, volume):
        # angle_z = self.get_random_variable('angle_z')
        angle_z = self.rot_range
        if self.get_random_variable('p_rotate') < self.p:
            volume = interpolation.rotate(volume, angle_z, order=0,
                                          mode='reflect', axes=(1, 2),
                                          reshape=False)
        return volume


class ElasticTransform(Transform):
    NATIVE_DTYPES = {'float32', 'float64'}
    PREFERRED_DTYPE = 'float32'

    def __init__(self, alpha, sigma, order=1, invert=False,
                 **super_kwargs):
        self._initial_dtype = None
        super(ElasticTransform, self).__init__(**super_kwargs)
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.invert = invert

    def cast(self, image):
        if image.dtype not in self.NATIVE_DTYPES:
            self._initial_dtype = image.dtype
            image = image.astype(self.PREFERRED_DTYPE)
        return image

    def uncast(self, image):
        if self._initial_dtype is not None:
            image = image.astype(self._initial_dtype)
        self._initial_dtype = None
        return image

    def build_random_variables(self, **kwargs):
        # All this is done just once per batch (i.e. until `clear_random_variables` is called)
        np.random.seed()
        imshape = kwargs.get('imshape')
        # Build and scale random fields
        random_field_x = np.random.uniform(-1, 1, imshape) * self.alpha
        random_field_y = np.random.uniform(-1, 1, imshape) * self.alpha
        # Smooth random field (this has to be done just once per reset)
        sdx = gaussian_filter(random_field_x, self.sigma, mode='reflect')
        sdy = gaussian_filter(random_field_y, self.sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
        # Make inversion coefficient
        _inverter = 1. if not self.invert else -1.
        # Distort meshgrid indices (invert if required)
        flow_y, flow_x = (y + _inverter * sdy).reshape(-1, 1), (
                x + _inverter * sdx).reshape(-1, 1)
        # Set random states
        self.set_random_variable('flow_x', flow_x)
        self.set_random_variable('flow_y', flow_y)

    def image_function(self, image):
        # Cast image to one of the native dtypes (one which that is supported by scipy)
        image = self.cast(image)
        # Take measurements
        imshape = image.shape
        # Obtain flows
        flows = self.get_random_variable('flow_y', imshape=imshape), \
                self.get_random_variable('flow_x', imshape=imshape)
        # Map cooordinates from image to distorted index set
        transformed_image = map_coordinates(image, flows,
                                            mode='reflect',
                                            order=self.order).reshape(imshape)
        # Uncast image to the original dtype
        transformed_image = self.uncast(transformed_image)
        return transformed_image


class ElasticTransform3D(Transform):
    """Random Elastic Transformation 3D. Modified by Irene"""
    NATIVE_DTYPES = {'float32', 'float64'}
    PREFERRED_DTYPE = 'float32'

    def __init__(self, alpha, sigma, order=1, invert=False,
                 **super_kwargs):
        self._initial_dtype = None
        super(ElasticTransform3D, self).__init__(**super_kwargs)
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.invert = invert

    def cast(self, image):
        if image.dtype not in self.NATIVE_DTYPES:
            self._initial_dtype = image.dtype
            image = image.astype(self.PREFERRED_DTYPE)
        return image

    def uncast(self, image):
        if self._initial_dtype is not None:
            image = image.astype(self._initial_dtype)
        self._initial_dtype = None
        return image

    def build_random_variables(self, **kwargs):
        # All this is done just once per batch (i.e. until `clear_random_variables` is called)
        np.random.seed()
        imshape = kwargs.get('imshape')
        # Build and scale random fields
        random_field_x = np.random.uniform(-1, 1, imshape) * self.alpha
        random_field_y = np.random.uniform(-1, 1, imshape) * self.alpha
        random_field_z = np.random.uniform(-1, 1, imshape) * self.alpha
        # Smooth random field (this has to be done just once per reset)
        sdx = gaussian_filter(random_field_x, self.sigma, mode='reflect')
        sdy = gaussian_filter(random_field_y, self.sigma, mode='reflect')
        sdz = gaussian_filter(random_field_z, self.sigma, mode='reflect')
        # Make meshgrid
        x, y, z = np.meshgrid(np.arange(imshape[2]), np.arange(imshape[1]),
                              np.arange(imshape[0]))
        # Make inversion coefficient
        _inverter = 1. if not self.invert else -1.
        # Distort meshgrid indices (invert if required)
        flow_z, flow_y, flow_x = (z + _inverter * sdz).reshape(-1, 1), \
                                 (y + _inverter * sdy).reshape(-1, 1), \
                                 (x + _inverter * sdx).reshape(-1, 1)
        # Set random states
        self.set_random_variable('flow_x', flow_x)
        self.set_random_variable('flow_y', flow_y)
        self.set_random_variable('flow_z', flow_z)

    def volume_function(self, volume):
        # Cast image to one of the native dtypes (one which that is supported by scipy)
        volume = self.cast(volume)
        # Take measurements
        imshape = volume.shape
        # print(imshape)

        # Obtain flows
        flow_z, flow_y, flow_x = self.get_random_variable('flow_z',
                                                          imshape=imshape), \
                                 self.get_random_variable('flow_y',
                                                          imshape=imshape), \
                                 self.get_random_variable('flow_x',
                                                          imshape=imshape)
        # To preserve orientation
        flows = flow_y, flow_z, flow_x

        # Map cooordinates from image to distorted index set
        transformed_image = map_coordinates(volume, flows, mode='reflect',
                                            order=0).reshape(imshape)
        # Uncast image to the original dtype
        transformed_image = self.uncast(transformed_image)
        return transformed_image


class SinusoidalElasticTransform3D(Transform):
    """Random Elastic Transformation 3D. Modified by Irene"""
    NATIVE_DTYPES = {'float32', 'float64'}
    PREFERRED_DTYPE = 'float32'

    def __init__(self, alpha, interp_step, order=3, **super_kwargs):
        self._initial_dtype = None
        super(SinusoidalElasticTransform3D, self).__init__(**super_kwargs)
        self.alpha = alpha
        self.order = order
        self.interp_factor = interp_step

    def cast(self, image):
        if image.dtype not in self.NATIVE_DTYPES:
            self._initial_dtype = image.dtype
            image = image.astype(self.PREFERRED_DTYPE)
        return image

    def uncast(self, image):
        if self._initial_dtype is not None:
            image = image.astype(self._initial_dtype)
        self._initial_dtype = None
        return image

    def build_random_variables(self, **kwargs):
        np.random.seed()
        imshape = kwargs.get('imshape')
        nz, ny, nx = [sh // self.interp_factor for sh in imshape]
        coarse_imshape = nz, ny, nx

        coarse_random_field_x = np.random.uniform(0, 2 * np.pi, coarse_imshape)
        coarse_random_field_y = np.random.uniform(0, 2 * np.pi, coarse_imshape)
        coarse_random_field_z = np.random.uniform(0, 2 * np.pi, coarse_imshape)

        coarse_random_field_x = np.sin(coarse_random_field_x) * self.alpha
        coarse_random_field_y = np.sin(coarse_random_field_y) * self.alpha
        coarse_random_field_z = np.sin(coarse_random_field_z) * self.alpha

        new_indices = np.mgrid[
                      0:nz - 1:self.interp_factor * nz * 1j,
                      0:ny - 1:self.interp_factor * ny * 1j,
                      0:nx - 1:self.interp_factor * nx * 1j
                      ]

        # interpolate:
        sdx = map_coordinates(coarse_random_field_x, new_indices,
                              order=self.order, output=coarse_random_field_x.dtype)
        sdy = map_coordinates(coarse_random_field_y, new_indices,
                              order=self.order, output=coarse_random_field_y.dtype)
        sdz = map_coordinates(coarse_random_field_z, new_indices,
                              order=self.order, output=coarse_random_field_z.dtype)
        # reshape
        sdx = sdx.reshape((self.interp_factor * nz, self.interp_factor * ny,
                           self.interp_factor * nx))
        sdy = sdy.reshape((self.interp_factor * nz, self.interp_factor * ny,
                           self.interp_factor * nx))
        sdz = sdz.reshape((self.interp_factor * nz, self.interp_factor * ny,
                           self.interp_factor * nx))
        # Make meshgrid
        x, y, z = np.meshgrid(np.arange(imshape[2]), np.arange(imshape[1]),
                              np.arange(imshape[0]))
        # print(x.shape)
        # Distort meshgrid indices (invert if required)
        flow_z, flow_y, flow_x = (z + sdz), (y + sdy), (x + sdx)
        # Set random states
        self.set_random_variable('flow_x', flow_x)
        self.set_random_variable('flow_y', flow_y)
        self.set_random_variable('flow_z', flow_z)

    def volume_function(self, volume):
        # Cast image to one of the native dtypes (one which that is supported by scipy)
        volume = self.cast(volume)
        # Take measurements
        imshape = volume.shape

        # Obtain flows
        flow_z = self.get_random_variable('flow_z', imshape=imshape)
        flow_y = self.get_random_variable('flow_y', imshape=imshape)
        flow_x = self.get_random_variable('flow_x', imshape=imshape)

        # To preserve orientation
        flows = flow_y, flow_z, flow_x
        # Map cooordinates from image to distorted index set
        transformed_image = map_coordinates(volume, flows, mode='reflect',
                                            order=0).reshape(imshape)
        # Uncast image to the original dtype
        transformed_image = self.uncast(transformed_image)
        return transformed_image


class AdditiveGaussianNoise(Transform):
    """Add gaussian noise to the input."""

    def __init__(self, sigma, epsilon=0.1, **super_kwargs):
        super(AdditiveGaussianNoise, self).__init__(**super_kwargs)
        self.sigma = sigma
        self.epsilon = epsilon

    def build_random_variables(self, **kwargs):
        np.random.seed()
        noise_radius = np.random.uniform(low=0, high=self.sigma, size=1)[0]
        gaussian_noise = np.random.normal(loc=0, scale=noise_radius,
                                          size=kwargs.get('imshape'))
        self.set_random_variable('noise', gaussian_noise)

        noise_amplitude = np.random.uniform(low=0, high=self.epsilon, size=1)[0]
        self.set_random_variable("noise_amplitude", noise_amplitude)

    def image_function(self, image):
        image = image + self.get_random_variable('noise', imshape=image.shape)
        return image

    def volume_function(self, volume):
        noise_level = self.get_random_variable('noise_amplitude')
        noise = noise_level * \
                self.get_random_variable('noise', imshape=volume.shape)
        volume = volume + noise
        return volume


class AdditiveSaltAndPepperNoise(Transform):
    """Add salt-and-pepper noise to the input."""

    def __init__(self, p=0.04, amplitude=1, **super_kwargs):
        super(AdditiveSaltAndPepperNoise, self).__init__(**super_kwargs)
        self.p = p
        self.amplitude = amplitude

    def build_random_variables(self, **kwargs):
        np.random.seed()
        noise_p = np.random.uniform(low=0, high=self.p, size=1)[0]
        noise_ampl = np.random.uniform(low=0, high=self.amplitude, size=1)[0]
        salt = np.random.binomial(n=1, p=noise_p, size=kwargs.get('imshape'))
        pepper = np.random.binomial(n=1, p=noise_p, size=kwargs.get('imshape'))
        salt_pepper = noise_ampl * (salt - pepper)
        self.set_random_variable('noise', salt_pepper)

    def image_function(self, image):
        image = image + self.get_random_variable('noise', imshape=image.shape)
        return image

    def volume_function(self, volume):
        volume = volume + self.get_random_variable('noise',
                                                   imshape=volume.shape)
        return volume


def transform_data_from_h5(training_data_path: str, label_name: str,
                           number_iter: int, output_data_path: str,
                           split=-1, transform_type='Gaussian',
                           sigma_gauss=1,
                           alpha_elastic=5,
                           interp_step=8,
                           p_rotation=0.8,
                           max_angle_rotation=90,
                           only_rotate_xy=False):
    raw_data, labeled_data = h5.read_training_data(training_data_path,
                                                   label_name=label_name,
                                                   split=split)
    numb_train = raw_data.shape[0]
    print(numb_train)
    raw_data = raw_data[None, :]
    labeled_data = labeled_data[None, :]
    for iteration in range(number_iter):
        if iteration == 0:
            # transform = False
            transformed_raw, transformed_labeled = raw_data, labeled_data
        else:
            transform = True
            transformed_raw, transformed_labeled = \
                transform_data(raw_data=raw_data,
                               labeled_data=labeled_data,
                               transform=transform,
                               transform_type=transform_type,
                               sigma_gauss=sigma_gauss,
                               alpha_elastic=alpha_elastic,
                               interp_step=interp_step,
                               p_rotation=p_rotation,
                               max_angle_rotation=max_angle_rotation,
                               only_rotate_xy=only_rotate_xy)

        with h5py.File(output_data_path, 'a') as f:
            for img_index in range(numb_train):
                subtomo_name = str(iteration) + "_" + str(img_index)
                subtomo_raw_h5_path = h5_internal_paths.RAW_SUBTOMOGRAMS
                subtomo_raw_h5_path = join(subtomo_raw_h5_path, subtomo_name)

                subtomo_label_h5_path = h5_internal_paths.LABELED_SUBTOMOGRAMS
                subtomo_label_h5_path = join(subtomo_label_h5_path,
                                             label_name)
                subtomo_label_h5_path = join(subtomo_label_h5_path,
                                             subtomo_name)

                f[subtomo_raw_h5_path] = transformed_raw[0, img_index, :, :, :]
                f[subtomo_label_h5_path] = transformed_labeled[0, img_index, :,
                                           :, :]
    return


def transform_data_from_h5_dice_multi_class(training_data_path: str,
                                            segmentation_names: list,
                                            number_iter: int,
                                            output_data_path: str):
    raw_data, labeled_data = h5.read_training_data_dice_multi_class(
        training_data_path,
        segmentation_names=segmentation_names,
        split=-1)
    numb_train = raw_data.shape[0]
    print(numb_train)
    raw_data = raw_data[None, :]
    # labeled_data = labeled_data[None, :]
    for iteration in range(number_iter):
        if iteration == 0:
            transform = False
        else:
            transform = True

        transformed_raw, transformed_labeled = transform_data_dice_multi_class(
            raw_data,
            labeled_data,
            transform)

        print("transformed_labeled.shape = ", transformed_labeled.shape)
        with h5py.File(output_data_path, 'a') as f:
            for img_index in range(numb_train):
                subtomo_name = str(iteration) + "_" + str(img_index)
                subtomo_raw_h5_path = h5_internal_paths.RAW_SUBTOMOGRAMS
                subtomo_raw_h5_path = join(subtomo_raw_h5_path, subtomo_name)
                f[subtomo_raw_h5_path] = transformed_raw[0, img_index, :, :, :]
                subtomo_label_h5_path = h5_internal_paths.LABELED_SUBTOMOGRAMS
                for channel, label_name in enumerate(segmentation_names):
                    subtomo_label_h5_name = join(subtomo_label_h5_path,
                                                 label_name)
                    subtomo_label_h5_name = join(subtomo_label_h5_name,
                                                 subtomo_name)
                    f[subtomo_label_h5_name] = transformed_labeled[img_index,
                                               channel, :, :, :]
    return


def transform_data(raw_data: np.array,
                   labeled_data: np.array,
                   transform=True,
                   transform_type='Gaussian',
                   sigma_gauss=1,
                   alpha_elastic=0.5,
                   interp_step=5,
                   p_rotation=0.8,
                   max_angle_rotation=90,
                   only_rotate_xy=False) -> tuple:
    if transform:
        if transform_type == 'Gaussian':
            sigma_gauss *= np.random.random()
            transform = AdditiveGaussianNoise(sigma=sigma_gauss)
            transformed_raw = transform(raw_data)
            return transformed_raw, labeled_data
        elif transform_type == 'SinElastic':
            interp_step = np.random.randint(1, interp_step)
            interp_step = 2 ** interp_step
            alpha = alpha_elastic * np.random.random()
            transform = SinusoidalElasticTransform3D(alpha=alpha,
                                                     interp_step=interp_step)
            transformed_raw = transform(raw_data)
            transformed_labeled = transform(labeled_data)
            return transformed_raw, transformed_labeled
        elif transform_type == "Rotation":
            # erases channel dim:
            transform = RandomRot3D(rot_range=max_angle_rotation, p=p_rotation,
                                    only_xy=only_rotate_xy)
            transformed_raw = transform(raw_data)
            transformed_labeled = transform(labeled_data)
            transformed_raw = transformed_raw[None, :]
            transformed_labeled = transformed_labeled[None, :]
            return transformed_raw, transformed_labeled
        elif transform_type == 'All':
            interp_step = np.random.randint(1, interp_step)
            interp_step = 2 ** interp_step
            alpha = alpha_elastic * np.random.random()
            transform = SinusoidalElasticTransform3D(alpha=alpha,
                                                     interp_step=interp_step)
            transformed_raw = transform(raw_data)
            transformed_labeled = transform(labeled_data)

            print("Rotations are blocked for the moment")
            transform = RandomRot3D(rot_range=max_angle_rotation,
                                    p=p_rotation,
                                    only_xy=only_rotate_xy)
            transformed_raw = np.array(transform(transformed_raw))
            transformed_labeled = np.array(transform(transformed_labeled))
            print("transformed_raw.shape = ", transformed_raw)
            print("transformed_labeled.shape = ", transformed_labeled.shape)
            transformed_raw = transformed_raw[None, :]
            transformed_labeled = transformed_labeled[None, :]

            sigma_gauss *= np.random.random()
            transform = AdditiveGaussianNoise(sigma=sigma_gauss)
            transformed_raw = transform(transformed_raw)
            return transformed_raw, transformed_labeled
        else:
            print("The requested transformation name is not valid.")
    else:
        print("The data in the first iteration is intact.")
        return raw_data, labeled_data


def transform_data_dice_multi_class(raw_data: np.array, labeled_data: np.array,
                                    transform=True) -> tuple:
    if transform:
        sigma = np.random.random()
        transform = AdditiveGaussianNoise(sigma=sigma)
        transformed_raw = transform(raw_data)
        transformed_labeled = labeled_data
        return transformed_raw, transformed_labeled
    else:
        print("The data in the first iteration is intact.")
        return raw_data, labeled_data
