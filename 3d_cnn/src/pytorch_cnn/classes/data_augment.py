import numpy as np
# import torchvision.transforms as transforms
from src.python.tomogram_utils.random_transformations import AdditiveGaussianNoise, \
    SinusoidalElasticTransform3D, RandomRot3D


def get_transforms_3d(sigma_gauss=1, alpha_elastic=0.5, interp_step=5,
                      p_rotation=0.8, max_angle_rotation=90,
                      only_rotate_xy=False) -> list:
    # FIXME all transformations erase channel dim:
    transforms_chain = list()

    sigma_gauss *= np.random.random()
    transforms_chain += [AdditiveGaussianNoise(sigma=sigma_gauss)]


    #
    # transforms_chain += [RandomRot3D(rot_range=max_angle_rotation, p=p_rotation,
    #                                  only_xy=only_rotate_xy)]

    interp_step = np.random.randint(1, interp_step)
    interp_step = 2 ** interp_step
    alpha = alpha_elastic * np.random.random()

    # transforms_chain += [
    #     SinusoidalElasticTransform3D(alpha=alpha, interp_step=interp_step)]
    #
    # return transforms.Compose(transforms_chain)
    return transforms_chain
