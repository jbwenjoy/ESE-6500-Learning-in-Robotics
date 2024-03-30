from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import imageio
import cv2


def load_colmap_data():
    """
    After using colmap2nerf.py to convert the colmap intrinsics and extrinsics,
    read in the transform_colmap.json file

    Expected Returns:
      An array of resized imgs, normalized to [0, 1]
      An array of poses, essentially the transform matrix
      Camera parameters: H, W, focal length

    NOTES:
      We recommend you resize the original images from 800x800 to lower resolution,
      i.e. 200x200, so it's easier for training. Change camera parameters accordingly
    """
    ## Implemented

    with open(os.path.join('transforms_train.json')) as file:
        json_data = json.load(file)

    # rescaling camera parameters
    original_h, original_w = 800, 800
    new_h, new_w = 200, 200

    camera_angle_x = json_data['camera_angle_x']
    focal_length = 0.5 * original_w / np.tan(0.5 * camera_angle_x)
    focal_resized = focal_length * (new_w / original_w)

    img_file_paths = []
    rotations = []
    poses = []
    imgs = []
    for frame in json_data['frames']:
        img_file_paths.append(frame['file_path'] + ".png")
        rotations.append(frame['rotation'])
        poses.append(frame['transform_matrix'])

        img = cv2.imread(frame['file_path'] + ".png")
        if img is None:
            print(f"Image {frame['file_path'] + '.png'} not found")
            continue
        img = cv2.resize(img, (new_w, new_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        imgs.append(img)

    imgs = np.array(imgs)
    poses = np.array(poses)
    H, W = new_h, new_w
    focal_length = focal_resized

    return imgs, poses, [H, W, focal_length]


def get_rays(H, W, f, T_wc):
    """
    Compute rays passing through each pixel.

    Parameters:
        H: height of the image
        W: width of the image
        f: focal length of the camera
        T_wc: camera pose of 1 image, shape (4, 4)

    Expected Returns:
        ray_origins: A tensor of shape (H, W, 3) denoting the centers of each ray.
        ray_directions: A tensor of shape (H, W, 3) denoting the direction of each
            ray. ray_directions[i][j] denotes the direction (x, y, z) of the ray
            passing through the pixel at row index `i` and column index `j`.
    """
    ## Implemented

    # Compute the intrinsic matrix K
    K = np.array([[f, 0, W / 2],
                  [0, f, H / 2],
                  [0, 0, 1]])

    # Create a meshgrid for pixel coordinates
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    # Convert 2D pixel coordinates to 3D homogeneous pixel coordinates
    pixel_coordinates = np.stack([i, j, np.ones_like(i)], axis=-1)

    # Reshape pixel coordinates to tensor (H*W, 3)
    pixel_coordinates = np.reshape(pixel_coordinates, (-1, 3))

    # Get camera coordinates
    K_inv = np.linalg.inv(K)
    camera_coordinates = (K_inv @ pixel_coordinates.T).T

    # Convert camera coordinates to homogeneous coordinates
    camera_coordinates = np.concatenate([camera_coordinates, np.ones((H * W, 1))], axis=1)

    # Get world coordinates
    world_coordinates = (T_wc @ camera_coordinates.T).T

    # Calculate ray origins (camera position in the world frame)
    ray_origins = T_wc[:3, -1].reshape(1, 3)

    # Calculate ray directions (world_coordinates - camera position)
    ray_directions = world_coordinates[:, :3] - ray_origins

    # Normalize ray directions
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)

    # Reshape rays to match the (H, W, 3) shape
    ray_origins = np.tile(ray_origins, (H * W, 1)).reshape(H, W, 3)
    ray_directions = ray_directions.reshape(H, W, 3)

    return ray_origins, ray_directions


def sample_points_from_rays():
    r"""Compute a set of 3D points given the bundle of rays

    Expected Returns:
      sampled_points: axis of the sampled points along each ray, shape (H, W, num_samples, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)
    """
    # TODO: sample_points_from_rays
    pass


def positional_encoding():
    r"""Apply positional encoding to the input. (Section 5.1 of original paper)
    We use positional encoding to map continuous input coordinates into a 
    higher dimensional space to enable our MLP to more easily approximate a 
    higher frequency function.

    Expected Returns:
      pos_out: positional encoding of the input tensor. 
               (H*W*num_samples, (include_input + 2*freq) * 3)
    """
    # TODO: positional_encoding
    pass


def volume_rendering(
        radiance_field: torch.Tensor,
        ray_origins: torch.Tensor,
        depth_values: torch.Tensor
) -> Tuple[torch.Tensor]:
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    bundle, and the sampled depth values along them.

    Args:
      radiance_field: at each query location (X, Y, Z), our model predict 
        RGB color and a volume density (sigma), shape (H, W, num_samples, 4)
      ray_origins: origin of each ray, shape (H, W, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)
    
    Expected Returns:
      rgb_map: rendered RGB image, shape (H, W, 3)
    """
    # TODO: volume_rendering
    pass


class TinyNeRF(torch.nn.Module):
    def __init__(self, pos_dim, fc_dim=128):
        r"""Initialize a tiny nerf network, which composed of linear layers and
        ReLU activation. More specifically: linear - relu - linear - relu - linear
        - relu -linear. The module is intentionally made small so that we could
        achieve reasonable training time

        Args:
          pos_dim: dimension of the positional encoding output
          fc_dim: dimension of the fully connected layer
        """
        super().__init__()

        self.nerf = nn.Sequential(
            nn.Linear(pos_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 4)
        )

    def forward(self, x):
        r"""Output volume density and RGB color (4 dimensions), given a set of
        positional encoded points sampled from the rays
        """
        x = self.nerf(x)
        return x


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def nerf_step_forward(height, width, focal_length, trans_matrix,
                      near_point, far_point, num_depth_samples_per_ray,
                      get_minibatches_function, model):
    r"""Perform one iteration of training, which take information of one of the
    training images, and try to predict its rgb values

    Args:
      height: height of the image
      width: width of the image
      focal_length: focal length of the camera
      trans_matrix: transformation matrix, which is also the camera pose
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      get_minibatches_function: function to cut the ray bundles into several chunks
        to avoid out-of-memory issue

    Expected Returns:
      rgb_predicted: predicted rgb values of the training image
    """
    ################### YOUR CODE START ###################
    # TODO: Get the "bundle" of rays through all image pixels

    # TODO: Sample points along each ray

    # TODO: positional encoding, shape of return [H*W*num_samples, (include_input + 2*freq) * 3]

    ################### YOUR CODE END ###################

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(positional_encoded_points, chunksize=16384)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)  # (H*W*num_samples, 4)

    # "Unflatten" the radiance field.
    unflattened_shape = [height, width, num_depth_samples_per_ray, 4]  # (H, W, num_samples, 4)
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)  # (H, W, num_samples, 4)

    ################### YOUR CODE START ###################
    # TODO: Perform differentiable volume rendering to re-synthesize the RGB image. # (H, W, 3)
    pass
    ################### YOUR CODE END ###################


def train(images, poses, hwf, near_point,
          far_point, num_depth_samples_per_ray,
          num_iters, model, DEVICE="cuda"):
    r"""Training a tiny nerf model

    Args:
      images: all the images extracted from dataset (including train, val, test)
      poses: poses of the camera, which are used as transformation matrix
      hwf: [height, width, focal_length]
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      num_iters: number of training iterations
      model: predefined tiny NeRF model
    """
    H, W, focal_length = hwf
    H = int(H)
    W = int(W)
    n_train = images.shape[0]

    # Optimizer parameters
    lr = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    for _ in tqdm(range(num_iters)):
        # Randomly pick a training image as the target, get rgb value and camera pose
        train_idx = np.random.randint(n_train)
        train_img_rgb = images[train_idx, ..., :3]
        train_pose = poses[train_idx]

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = nerf_step_forward(H, W, focal_length,
                                          train_pose, near_point,
                                          far_point, num_depth_samples_per_ray,
                                          get_minibatches, model)

        # Compute mean-squared error between the predicted and target images
        loss = torch.nn.functional.mse_loss(rgb_predicted, train_img_rgb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('Finish training')


if __name__ == "__main__":

    # Load data
    imgs, poses, camera_hwf = load_colmap_data()

    # Test if the images are loaded correctly
    # plt.imshow(imgs[0])
    # plt.show()

    # get rays
    ray_origins, ray_directions = get_rays(*camera_hwf, poses[0])

    # TODO
    pass
