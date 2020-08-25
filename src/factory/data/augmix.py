# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""
import numpy as np
from PIL import Image

from .augmentations import augmentations_all


def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image
  pil_img = op(pil_img, severity)
  x = np.asarray(pil_img)
  return x


def augment_and_mix(image, severity=-1, width=3, depth=-1, alpha=1.):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
      -1 enables stochastic severity from [1, 10]
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))
  mix = np.zeros_like(image).astype('float32')

  for i in range(width):
    image_aug = image.copy()
    depth = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(depth):
      op = np.random.choice(augmentations_all)
      severity = severity if severity > 0 else int(round(10*np.random.beta(5, 10)))
      image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * image_aug.astype('float32')

  mixed = (1 - m) * image.astype('float32') + m * mix
  return mixed


class AugMix:

  def __init__(self, severity=-1, width=3, depth=-1, alpha=1.):
    self.severity = severity
    self.width = width
    self.depth = depth
    self.alpha = alpha

  def __call__(self, image):
    return {'image': augment_and_mix(image, self.severity, self.width, self.depth, self.alpha)}
