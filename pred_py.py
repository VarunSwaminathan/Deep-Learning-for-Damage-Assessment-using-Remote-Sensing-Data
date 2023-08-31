import os.path
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import matplotlib as mlp
import kornia
import argparse

mlp.use('Qt5Agg')

from imageio import imsave
from torch.utils import model_zoo

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from building_footprint_segmentation.seg.binary.models import ReFineNet
from building_footprint_segmentation.helpers.normalizer import min_max_image_net
from building_footprint_segmentation.utils.py_network import (
    to_input_image_tensor,
    add_extra_dimension,
    convert_tensor_to_numpy,
    load_parallel_model,
    adjust_model,
)
from building_footprint_segmentation.utils.operations import handle_image_size

MAX_SIZE = 384
TRAINED_MODEL = ReFineNet()
MODEL_URL = "https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip"


def i_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def i_nd(image_tensor):
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).float()
    return np.array(image_tensor)


def clean_binary_mask(binary_mask, kernel_size=3):
    device = binary_mask.device
    kernel = torch.ones(kernel_size, kernel_size, device=device)
    eroded_mask = kornia.morphology.erosion(binary_mask, kernel)
    dilated_mask = kornia.morphology.dilation(eroded_mask, kernel)
    return dilated_mask


def set_model_weights():
    state_dict = model_zoo.load_url(MODEL_URL, progress=True, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    TRAINED_MODEL.load_state_dict(adjust_model(state_dict))


def extract(original_image):
    original_height, original_width = original_image.shape[:2]

    if (original_height, original_width) != (MAX_SIZE, MAX_SIZE):
        original_image = handle_image_size(original_image, (MAX_SIZE, MAX_SIZE))

    # Apply Normalization
    normalized_image = min_max_image_net(img=original_image)

    tensor_image = add_extra_dimension(to_input_image_tensor(normalized_image))

    with torch.no_grad():
        # Perform prediction
        prediction = TRAINED_MODEL(tensor_image)
        prediction = prediction.sigmoid()

    prediction_binary = convert_tensor_to_numpy(prediction[0]).reshape(
        (MAX_SIZE, MAX_SIZE)
    )

    prediction_3_channels = cv2.cvtColor(prediction_binary, cv2.COLOR_GRAY2RGB)

    dst = cv2.addWeighted(
        original_image,
        1,
        (prediction_3_channels * (0, 255, 0)).astype(np.uint8),
        0.4,
        0,
    )
    return prediction_binary, prediction_3_channels, dst


def run(image_path):
    original_image = Image.open(image_path).convert("RGB")
    original_image = np.asarray(torchvision.transforms.Resize(MAX_SIZE + 1)(original_image))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = original_image[:MAX_SIZE, :MAX_SIZE, :]
    set_model_weights()

    prediction_binary, prediction_3_channels, dst = extract(original_image)
    return prediction_binary, prediction_3_channels.astype(float), original_image.astype(float) / 255


def predict(before_PTH, after_PTH, result_PTH):
    _, b_mask, b_out = run(before_PTH)
    _, a_mask, a_out = run(after_PTH)

    b_mask = i_nd(kornia.filters.gaussian_blur2d(i_tensor(b_mask), (9, 9), (2, 2)))
    a_mask = i_nd(kornia.filters.gaussian_blur2d(i_tensor(a_mask), (9, 9), (2, 2))) * b_mask

    b_mask = i_nd(clean_binary_mask(i_tensor(b_mask), kernel_size=29))
    a_mask = i_nd(kornia.morphology.dilation(i_tensor(a_mask), torch.ones(9, 9)))

    red_mask = np.ones_like(b_mask) * (1, 0, 0)
    green_mask = np.ones_like(b_mask) * (0, 1, 0)
    mask = red_mask * b_mask + (green_mask - red_mask * 0.5) * a_mask

    r = np.sum(mask[0, :, :])
    g = np.sum(mask[1, :, :])
    print(f"Damage ratio: {r/(r+g)*100:0.2f} %")

    # %%
    fig, ax = plt.subplots(1, 4, figsize=(15, 15))
    ax = ax.ravel()

    ax[0].imshow(a_out)
    ax[0].set_title('a) After Disaster', y=-0.4)

    ax[1].imshow(b_out)
    ax[1].set_title('b) Before Disaster', y=-0.4)

    ax[2].imshow(mask)
    ax[2].set_title('c) Mask', y=-0.4)

    ax[3].imshow(np.clip(b_out * 0.7 + mask * 0.5, 0, 1))
    ax[3].set_title('d) Final Results', y=-0.4)
    plt.savefig(result_PTH)
    plt.show()




def parse_args():
    parser = argparse.ArgumentParser(description="Task 2")

    parser.add_argument("--before_PTH", type=str, default="ex/1_before.png", help="path to the image before disaster")
    parser.add_argument("--after_PTH", type=str, default="ex/2_after.png", help="path to the image after disaster")
    parser.add_argument("--result_PTH", type=str, default="results.png", help="path to results.png file")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    predict(args.before_PTH, args.after_PTH, args.result_PTH)
    print("Done !")


if __name__ == "__main__":
    main()