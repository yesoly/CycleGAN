import argparse
import os
import glob
import torch
from PIL import Image

import torchvision.transforms as transforms

from models import Generator


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="datasets/inside")
parser.add_argument("--checkpoint_path", type=str)
parser.add_argument("--size", type=int, default=416)

args = parser.parse_args()


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 1, Model
    num_blocks = 6 if args.size <= 256 else 8
    netG_A2B = Generator(num_blocks).to(device)
    netG_B2A = Generator(num_blocks).to(device)

    # 2
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    netG_A2B.load_state_dict(checkpoint["netG_A2B_state_dict"], strict=False)
    netG_B2A.load_state_dict(checkpoint["netG_B2A_state_dict"], strict=False)

    netG_A2B.eval()
    netG_B2A.eval()

    # 3, Dataset
    transform_to_tensor = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    # 4
    transform_to_image = transforms.Compose(
        [
            transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
            transforms.ToPILImage(),
        ]
    )

    # 5
    dataset_name = os.path.basename(args.dataset_path)

    os.makedirs(f"results/{dataset_name}/testA", exist_ok=True)
    os.makedirs(f"results/{dataset_name}/testB", exist_ok=True)

    test_list = [["testA", netG_A2B], ["testB", netG_B2A]]

    # 6
    for folder_name, model in test_list:
        print(folder_name)
        image_path_list = sorted(
            glob.glob(os.path.join(args.dataset_path, folder_name) + "/*")
        )
        for idx, image_path in enumerate(image_path_list):
            image_name = os.path.basename(image_path)
            print(f"{idx}/{len(image_path_list)} {image_name}")

            # 7
            image = Image.open(image_path)
            image = transform_to_tensor(image).unsqueeze(0).to(device)

            output = model(image)

            # 8
            output = transform_to_image(output.squeeze())
            output.save(os.path.join("results", dataset_name, folder_name, image_name))


if __name__ == "__main__":
    test()

# $ python test.py --dataset_path datasets/mini  --checkpoint_path checkpoint/mini/500.pth