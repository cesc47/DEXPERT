"""
This script generates the object detection results for the Date Estimation in the Wild dataset. The model used is the
DETR model from Facebook AI. The results are saved in the results_object_detection_detr_Balanced folder.
"""

import os
import json
import torch

from PIL import Image
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection


def print_results(results, model):
    """
    Print the results of the object detection model.
    :param results: results of the object detection model
    :param model: model used for object detection
    """
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )


def initialize_folders(name):
    """
    Generate folders for the results: in the same structure as the images.
    :param name: name of the folder to generate
    """
    os.makedirs(name, exist_ok=True)
    for k in range(1930, 2000):
        os.makedirs(f'{name}/{k}', exist_ok=True)
        for i in range(1, 10):
            os.makedirs(f'{name}/{k}/{i}', exist_ok=True)
            for j in range(100):
                os.makedirs(f'{name}/{k}/{i}/{j:02d}',
                            exist_ok=True)


def generate_detections(extractor, model, name, path_db):
    """
    Generate the detections for the given dataset. The results are saved in folder called name. You can do it for the
    DEW or DEW-B datasets.
    :param name: name of the folder to save the results
    :param path_db: path to the database where the images are stored
    :param extractor: feature extractor
    :param model: object detection model
    """
    # iterate through folders
    for i in range(1930, 2000):
        print(f'Generating detections for year {i}')
        for j in tqdm(range(1, 10)):
            for k in range(100):
                try:
                    for filename in os.listdir(f'{path_db}/{name}/{i}/{j}/{k:02d}'):
                        if not os.path.exists(f'{name}/{i}/{j}/{k:02d}/{filename}.json'):
                            image = Image.open(
                                f'{path_db}/{name}/{i}/{j}/{k:02d}/{filename}')
                            image = image.convert('RGB')
                            inputs = extractor(images=image, return_tensors="pt")
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                            outputs = model(**inputs)
                            target_sizes = torch.tensor([image.size[::-1]])
                            target_sizes = target_sizes.to("cuda")
                            results = \
                                extractor.post_process_object_detection(outputs, target_sizes=target_sizes,
                                                                        threshold=0.9)[0]
                            # save the results dict in a json file named as dataset.filenames[i]
                            results = {k: v.cpu().detach().numpy().tolist() for k, v in results.items()}
                            # split filename and extension using split
                            filename = filename.split('.')[0]
                            with open(f'{name}/{i}/{j}/{k:02d}/{filename}.json', 'w') as f:
                                json.dump(results, f)

                except FileNotFoundError:
                    print(f'Folder {i}/{j}/{k:02d} does not exist')
                    continue


def main(args):
    # initialize folders
    initialize_folders(args.folder_name)

    # initialize the object detection model
    extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModelForObjectDetection.from_pretrained(args.model_name)
    if args.cuda:
        model.to("cuda")

    # generate the detections
    generate_detections(extractor, model, args.folder_name, args.path_to_db)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Generate detections from DETR object detector", add_help=add_help)
    parser.add_argument('--path_to_db', default='/datatmp/datasets/Date_Estimation_in_the_Wild_Balanced',
                        help='Path to the database DEW-B where the images are stored')
    parser.add_argument('--folder_name', default='results_object_detection',
                        help='Name of the folder where the results will be stored (can be in the same DEW-B)')
    parser.add_argument('--model_name', default='facebook/detr-resnet-50', help='Name of the model to use')
    parser.add_argument('--cuda', action='store_true', help='Use cuda')

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)