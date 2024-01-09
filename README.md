# DEXPERT

TBD

This repo provides the dataset, code, and models' weights for the paper "A Transformer-based object-centric approach for date estimation of historical photographs".

```
@inproceedings{net2017dexpert,
  title={A Transformer-based object-centric approach for date estimation of historical photographs},
  author={Net, Francesc and Hernández, Núria and Molina, Adrià and Gomez, Lluis},
  booktitle={46th European Conference on Information Retrieval, ECIR 2024},
  year={2024},
  organization={Springer}
}
```

## DEW-B Dataset

The DEW-B (Date Estimation in the Wild - Balanced) dataset, aims to provide more training images for the years that are less represented in the [DEW dataset](https://www.radar-service.eu/radar/en/dataset/tJzxrsYUkvPklBOw), from 1930 to 1969. The contributed training data has been harvested from three sources: the [LAION-5B dataset](https://laion.ai/blog/laion-5b/), and the [Flickr](https://www.flickr.com/) and [Europeana](https://www.europeana.eu/es) web portals.


### Images URLs and annotation files (TBD)

### Object detection files

These files are used to train the specialists models and then the DEXPERT. By specifying the path to the DEW/DEW-B dataset and the folder name, detections will be performed and saved in the specified folder:
```
python detr_inference.py --path_to_db <path_to_db> --folder_name <folder_name> 
```
Otherwise, detections can be also downloaded [here.](https://todo)

## DEXPERT Model

The proposed DEXPERT model consists of an object detector, an ensemble of Convolutional Neural Network (CNNs) experts, and a Transformer encoder that aggregates the information from the different experts to generate a final prediction.

![](./dexpert.png)

### Code and pre-trained weights (TDB)
