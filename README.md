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



## DEXPERT Model

The proposed DEXPERT model consists of an object detector, an ensemble of Convolutional Neural Network (CNNs) experts, and a Transformer encoder that aggregates the information from the different experts to generate a final prediction.

![](./dexpert.png)

### Object detection module (TBD)

These files are used to train the specialists models and then the DEXPERT. By specifying the path to the DEW / DEW-B dataset and the folder name, detections will be performed and saved in the specified folder:
```
python detr_inference.py --path_to_db <path_to_db> --folder_name <folder_name> 
```
Otherwise, **detections** can be also **downloaded** [here](https://todo).


### Date indicator experts & Global CNN expert (TBD)
In order to train the date indicator experts or the global CNN, the following command must be executed:
```
python train.py --data-path <data_path> --model convnext_base --batch-size <batch_size> --opt adamw --lr 1e-5 --lr-scheduler cosineannealinglr --auto-augment ta_wide --epochs 70 --weight-decay 0.05 --norm-weight-decay 0.0 --train-crop-size 176 --val-resize-size 232 --ra-sampler --weights ConvNeXt_Base_Weights.DEFAULT --specialist <specialist> --resume <resume> --balanced 
```
- `<specialist>`: represents which date indicator expert you are training `Person`, `Train`, `Car`, `Boat`, `Bus`, `Airplane` or `None` (global).
- `<data_path>`: where the dataset is located.
- `<balanced>`: if the dataset is DEW-B, this flag must be specified, else, it must be omitted.
- `<resume>`: if you want to resume the training from a checkpoint, you must specify the path to the checkpoint, else, it must be omitted. In this case, in the paper we have trained first the global CNN and then the date indicator experts.

The other parameters are optional and can be modified, but the ones specified in the command are the ones used in the paper.
If you want to **download** the **weights** of the experts, you can do it [here](https://todo).

### DEXPERT (TBD)
The DEXPERT module uses the pretrained Date Indicator Experts and Global CNN to train the Transformer encoder. In order to train the DEXPERT, the commands that were specified in the previous section must be executed first or the weights must be downloaded. Then, in order to train the DEXPERT module the following command must be executed:
```
python train_DEXPERT.py train_DEXPERT.py --data-path <data_path> --model convnext_base --batch-size <batch_size> --opt adamw --lr 1e-3 --lr-scheduler cosineannealinglr --auto-augment ta_wide --epochs 50 --weight-decay 0.05 --norm-weight-decay 0.0 --ra-sampler --balanced
```
- `<data_path>`: where the dataset is located.
- `<balanced>`: if the dataset is DEW-B, this flag must be specified, else, it must be omitted.

The other parameters are optional and can be modified, but the ones specified in the command are the ones used in the paper.
If you want to **download** the **weights** of the DEXPERT model, you can do it [here](https://todo).