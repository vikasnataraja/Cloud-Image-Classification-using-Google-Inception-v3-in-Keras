# Cloud-Image-Classification-using-Google-Inception-v3-in-Keras
This repo explores the understanding of cloud imagery using Google's Inception v3 model for segmentation and classification and written in Keras.

## Overview - What this project is about:

This project comes via [Kaggle through one of their competitions](https://www.kaggle.com/c/understanding_cloud_organization/overview). The dataset is of cloud imagery and each cloud image has a certain number of broadly defined shapes namely Fish, Gravel, Sugar, and Flower. Each image has at least one of these shapes and possibly more. The task is to train the model to predict these classes using those images.
![The 4 classes for the cloud images](https://github.com/vikasnataraja/Cloud-Image-Classification-using-Google-Inception-v3-in-Keras/blob/master/extras/data_classes.png)

## Requirements -  What you will need to run this on your system:

* Python 3
* Tensorflow >=1.14 (tested on 1.14 and 2.0)
* Keras >=2.2.4
* Other standard libraries - Pandas, Numpy, Scikit-Learn, albumentations
* Plenty of hard disk space ~60GB since the dataset itself is around 43GB. I'd recommend using a GPU or an online cloud service like AWS or Google Cloud.
* Dataset - Available on Kaggle, [click here to download the zip file](https://www.kaggle.com/c/13333/download-all). Alternatively, you can download the one on my Google Bucket Storage using this command: `gsutil cp gs://vikas-cloud/understanding_cloud_organization.zip <your_local_directory>`

## Running the model

1. Clone this repo to your local system, `cd` into the folder. 
2. Next, download the dataset zip file, extract its contents.
3. Create another folder to save your models. I called mine `save_model`.

Ultimately, your folder structure should look similar to this:

```
workspace
│   sample_submission.csv
│   train.csv    
│
└───train_images
│   │   0011165.jpg
│   │   002be4f.jpg
│   │   ...
└───test_images
│   │   002f507.jpg
│   │   0035ae9.jpg
│   │   ...
└───save_model
│   │   <empty directory>
```

From that directory, run `main.py` using Python 3 with an argument for model directory. There are various arguments you can give to customize, here are a few examples:
```
python3 main.py --model_dir './save_model

python3 main.py --model_dir './save_model --batch_size 64

python3 main.py --model_dir './save_model --batch_size 64 --model_name 'modelname.h5'
```
You can use `python3 main.py --help` to view more arguments.

## Additional Resources
(Link to my complete project report coming soon).






