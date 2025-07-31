# Leap Labs
Adversarial Noise for Leap Labs

This repository is a python tool to generate targeted adversarial examples on images using Projected Gradient Descent with pretrained ConvNeXt model.

## Setup
Create a virtual environment with dependancies listed on requirements.txt

### Run on terminal
```
python main.py --image_path <path to image> --target_class <number>
```

For testing some examples image files from ImageNet validation set are included in `./test_samples`. File name format is `<orginal_class_no>.JPEG` 


### Run Gradio App

```
python app.py
```
Then open URL on browser, usually in the format of `http://127.0.0.1:7860`. 

### Notes

- Only tested on ImageNet sample images. 
- ImageNet normalisation is not applied. 
- Quality of the adversarial sample may be poor due to resizing 