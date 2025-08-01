# Leap Labs
Adversarial Noise for Leap Labs

This repository is a python tool to generate targeted adversarial examples on images using Projected Gradient Descent with pretrained ConvNeXt model.

## Setup
1. Clone this repository 
2. Create a virtual environment with dependancies listed on requirements.txt

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

- Quality of the adversarial sample may be poor due to resizing
- There is an option to use gpu when running on terminal with `--device`, gradio app will select gpu if available.

### Updates
- [01/08] Added ImageNet normalisation and tested on non-ImageNet samples