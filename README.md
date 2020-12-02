# Jsonify

Posnet based classification

## Getting Ready
1. Make sure you have the 64-bit python 3.8.x or greater
2. Create a new virtualenv
```bash
virtualenv cpu_only
cpu_only\Scripts\activate
```
3. Install Pytorch from the [official site](https://pytorch.org/get-started/locally/) // OR use the following command
```bash
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```


4. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install remaining packages.

```bash
pip install -r requirements.txt
```

## Running the code
On command line
```bash
python final.py
```


### Pytorch Posenet port was taken [from this repo](https://github.com/rwightman/posenet-pytorch)

### Optional
In the output folder, I have added a port of PyTorch model to ONNX for loading in Tensorflow.