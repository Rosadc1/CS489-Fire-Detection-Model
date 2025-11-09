The following notes will be useful to guide you on specific issues that might come up.

1. Run the code on the cpu using `python main.py` on the terminal to download the ResNet50 model. 
   Once the download completed, you can halt the execution using `Ctrl+C` and run your code on CUDE.
2. Follow the comments on each and each function to guide what's required.
3. Easiest way to print the architecture of a model is to simply print it, or use torchsummary.
4. Freeze all layers of a model as below (Can also be used to print all parameters of a model)
    for param in model.parameters():
        param.requires_grad = False
5. To Unfreeze a specific part of a model
    for param in model.layer[n].parameters():
        param.requires_grad = True
6. Feel Free to implement the metrics using PyTorch (with GPU) or Numpy(with CPU).
7. Loss functions must be implemented using PyTorch.
8. tensor.cpu().numpy() can be used to convert PyTorch tensors into numpy arrays.