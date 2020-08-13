# Dogs VS Cats Classifier 🐶 🆚 🐱

## built using *Pytorch* as a framework and *CNN model* with the following architecture:

1. **Conv2d**(1, 32, kernel_size=(3, 3))    - **ReLU**() - **MaxPool2d**(kernel_size=2, stride=2)
1. **Conv2d**(32, 64, kernel_size=(3, 3))   - **ReLU**() - **MaxPool2d**(kernel_size=2, stride=2)
1. **Conv2d**(64, 128, kernel_size=(3, 3))  - **ReLU**() - **MaxPool2d**(kernel_size=2, stride=2)
1. **Conv2d**(128, 256, kernel_size=(3, 3)) - **ReLU**() - **MaxPool2d**(kernel_size=2, stride=2)
1. **Flatten**()
1. **Linear**(in_features=9216, out_features=1000) - **ReLU**()
1. **Linear**(in_features=1000, out_features=200) - **ReLU**()
1. **Linear**(in_features=200, out_features=1) - **Sigmoid**()

--- 

- [Download data from here](https://www.kaggle.com/c/dogs-vs-cats/data)

|**feature**|**value**|
|:---:|:---:|
|Image size|128 * 128|
|Coloring| gray_scale|
|Data splitting| 0.8 : 0.2 |
|Epochs|10|
|Accuracy|0.82|
