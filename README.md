# Fruit Detection with a Custom CNN in Arduino using TFLite
The main goal of this project is to create a CNN Network and later deploy it on Arduino 33 BLE SENSE using TensorFlow Lite. The CNN can predict what type of fruit there is on the image(Multi-CLass problem). The code was created on Google Collab, thus making it easy to replicate. Essentially the only thing that needs to be changed is the folder path (but keeping the folder structure identical to the one on Github).

***IMPORTANT***
- A few words about the arduino storage space, which will be useful later. Arduino has 1MB of flash memory and 256KB of SRAM. Flash memory is where the arduino sketch is stored, the model and all the constant variables are stored. SRAM is where the variables created by the sketch will be stored, as well as any dynamic variables such as the ones of tensor arena.
- TFlite model when deployed to a microcontroler doesn't require a dynamic memoryt allocation. All the memory that will be needed by the input,output,intermediate activations etc. is predefined in the sketch. The size that they will "occupy" is defined by the variable tensor arena. This memory allocation takes "space" from SRAM. The model itself (weights,architecture,quantization information) are stored in Flash memory.

There are 4 sections:

1. General details about the experiments that lead to the final results
2. Observations
3. Problems regarding the TFLite and Arduino Deployment
4. Results

## 0 - Upload Models directly to Arduino
- Go to the folders fruit_detect and fruit_Detect_quant_io. Both models there have full integer quantization but only the second has input quantization, the first one doesn't.
- Models are already in their respective folder so you can just go to either fruit_detect(no input-output quantization) or tofruit_Detect_quant_io(with input quantization) and upload the models to the arduino right away.

## 1 - End to End walkthrough

### Model Training and Quantization Aware Finetuning

- Go to the custom_cnn3.ipynb notebook
- The notebook contains all the necessary code in order to train and convert to TFLite both a normally trained model as well the ability to finetune this model using quantization aware training. These models can be then quantized using Full Intger Quantization with either input quantization or without it, by running the respective cell commands.
- It should be noted that a quantization aware model isn't quantized by itself nor does it have reduced size. Quantization aware training only leads to weights that are more robust to quantization. This means that we still need to perform full interger quantization in order to see size reduction
- More informationr regarding the intermediate steps can be found inside the notebook
- 
### Weight clustering Finetuning

- Go to the weight_clustering.ipynb notebook
- The notebook contains all the necessary steps and information in order to apply to an ALREADY TRAINED model weight clustering optimization.
- Again in order to see improvements on size we need to quantize the final model.
- Clustering, or weight sharing, reduces the number of unique weight values in a model, leading to benefits for deployment. It first groups the weights of each layer into N clusters, then shares the cluster's centroid value for all the weights belonging to the cluster. .This technique brings improvements via model compression

### Weight pruning Finetuning

- Go to the pruning.ipynb notebook
- The notebook contains all the necessary steps and information in order to apply to an ALREADY TRAINED model weight pruning optimization.
- Again in order to see improvements on size we need to quantize the final model.
- Magnitude-based weight pruning gradually zeroes out model weights during the training process to achieve model sparsity. Sparse models are easier to compress, and we can skip the zeroes during inference for latency improvements. This technique brings improvements via model compression

#### Load Model to Arduino
- All 4 approaches generate a model.h file which is necessary in order to upload the model to arduino. In order to get the model you need to copy it from TFLite_Models folder
- On the next step you should put the model on the respective arduino folder. If you selected to quantize the input copy the model in the folder fruit_Detect_quant_io. If you selected to not quantize the input the copy the model in the folder fruit_detect.
- Then all you have to do is to open the .ino file using the arduino IDE and just upload the model to the arduino device :).

***IMPORTANT***
- If you wish to alter the code and create new models you will have to move them from the TFLITE_models folder to their respective arduino folder. Either fruit_detect(no input-output quantization) or to fruit_Detect_quant_io(with input quantization). From IDE all you have to do is to click upload(top left)


## 2 - Observations

- Even with small architectures, size restriction seems to set a great limitation for the performance
- All 4 approaches seem to result in great compression and accuracy preservesion. More experiments need to be performed in order to identify which is one performs best overall. The pruning and clustering methods can be further optimized by changing the hyperparameters
- Only the full integer quantization methods with and without input quantization works
- Our model had 74,088 parameters.
- Average inference time in both cases and for all 4 approaches for one observation on arduino was roughly 1.8 sec
- Significant model size reduction, from roughly 300KB model size, quantizationS led to models of size roughly 88KB.
- When uploaded, the models used roughly 50% of the FLASH memory and 90% of the SRAM memory. Should be noted that the size selected for tensor arena was 100*1024 Bytes/ 100KB, tensor arena is stored in SRAM. We can increase or decrease(x * 1024) the size of tensor arena to match the requirements of our model memory requirements as well as reduce the memory requirements.
- Accuracy wise the original model and the optimized had similar performance. More details can be seen at the bottom of the readme

***IMPORTANT***
- Both quantization aware, clustering and pruning need to be trained first normaly before applying those models. After training we finetune our model for 3-5 epochs using these methods. If we train our model from scratch using this methods accuracy will be significantly low.

## 3 - Problems (Personal opinions included, I could be wrong in some. Feel free to message me or send a pull request if you believe so)

- The main issue was the outdated documentation and the fact that many functions/approaches are no longer supported by either arduino or tensorflow.
- Many microcontrollers models seem to not work correctly and their software support for TF seems to be outdated, the safest options would be ARDUINO 33 BLE SENSE and Rasberry PICO.
- It seems that only full integer quantization (with and wihout input/output quantization) seem to work. Otherwise an error called HYBRID MODELS not supported occurs. This seems to be due to optimization leaving part of the operations as floating point (hence hybrid) and mixing isn't supported in micro.
(https://github.com/tensorflow/tensorflow/issues/43386)
- Input/Output uint8 quantization is no longer supported.
(https://github.com/tensorflow/tflite-micro/issues/280)
- Input/Output int8 quantization had a significant accuracy drop and weird behavior.More specifically the output when transformed back had different results than the ones we expected. Thus only input quantization was preferred.
(https://github.com/tensorflow/tflite-micro/issues/396)
- In order to truly leverage the optimization done from the pruned and clustered models we would need to zip them. But this would make the model unusable from arduino

## 4 -  Results

| Method  | Post Training Quantization | Test Accuracy  | Model Size in KB   | Fit Arduino Memory |
| --------------------------------------------------------------------------------- | ---------- | -------------------- | ------------------------ | -------------- |
| Quantization Aware   | Full Integer Without Input quantization        |    0.82      | 89.16                 | ✅             |
| Quantization Aware   | Full Integer With Input quantization        |     0.82     |    88.87              | ✅             |
| Quantization Aware   | None        |    0.84       | 603                 |              |
| None   | Full Integer Without Input quantization        |    0.83       | 88.32                | ✅             |
| None    | Full Integer With Input quantization        |     0.84     |         88.14         | ✅             |
| None    | None        |     0.83       |            298      |           | 
| Pruning   | Full Integer Without Input quantization        |  0.82        |       88.62           | ✅             |
| Pruning    | Full Integer With Input quantization        |     0.82     |           88.14       | ✅             |
| Pruning    | None       |   0.84      |       x           |             | 
| Clustering   | Full Integer Without Input quantization        |   0.86       |    88.21             | ✅             |
| Clustering    | Full Integer With Input quantization        |    0.84      |          88.05       | ✅             |
| Clustering    | None       | 0.88        |        x          |             |

## 5 - Requirements

tensorflow version == 2.8.2

tensorflow.keras version == 2.8.0
 
tensorflow_model_optimization version == 0.7.2

arduino_tensorflowlite == 2.4.0-ALPHA
