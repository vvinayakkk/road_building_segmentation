# **Building and Road Segmentation from Aerial Images using EffUNet**

Urban infrastructure data—covering roads, buildings, water supply, power lines, and more—is critical for effective city planning. Accurate segmentation of these elements from aerial imagery helps decision-makers assess their spread, location, and capacity. This project focuses on segmenting buildings and roads from satellite and UAV images, leveraging modern deep learning techniques for precise urban mapping.

In this project, we introduce a novel segmentation architecture using EfficientNetV2 as the encoder and UNet as the decoder, which together form the EffUNet model. This hybrid architecture efficiently captures high-level features from aerial images and reconstructs precise segmentation maps. Our approach sets a new benchmark on the Massachusetts Building and Road Segmentation dataset, achieving a mean Intersection over Union (mIOU) score of **0.8365 for buildings** and **0.9153 for roads**.

---

## **Project Overview**

This repository contains the implementation of a deep learning-based approach for segmenting roads and buildings from aerial images. The backbone of our architecture is based on the EfficientNetV2 encoder, which significantly improves performance and computational efficiency. The UNet decoder reconstructs the segmented maps with high precision, making it suitable for large-scale urban mapping tasks.

The project has been tested on the Massachusetts Building and Road Segmentation dataset, where the model performed exceptionally well across various metrics.

---

## **Key Features**

- **EfficientNetV2 Encoder**: Captures high-level, rich spatial features from aerial images with superior efficiency and accuracy.
- **UNet Decoder**: Reconstructs the segmentation maps with high detail, producing sharp boundaries and accurate object localization.
- **Custom Loss Function**: Combines Dice Loss and Cross-Entropy Loss to handle class imbalance and improve segmentation quality.
- **Metrics for Evaluation**: We use multiple metrics to evaluate model performance, including mIOU, Dice Loss, Precision, Recall, F1 Score, and Accuracy.

---

## **Dataset**

We use the **Massachusetts Building and Road Segmentation dataset**, a publicly available dataset containing aerial images with labeled buildings and roads. The dataset provides a robust challenge for semantic segmentation tasks due to the varying sizes and shapes of urban objects.

---

## **Installation**

To replicate the project, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Building-and-Road-Segmentation-from-Aerial-Images-using-EffUNet.git
   cd Building-and-Road-Segmentation-from-Aerial-Images-using-EffUNet

## **Model Architecture**

The architecture used in this project is EffUNet, which combines the power of Google's EfficientNetV2 for feature extraction with UNet's decoding capabilities to create high-quality segmentation masks. Here's a breakdown of the architecture:

1. **EfficientNetV2 Encoder**: EfficientNetV2 is a state-of-the-art CNN designed for both high performance and computational efficiency. It extracts high-level features from the aerial images while using fewer parameters compared to traditional architectures.
   
2. **UNet Decoder**: The UNet architecture is well-known for its performance in image segmentation tasks. It consists of upsampling layers that reconstruct the final segmentation map by combining the feature maps from the encoder using skip connections. This ensures that spatial information is preserved throughout the process.

3. **Skip Connections**: Skip connections are used to transfer spatial information directly from the encoder to the corresponding decoder layers. This helps in retaining the fine details, such as the boundaries of roads and buildings, which are crucial in segmentation tasks.

4. **Output Layer**: A final convolutional layer is applied to produce the segmentation map, followed by a softmax activation function to generate the probability distribution for each class (building, road, background, etc.).

---

## **Results**

### **Building Segmentation Performance**

The following table summarizes the performance of different model variants on the building segmentation task:

| Model     | mIOU  | Dice Loss | Precision | Recall | F1 Score | Accuracy |
|-----------|-------|-----------|-----------|--------|----------|----------|
| V2S+UNet  | 0.8159 | 0.1054    | 0.8746    | 0.9220 | 0.8977   | 0.8997   |
| V2M+UNet  | 0.8293 | 0.0977    | 0.8821    | 0.9316 | 0.9062   | 0.9080   |
| B7+UNet   | 0.8359 | 0.0934    | 0.8863    | 0.9352 | 0.9101   | 0.9119   |
| V2L+UNet  | **0.8365** | **0.0925** | **0.8865** | **0.9356** | **0.9104** | **0.9122** |

### **Road Segmentation Performance**

The following table summarizes the performance of different model variants on the road segmentation task:

| Model     | mIOU  | Dice Loss | Precision | Recall | F1 Score | Accuracy |
|-----------|-------|-----------|-----------|--------|----------|----------|
| V2S+UNet  | 0.9139 | 0.0453    | 0.9321    | 0.9786 | 0.9548   | 0.9558   |
| V2M+UNet  | 0.9140 | 0.0475    | 0.9323    | 0.9786 | 0.9549   | 0.9559   |
| V2L+UNet  | 0.9147 | 0.0468    | 0.9328    | 0.9790 | 0.9553   | 0.9563   |
| B7+UNet   | **0.9153** | **0.0461** | **0.9332** | **0.9792** | **0.9556** | **0.9566** |

---

## **Visual Results**

### **Building Segmentation**

Here is an example of a segmented building image using our model:

![Building Segmentation Result](https://github.com/lostmartian/Building-and-Road-Segmentation-from-Aerial-Images-using-EffUNet/blob/main/images/bout.png)

### **Road Segmentation**

Here is an example of a segmented road image using our model:

![Road Segmentation Result](https://github.com/lostmartian/Building-and-Road-Segmentation-from-Aerial-Images-using-EffUNet/blob/main/images/rout.png)

---


## **Tech Stack**

The project is developed using the following tools and libraries:

- **Language**: Python
- **Deep Learning Framework**: PyTorch
- **Visualization**: Matplotlib
- **Numerical Computing**: Numpy
- **Segmentation Library**: Segmentation Models PyTorch (SMP)

---

## **Conclusion**

EffUNet demonstrates excellent performance on the task of segmenting buildings and roads from aerial imagery, surpassing existing benchmarks on the Massachusetts Building and Road Segmentation dataset. This architecture balances the accuracy and efficiency required for large-scale applications, making it a strong solution for urban planning, disaster management, and infrastructure monitoring.



