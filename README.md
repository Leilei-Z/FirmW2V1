# FirmW2V: A Fine-Grained Firmware Identification Framework for Online IoT Devices

The proliferation of Internet of Things (IoT) devices necessitates continuous firmware updates to patch vulnerabilities, optimize performance, and incorporate features. However, iterative firmware development often results in substantial code and configuration similarities, leading to widespread propagation of vulnerabilities across different versions. 

In view of the above issue, we present FirmW2V, the first deep learning-based framework for fine-grained identification of firmware in online IoT devices. To this end, we employ large language models (LLMs) for automated feature extraction and propose a novel metric loss function—Triplet-Center Similarity Loss (TCSL)—to enhance intra-class compactness and inter-class separability in firmware version classification. To validate this framework, we gathered 4,442 firmware images and acquired 130,445 valid embedded web interface files for performance evaluation experiments. The results show that FirmW2V significantly outperforms the state-of-the-art methods in accuracy, achieving an AUC of 0.984. Compared to scanning all web files as in traditional approaches, FirmW2V requires scanning only 1.2% of the files for firmware recognition, greatly enhancing efficiency. Furthermore, the improvement in recognition accuracy and efficiency enables prompt and precise assessment of vulnerability correlations across firmware versions in real-world devices. Our observations in real Internet environments reveal that, on average, approximately 22.18% of the IoT devices exhibit similar vulnerabilities. Our findings underscore the critical issue of vulnerability transmission across firmware and highlighting the need for vigilant security practices in firmware management.

Due to the sensitivity of data sourced from firmware or online devices, we provide only simplified pretrained models and code in the repository to facilitate the reproduction of our work.

## Installation

1. Clone the repository:

    ```bash
    Download all files
    cd FirmW2V
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a version of PyTorch with CUDA support installed if using a GPU.

## Usage Instructions

### Data Preparation

Data preprocessing is located in the `utils` directory, with key files including:
1. The dataset is stored in `dataset_demo`, containing embedded web pages from online IoT devices, from seven firmware versions of four manufacturers: D-Link, TP-LINK, Linksys, and Netgear.
2. The `data_processing.py` file contains some utility functions for data preprocessing and loading. These functions are mainly used for processing and preparing text data.
3. The `feature_extraction.py` file contains functions for generating embeddings by calling the large language model on long texts.
4. `training.py` and `testing.py` contain functions for training the model and evaluating the model, respectively.

### Model Definition
Model definitions are located in the `models` directory, with key files including:
1. `autoencoder.py`: Defines the autoencoder model.
2. `classifier.py`: Fully connected layers + softmax for final classification.
3. `TCS_loss.py`: Custom Triplet-Center Similarity Loss for optimizing model training.

### Training the Model

Configure model parameters and training settings, such as learning rate, batch size, and the number of training epochs in `train.py`. Run the script to train the model.

```bash
python train.py
```

After training is complete, the model weights will be saved in the `trained_models` directory.

### Testing the Model

Use `test.py` to test the trained model. This module loads the saved model weights and evaluates model performance.

```bash
python test.py
```

The test results include the classification accuracy of the model.

## Notes

- Ensure consistent `device` settings across all scripts to avoid tensor mismatch issues between GPU/CPU.
- Data file paths should be correctly set in the code.
- When using a GPU for training and testing, ensure CUDA is available and properly configured.
