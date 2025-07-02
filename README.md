# Jute Pest Classification with PyTorch

## Overview
This project develops a deep learning system to classify 17 types of jute pests (e.g., Jute Stem Weevil, Beet Armyworm, Yellow Mite) using the "Jute Pest Dataset" with PyTorch. Achieving a **98% test accuracy** through a super ensemble model, the project combines custom-built convolutional neural networks (CNNs), fine-tuned pre-trained models, and a Flask-based API deployed on a responsive webpage. Despite starting as a beginner and pausing work due to other commitments, I resumed and completed the project 2-3 months ago, overcoming challenges like dataset imbalance and deployment constraints.

## Dataset
- **Source**: Jute Pest Dataset (7,236 RGB images across train, validation, and test sets).
- **Classes**: 17 pest types (e.g., Jute Stem Weevil: 676 train images, Beet Armyworm: 199 train images).
- **Splits**: 6,444 train, 413 validation, 379 test images.
- **Challenges**:
  - Imbalanced classes (e.g., Jute Stem Weevil vs. Beet Armyworm).
  - Variable image sizes (e.g., 612x612 to 3056x4592).
  - Corrupt images (e.g., unreadable files).

## Methodology
### Preprocessing
- **Image Standardization**: Resized images to 256x256 (224x224 for ResNet18) using `transforms.Resize`.
- **Augmentation** (for balanced dataset): Random horizontal flips, rotations (10°), normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]; ImageNet stats for ResNet18).
- **Corrupt Image Handling**: Renamed unreadable files (e.g., to `image1.jpg`) and used `cv2.imread` for validation.
- **Data Loading**: Utilized `ImageFolder` and `DataLoader` (batch size=32, `num_workers=4`, shuffle=True).

### Dual Dataset Approach
- **Balanced Dataset**: Augmented underrepresented classes (e.g., added images for Jute Aphid) using `augment_and_save` to balance class distribution.
- **Imbalanced Dataset**: Used the original dataset without augmentation, as it yielded ~2% higher accuracy.
- **Decision**: Prioritized the imbalanced dataset for final models due to better performance, with balanced dataset code available for reference.

### Model Development
- **Custom CNN**:
  - Built from scratch with layers: `Conv2d(3, 32)`, up to `Conv2d(512, 1024)`, batch normalization, ReLU, max pooling, and fully connected layers (`Linear(1024*1*1, 256)`, `Linear(256, 17)`).
  - Dropout (p=0.4) to prevent overfitting.
- **Pre-trained ResNet18**:
  - Fine-tuned by replacing the final fully connected layer for 17 classes.
  - Used ImageNet weights and normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
- **Hyperparameter Tuning**:
  - Optimized learning rate (0.001), filter sizes, and network depth using TensorBoard to visualize loss/accuracy trends.
- **Ensemble Models**:
  - Combined custom CNN and ResNet18 predictions by averaging softmax probabilities (e.g., Model 7 + Model 3: 97.1% accuracy).
- **Super Ensemble**:
  - Merged balanced and imbalanced dataset ensembles into a single model (`super_ensemble_model.pt`), achieving 98% accuracy.

### Training
- **Hyperparameters**:
  - Optimizer: Adam (lr=0.001)
  - Loss: CrossEntropyLoss
  - Epochs: 50
  - Early Stopping: Patience of 5 epochs based on validation loss
  - Batch Size: 32
- **Efficiency**: Used mixed precision training (`torch.cuda.amp`) and optimized CUDA settings (`cudnn.benchmark=True`).
- **Reproducibility**: Set `random.seed(42)` for consistent results.
- **Model Saving**: Saved models (e.g., `Models/1.pth`, `ensemble_model.pt`, `super_ensemble_model.pt`) using TorchScript.

### API Deployment
- **Framework**: Built a Flask-based API for real-time pest classification.
- **Platform**: Hosted on PythonAnywhere with a responsive webpage for mobile and desktop access.
- **Challenge**: Byte and Wear’s 100MB limit required splitting the model into two files, uploading them, and merging them online.
- **Solution**: Manually split the model (e.g., weights and architecture), uploaded the parts, and merged them on the server, enabling global access.

## Results
- **Test Accuracies**:
  - Model 1 (ResNet18): 90.7%
  - Model 5 (Custom CNN): 71.24%
  - Model 6: 87.0%
  - Ensemble (Model 7 + Model 3): 97.1%
  - Super Ensemble: 98% (macro/weighted average precision, recall, F1-score).
- **Classification Report** (Super Ensemble):
  - Precision/Recall/F1: 0.98 (macro/weighted average) across 379 test samples.
  - High-performing classes: Jute Stem Girdler, Leaf Beetle (1.00); Termite Odontotermes (0.96).
- **Visualizations**:
  - Confusion matrices and loss/accuracy curves plotted with `seaborn` and `matplotlib`.
  - 30 sample predictions saved to `D:/Telegram Downloads/.../PPredicted_Images` with true/predicted labels and confidence scores.

## Challenges and Solutions
1. **Dataset Imbalance**:
   - **Challenge**: Significant class disparity reduces model fairness.
   - **Solution**: Tested balanced dataset with augmentation but chose imbalanced dataset for ~2% higher accuracy.
2. **Variable Image Sizes**:
   - **Challenge**: Inconsistent dimensions (e.g., 612x612 to 3056x4592).
   - **Solution**: Standardized to 256x256 or 224x224.
3. **Corrupt Images**:
   - **Challenge**: Unreadable files caused loading errors.
   - **Solution**: Renamed files and used `cv2.imread` checks, possibly with `SafeImageFolder`.
4. **Overfitting**:
   - **Challenge**: Complex models risked overfitting.
   - **Solution**: Applied dropout (p=0.4), early stopping, and augmentation (for balanced dataset).
5. **Model Size for Deployment**:
   - **Challenge**: Model exceeded PythonAnywhere’s 100MB limit.
   - **Solution**: Split the model into two files, uploaded, and merged online.
6. **Training Stability**:
   - **Challenge**: Potential errors (e.g., `RuntimeError` in early iterations).
   - **Solution**: Debugged tensor shapes and ensured stable training in final models.

## My Journey
Starting as a beginner, I faced challenges balancing this project with other commitments. After a hiatus, I resumed work with determination and completed it 2-3 months ago. This journey taught me resilience, the value of experimentation (e.g., balanced vs. imbalanced datasets), and the power of combining custom and pre-trained models. The result is a robust, practical solution for agricultural pest identification.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd jute-pest-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Requires `torch`, `torchvision`, `opencv-python`, `scikit-learn`, `seaborn`, `matplotlib`, `flask`).
3. Download the dataset and place it in `data/`.
4. Run training scripts:
   ```bash
   python train.py
   ```
5. Deploy the Flask API:
   ```bash
   python app.py
   ```

## Usage
- **Training**: Run `train.py` to train models (custom CNN, ResNet18, ensembles).
- **Inference**: Use `predict.py` for single-image predictions or the Flask API (`app.py`) for web-based predictions.
- **API Access**: Visit the responsive webpage hosted on PythonAnywhere to upload images and get predictions(<https://mohamed333.pythonanywhere.com/>).
- **Demo**: Watch the video demo to see the API in action.

## Future Work
- Explore advanced augmentation (e.g., GANs) to improve balanced dataset performance.
- Automate model splitting/merging for Byte and Wear deployment.
- Optimize model size with pruning or quantization.
- Test on diverse agricultural datasets for broader applicability.

## Acknowledgments
Thanks to the "Jute Pest Dataset" providers and the open-source PyTorch community. This project reflects months of learning, experimentation, and dedication to advancing #AIinAgriculture.

## Contact
Connect with me on [LinkedIn](https://www.linkedin.com/in/mohamed-mahmoud-elsayed/) to discuss this project or collaborate on AI for agriculture!
