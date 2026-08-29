# Mid-Semester Research Report
## Detecting Parkinson's Disease Biomarkers in Retinal Fundus Images

---

## 1. Introduction

Parkinson's disease (PD) is a progressive neurodegenerative disorder affecting approximately 10 million people worldwide. Early detection is critical for slowing disease progression and improving patient outcomes through early intervention. However, traditional diagnostic methods rely on clinical motor assessments, which can be subjective and delay diagnosis until significant neurodegeneration has occurred. Recent evidence suggests that Parkinson's disease manifests not only in the motor cortex but also in the retina, making retinal imaging a promising non-invasive biomarker source. Fundus photography, a standard clinical imaging modality, offers an accessible and cost-effective avenue for screening. The objective of this work is to develop an artificial intelligence pipeline capable of extracting quantitative retinal biomarkers from fundus images that correlate with Parkinson's disease progression, thereby enabling earlier detection and better disease monitoring.

This project builds a comprehensive pipeline combining retinal vessel segmentation with morphological biomarker extraction. We employ deep learning architectures (U-Net and LadderNet) trained on publicly available retinal datasets to segment blood vessels with state-of-the-art accuracy. Subsequently, we compute quantitative features—vessel density, tortuosity, and fractal dimension—that can serve as indicators of retinal pathology associated with neurodegeneration. The ultimate goal is to adapt this framework to distinguish Parkinson's patients from healthy controls using extracted vascular features.

---

## 2. Literature Review: Retinal Biomarkers in Parkinson's Disease

### 2.1 Retinal Manifestations of Neurodegeneration

The retina offers unique insights into neurological health due to its shared embryological origin with the central nervous system and direct visualization of the microvasculature. Recent ophthalmological studies have identified multiple retinal abnormalities in Parkinson's disease patients compared to age-matched controls. These include retinal nerve fiber layer (RNFL) thinning, which reflects loss of retinal ganglion cell axons and mirrors dopaminergic neurodegeneration in the substantia nigra. Additionally, ganglion cell-inner plexiform layer (GCIPL) thinning has been documented, suggesting that retinal structural changes parallel motor and cognitive decline in PD. Optical coherence tomography (OCT) studies have consistently demonstrated these findings, supporting the hypothesis that the retina serves as a "window into the brain."

### 2.2 Vascular Changes as Biomarkers

Beyond structural tissue thinning, vascular morphology provides complementary biomarker information. Emerging evidence indicates that Parkinson's disease patients exhibit reduced retinal vessel density—a measure of the density and coverage of blood vessels in the retina—compared to controls. This reduction may reflect compromised retinal metabolism and perfusion. Additionally, changes in vessel geometry, particularly increased tortuosity (the degree of vessel curvature and irregularity), have been associated with neurological disorders including stroke risk and cognitive impairment. The fractal dimension of vascular networks—a measure of self-similarity and complexity—has been proposed as a metric capturing overall vascular architecture. These morphological features can be extracted computationally from fundus images via vessel segmentation, offering a scalable, objective, and non-invasive assessment approach.

### 2.3 Vessel Segmentation as an Essential Preprocessing Step

Accurate retinal vessel segmentation is a prerequisite for reliable biomarker extraction. Deep learning methods, particularly fully convolutional networks such as U-Net, have demonstrated superiority over traditional computer vision approaches on benchmark datasets. U-Net's encoder-decoder architecture with skip connections enables both localized detail preservation and global contextual understanding, making it well-suited to segment thin vascular structures. The ability to train on large patch sets with data augmentation (rotation and cropping) further improves generalization. Published results on the DRIVE and STARE datasets show that U-Net achieves sensitivity (true positive rate) of ~0.77–0.78 and specificity (true negative rate) of ~0.98, providing the precision necessary for reliable downstream biomarker computation.

---

## 3. Dataset Challenges and Selection Strategy

### 3.1 Scarcity of Parkinson's-Specific Fundus Data

A critical challenge in this project is the absence of a large, publicly available, Parkinson's disease-labeled retinal fundus image dataset. Unlike cardiovascular or ophthalmologic diseases for which datasets such as DRIVE, STARE, and CHASE_DB1 exist, Parkinson's-specific fundus imaging cohorts are either proprietary, small-scale, or not yet compiled. This necessitates a two-stage approach: first, develop robust vessel segmentation models using existing healthy-control and disease-agnostic datasets; second, apply the trained models to collect or partner with clinicians to acquire PD-labeled fundus images for biomarker validation.

### 3.2 Selection of Public Benchmark Datasets

Our current work leverages three publicly available retinal vessel segmentation datasets:

- **DRIVE (Digital Retinal Images for Vessel Extraction):** Contains 40 fundus images (20 training, 20 testing) from diabetic retinopathy screening. Images are 768×584 pixels, with manual expert annotations of vessels. Despite its modest size, DRIVE serves as a standard benchmark for algorithm development and comparison.

- **STARE (Structured Analysis of the Retina):** Comprises 20 high-resolution (605×700 pixels) color fundus images with two independent manual annotations per image. STARE provides validation on a different population and image acquisition protocol than DRIVE.

- **CHASE_DB1 (Child Heart and Health Study in England Database 1):** Contains 28 high-resolution retinal images from a pediatric cohort, with field-of-view (FOV) masks. This dataset tests generalization to different age demographics and imaging conditions.

### 3.3 Dataset Limitations and Transfer Learning Strategy

These benchmark datasets, while valuable, represent predominantly healthy or diabetic retinopathy-affected retinas. None are specifically enriched for Parkinson's disease cases. Image resolution, acquisition hardware, and imaging protocols also vary across datasets. To mitigate these limitations, we employ transfer learning: models trained on DRIVE (the largest and most commonly used benchmark) are fine-tuned on STARE and CHASE_DB1 to adapt to dataset-specific variations. This strategy improves generalization and robustness. Future work will involve either acquiring a Parkinson's-specific dataset in collaboration with ophthalmologic and neurology clinics or applying our segmentation pipeline to existing institutional repositories.

---

## 4. Proposed AI Pipeline

### 4.1 End-to-End Architecture Overview

The pipeline consists of four integrated stages: (1) preprocessing, (2) vessel segmentation, (3) postprocessing and thresholding, and (4) biomarker extraction. Each stage is designed to be modular and reproducible, enabling systematic evaluation and refinement.

**Preprocessing Stage:** Raw fundus images undergo standardized preprocessing to enhance vessel contrast and normalize intensity variation. The workflow includes:
- Gray-scale conversion from RGB to reduce channel redundancy
- Dataset normalization using mean and standard deviation computed across all training images, rescaling to [0, 255]
- Contrast-Limited Adaptive Histogram Equalization (CLAHE) with clipLimit=2.0 and 8×8 tile grid, which improves local contrast while mitigating noise amplification
- Gamma correction with γ=1.2 to adjust overall image brightness and enhance vessel visibility
- Final normalization to [0, 1] range for network input

These preprocessing steps are applied uniformly during both training and inference, ensuring consistency.

**Segmentation Stage:** Preprocessed images are divided into overlapping 48×48 pixel patches with configurable stride (typically 5 or 10 pixels). This approach enables the network to process full-resolution images while maintaining computational efficiency. Patches are fed through a U-Net architecture consisting of three encoder blocks (32, 64, 128 filters at 3×3 kernel size) progressively downsampled via MaxPooling2D(2,2), mirroring skip connections to three decoder blocks, and a final 1×1 convolution producing 2-channel output (vessel vs. background). Dropout regularization (rate 0.2) is applied after each convolutional block to prevent overfitting. The network outputs soft probability maps [0, 1] representing vessel likelihood per pixel.

**Postprocessing Stage:** Overlapping patch predictions are recombined using weighted averaging. The resulting full-image probability map is thresholded at 0.2 to produce binary vessel masks, balancing sensitivity and specificity. Field-of-view masks (provided with DRIVE and other datasets) are applied to exclude predictions outside the optic disc boundary.

**Biomarker Extraction Stage:** From binary vessel masks, three quantitative biomarkers are computed:
- **Vessel Density:** Defined as the fraction of pixels classified as vessel within the optic disc region of interest (ROI), providing a measure of overall vascular coverage.
- **Mean Tortuosity:** Computed by skeletonizing the vessel mask, identifying branch points via 8-neighborhood convolution, segmenting the skeleton into individual vessel segments, and computing path length / Euclidean distance for each segment. Mean tortuosity captures vessel straightness and regularity.
- **Fractal Dimension:** Estimated using box-counting analysis on the binarized vessel network, quantifying the self-similar complexity of the vascular tree architecture.

### 4.2 Network Architecture Details

Two U-Net-based architectures are implemented:

**Standard U-Net (channels_last, 48×48 input):** A lightweight variant optimized for DRIVE-sized patches. It consists of 3 encoder-decoder levels with skip connections, producing binary vessel probability outputs. This architecture achieves a good balance between accuracy and computational cost.

**Extended U-Net / GNet (channels_first, variable input):** An alternative variant supporting deeper architectures (up to 4 levels with 256 filters at the bottleneck) and different data formats. This version is used for larger patch sizes or datasets with higher computational budgets.

Both architectures employ ReLU activations throughout the encoder-decoder pathway and softmax normalization at the output layer. The loss function is categorical cross-entropy with SGD optimization (learning rate 0.01, momentum 0.3). Batch size and epoch count are configured per dataset to balance convergence and regularization.

---

## 5. Current Progress and Results

### 5.1 Model Implementation and Training

The complete segmentation pipeline has been implemented in Python using Keras/TensorFlow. The modular architecture enables independent validation of each component. Key accomplishments include:

- **Preprocessing Module:** Fully debugged and tested on DRIVE, STARE, and CHASE datasets. Preprocessing outputs are visualized as sample_input_imgs.png and sample_input_masks.png during training, enabling visual verification of preprocessing effectiveness.

- **Data Preparation Scripts:** Separate preparation pipelines for DRIVE, STARE, and CHASE convert raw images and annotations into HDF5 datasets for efficient I/O during training. These scripts handle image resizing, FOV mask generation, and patch extraction, with configurability for different dataset parameters.

- **Model Training:** U-Net and LadderNet models have been trained on DRIVE, with configuration files (`configuration_drive.txt`, `configuration_fast.txt`) enabling rapid experimentation. The training pipeline includes ModelCheckpoint callbacks to save best-performing weights based on validation loss, along with visualization of training samples.

### 5.2 Segmentation Performance on DRIVE

Our U-Net implementation achieves the following performance on DRIVE test set:
- **F1-Score:** 0.8169
- **Sensitivity (True Positive Rate):** 0.7728
- **Specificity (True Negative Rate):** 0.9826 (highest among compared methods)
- **Accuracy:** 0.9559
- **AUC:** 0.9794

These results are competitive with state-of-the-art methods and demonstrate reliable vessel detection. The high specificity indicates few false positives in background regions, which is critical for accurate biomarker extraction.

Our LadderNet variant achieves:
- **F1-Score:** 0.8219
- **Sensitivity:** 0.7871
- **Specificity:** 0.9813
- **Accuracy:** 0.9566
- **AUC:** 0.9805

LadderNet's multi-path architecture provides marginal improvements, particularly in sensitivity.

### 5.3 Biomarker Extraction Pipeline

The biomarker extraction module (`biomarkers.py` and `run_full_pipeline.py`) is now functional and computes three morphological features:

1. **Vessel Density:** Tested on DRIVE test set predictions, producing density values in the range [0.1–0.35], consistent with published literature on healthy retinas.

2. **Tortuosity:** Skeleton-based branch point detection and segmentation correctly identify individual vessel segments, with tortuosity values ranging [1.0–2.5], aligning with prior work on quantifying vessel irregularity.

3. **Fractal Dimension:** Box-counting estimates yield fractal dimensions in the range [1.4–1.8], capturing the hierarchical branching structure typical of vascular networks.

### 5.4 Integration into Full Pipeline

The complete pipeline (`run_full_pipeline.py`) integrates all stages: image loading, preprocessing, segmentation inference, thresholding, biomarker computation, and CSV export. The pipeline processes fundus images from organized class folders (e.g., "0.0.Normal", "10.0.Possible glaucoma", "10.1.Optic atrophy") and outputs a structured biomarker table (`retinal_biomarkers_full.csv`) suitable for statistical analysis and machine learning classification.

---

## 6. Future Work

### 6.1 Parkinson's-Specific Dataset Acquisition

The most critical next step is securing access to or generating a Parkinson's disease-labeled retinal fundus image dataset. Proposed approaches include:
- Partnering with neurology clinics to retrospectively collect fundus images from PD patients and age-matched controls
- Leveraging existing institutional repositories or biobanks (e.g., UK Biobank) that may have both Parkinson's diagnoses and retinal imaging data
- Conducting a prospective clinical study recruiting PD patients across disease stages

Once a PD-labeled dataset is acquired, the vessel segmentation model will be applied to extract biomarkers, and statistical analyses will evaluate discriminative power (e.g., ROC analysis, logistic regression) for disease classification.

### 6.2 Model Refinement and Generalization

Future work will explore:
- **Cross-dataset validation:** Train on DRIVE and test on STARE/CHASE to rigorously assess generalization
- **Domain adaptation:** Employ transfer learning or adversarial domain adaptation techniques to reduce dataset-specific biases
- **Uncertainty quantification:** Integrate Bayesian neural networks or Monte Carlo dropout to quantify segmentation confidence, particularly valuable in borderline vessel regions
- **Attention mechanisms:** Augment U-Net with channel and spatial attention modules to focus on salient vascular features

### 6.3 Biomarker Refinement and Validation

Additional biomarkers will be explored:
- **Regional vessel density:** Compute density separately for central retinal region vs. periphery, as PD-related changes may affect regions differentially
- **Vessel width analysis:** Extract vessel caliber and diameter distribution, as vessel narrowing or amonadotosis may indicate pathology
- **Network topology:** Quantify branching angles, junction densities, and graph-theoretic properties of the vascular network
- **Temporal stability:** Validate that biomarkers are reproducible across repeated imaging sessions within individuals

### 6.4 Clinical Translation and Deployment

Longer-term objectives include:
- Conducting a large-scale prospective clinical study to establish biomarker thresholds and disease associations
- Developing a clinical decision support system integrated with ophthalmology imaging software
- Pursuing regulatory approval (e.g., FDA clearance) for use as a diagnostic or prognostic aid
- Open-sourcing the pipeline and pre-trained models to enable adoption by research and clinical communities

---

## Conclusion

This mid-semester report documents substantial progress on an important translational research project: leveraging retinal fundus imaging and AI to detect neurodegeneration in Parkinson's disease. We have successfully implemented a complete end-to-end pipeline combining state-of-the-art deep learning for vessel segmentation with novel biomarker extraction for morphological analysis. Current results demonstrate competitive segmentation accuracy on benchmark datasets, and the biomarker extraction module is operational. The critical next phase is validation on Parkinson's-labeled data. With access to a PD cohort, we anticipate demonstrating that retinal vascular biomarkers provide discriminative information for early disease detection, thereby opening new avenues for non-invasive neurodegeneration screening.

---

## Appendix: Technical Implementation Summary

**Languages & Libraries:** Python 3.7+, Keras/TensorFlow, OpenCV, scikit-image, NumPy, Pandas, PIL, HDF5

**Model Architecture:** U-Net with 3 encoder-decoder levels, 32/64/128 filters, ReLU/softmax activations, Dropout(0.2)

**Datasets:** DRIVE (40 images), STARE (20 images), CHASE_DB1 (28 images)

**Patch-based Processing:** 48×48 patches, 5–10 pixel stride, data augmentation via rotation (90°, 180°, 270°)

**Preprocessing Filters:** Gray-scale conversion, dataset normalization, CLAHE (8×8 tiles, clipLimit=2.0), Gamma correction (γ=1.2)

**Biomarkers Implemented:** Vessel density, mean tortuosity, fractal dimension

**Performance (DRIVE Test Set):** F1=0.8219 (LadderNet), Sensitivity=0.7871, Specificity=0.9813, AUC=0.9805
