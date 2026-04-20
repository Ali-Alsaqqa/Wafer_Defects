The purpose of this repo is to classify defects on the WM-811k dataset, which includes wafers classified as either 'none' or one of the many defective classes. The state of the art is Wafer map defect classification using deep learning framework with data augmentation on imbalance datasets by Tsung-Han Tsai & Chieng-Yang Wang (2025).

Our work is based on Zernike and geometric feature engineering, with a total of 61 features fed into a two-layer MLP, achieving macro-F1 of 0.87 on WM-811K using 6,345 parameters and no data augmentation, compared to Tsai & Wang (2025) which reports 0.93 macro-F1 using 46,772 parameters plus a separate G2LGAN augmentation system (~7× larger inference model, with additional training-time GAN infrastructure).

The feature set combines four families: Zernike polynomial magnitudes up to order 8 (45 rotation-invariant coefficients describing radial and angular structure on the unit disk), angle-radius ring features (mean and max defect density per radial ring, collapsed over angle for rotation invariance), connected-component statistics on the native binary defect mask (component count, largest-component fraction, eccentricity, elongation), and radial extent of the largest connected component (max rho, rho span). L1 regularization on the first MLP layer is used to prune uninformative features, reducing the input dimension from 69 to 61 while preserving accuracy.

The dataset is the publicly available WM-811K (downloadable from Kaggle at https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map). The pipeline is implemented in a single Colab notebook: mount Google Drive with the pickle file placed at the path indicated in the load cell, then run all cells. Training runs in under a minute on a free-tier Colab GPU; feature extraction over the labeled subset (~172k wafers) takes the majority of the runtime.
Per-class F1 on the test set: none 0.90, Center 0.96, Donut 0.85, Edge-Loc 0.84, Edge-Ring 0.97, Loc 0.80, Near-full 0.92, Random 0.88, Scratch 0.70. The primary remaining confusions are between Loc, Edge-Loc, and Scratch, which is consistent with known label noise between these classes in WM-811K.

<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/9eacb1d5-383c-4ae8-ab4b-f90e5ddec092" />

<img width="692" height="590" alt="image" src="https://github.com/user-attachments/assets/e71cfb37-2204-4e76-b1f5-ed844733f15c" />

The dataset can be downloaded from https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map. 
Training was done using Google's colab platform, using T4 GPUs.
