The purpose of this repo is to classify defects on the WM-811k dataset, which
includes wafers classified as either 'none' or one of the many defective classes. 
The state of the art is (Wafer map defect classification using deep learning 
framework with data augmentation on imbalance datasets) by Tsung-Han Tsai & 
Chieng-Yang Wang.

Our work is based on Zernike and Geometric feature engineering, with a total of
61 features fed into a two-layer MLP, achieving macro-F1 of 0.87 on WM-811K using
6,345 parameters and no data augmentation, compared to Tsai & Wang (2025) which 
reports 0.93 macro-F1 using 46,772 parameters plus a separate G2LGAN augmentation
system (~7× larger inference model, with additional training-time GAN infrastructure).

<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/9eacb1d5-383c-4ae8-ab4b-f90e5ddec092" />
