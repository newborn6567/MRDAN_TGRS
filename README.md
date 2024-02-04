# MRDAN
The code of the paper [Multi-Representation Dynamic Adaptation Network for Remote Sensing Scene Classification](https://ieeexplore.ieee.org/document/9930794) (IEEE TGRS 2022)

## Abstract
In recent years, convolutional neural networks (CNNs) have made significant progress in remote sensing scene classification (RSSC) tasks. Because obtaining a large number of labeled images is time-consuming and expensive and the generalization ability of supervised models is limited, domain adaptation is widely introduced into RSSC. However, existing adaptation approaches mainly aim to align the distribution of features in a single representation space, which results in losing information and limiting the spatial range for extracting domain-invariant features. In addition, some of the methods simultaneously align pixel-level (local) and image-level (global) features for better results but suffer from searching for the best weight of the two parts manually, which is time-consuming and computing-expensive. To overcome the above issues, a novel feature fusion-and-alignment approach named Multi-Representation Dynamic Adaption Network (MRDAN) is proposed for cross-domain RSSC. Concretely, a Feature-Fusion Adaptation (FFA) module is embedded into the network, which maps samples to multiple representations and fuses them to obtain a broader domain-invariant feature space. Based on this hybrid space, we introduce a cross-domain Dynamic Feature-Alignment Mechanism (DFAM) to quantitatively evaluate and adjust the relative importance of the local and global adaptation losses during domain adaptation. The experimental results on the 12 transfer tasks between the UC Merced land-use, WHU-RS19, AID, and RSSCN7 data sets demonstrate the effectiveness of the proposed MRDAN over the state-of-the-art domain adaptation methods in RSSC.
