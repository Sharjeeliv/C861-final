## **A Comprehensive Study of Deep Learning Models and Hybridization for Sentiment Analysis on the Twitter Dataset**
Author: Sharjeel Mustafa  

### Introduction
This repository benchmarks rule-based, traditional machine learning, and deep learning approaches for multiclass [Twitter sentiment](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) classification. We study the impact of data augmentation and hybrid architectures, showing a clear performance hierarchy: deep learning models outperform classical machine learning, which in turn outperform rule-based baselines.

Sequential models benefit strongly from data augmentation, while convolutional models remain comparatively robust. The best results are achieved using neural networks as feature extractors for linear classifiers, with an LSTM → CNN → SVM hybrid achieving the top performance (Accuracy 0.973, F1 0.973). Overall, the results highlight the importance of both model selection and data preparation in sentiment analysis.

### **Citation**
If you find this work useful, please consider citing:
```
@misc{sharjeelc861m,
  title={A Comprehensive Study of Deep Learning Models and Hybridization for Sentiment Analysis on the Twitter Dataset},
  author={Sharjeel Mustafa},
  year={2025},
  url={https://github.com/Sharjeeliv/C861-final/}
}
```
