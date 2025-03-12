# TransABseq

The antigen-antibody interaction represents a critical mechanism in host defense, contributing to pathogen neutralization, tumor surveillance, immunotherapy, and in vitro disease detection. Owing to their exceptional specificity, affinity, and selectivity, antibodies have been extensively utilized in the development of clinical diagnostic, therapeutic, and prophylactic strategies. In this study, we propose TransABseq, a novel computational framework specifically designed to predict the effects of missense mutations on antigen-antibody interactions. The model’s innovative two-stage architecture enables comprehensive feature analysis: in the first stage, multiple embeddings of protein language models are processed through a Transformer encoder module and a multiscale convolutional module; in the second stage, the XGBOOST model is used to perform quantitative output based on the deeply fused features. A critical advancement contributing to the effectiveness of TransABseq is the deep feature fusion strategy, which reveals the biochemical properties of proteins. By leveraging the multi-layer self-attention mechanism of the Transformer to capture complex global dependencies within sequences, and mining features at different hierarchical levels through multiscale convolution, the feature abstraction capability of TransABseq is significantly enhanced. We evaluated TransABseq through three distinct cross-validation strategies on two established benchmarks and a newly reconstructed dataset. As a result, TransABseq achieved average PCC values of 0.607, 0.843, and 0.794, and average RMSE values of 1.166, 1.314, and 1.337 kcal/mol in 10-fold cross-validation. Furthermore, its robustness and predictive accuracy were validated on blind test datasets, where TransABseq outperformed existing methods, enabling it to attain a PCC of 0.721 and an RMSE of 0.925 kcal/mol. 

# Install Dependencies

```
    - absl-py==0.15.0
    - h5py==2.10.0
    - joblib==1.3.2
    - keras==2.4.3
    - matplotlib==3.2.0
    - numpy==1.19.5
    - pandas==1.1.5
    - scikit-learn==1.2.2
    - scipy==1.7.3
    - tensorboard==2.11.2
    - tensorboard-data-server==0.6.1
    - tensorboard-plugin-wit==1.8.1
    - tensorflow-estimator==2.5.0
    - tensorflow-gpu==2.4.0
    - python==3.8
```

# Dataset

We provide the datasets file used in this paper, namely  datasets.

# Codes and Run

We provide the model file used in this paper, namely `models/model.py`.
We provide two essential component modules from the paper, namely `models/Encoder.py` and  `models/Mutil_scale_prot.py`, respectively.
To train your own data, use the `models/train_kfold_cross_validation_S1586(xgboost).py` , `models/train_CV3_cross_validation_S1586(xgboost).py` or `models/train_kfold_cross_validation_S1586(no xgboost).py`.
To validate the model's performance on the independent test set, please use the `models/test_indep_HM86(xgboost).py` , `models/test_indep_HM86(no xgboost).py`, `save_model/model_CV1_161(xgboost).h5` and `save_model/xgb_model.pkl` file.

# TransABseq Prediction Performance Comparison

![TransABseq Prediction Performance Comparison0](images/compare_TransABseq_label.jpg)
![TransABseq Prediction Performance Comparison1](images/two_row_six_images.jpg)

# Contact

If you are interested in our work, OR, if you have any suggestions/questions about our work, PLEASE contact with us. E-mail: [231210701110@stu.just.edu.cn](mailto:231210701110@stu.just.edu.cn)
