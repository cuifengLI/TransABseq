# TransABseq

The antigen-antibody reaction is a critical event in host defense against pathogens, tumors, immunotherapy, and in vitro disease detection. Due to their unparalleled high specificity, high affinity, and selectivity, antibodies have been widely applied in the development of tools for clinical diagnosis, treatment, and prevention. In this study, we propose a novel model, TransABseq, a computational method specifically designed to predict the effects of missense mutations on antigen-antibody interactions. The model adopts a two-stage architecture: in the first stage, enriched protein language model embeddings are processed through a transformer encoder module and a multiscale convolution module, enabling comprehensive feature extraction and processing. In the second stage, the XGBOOST model is used to perform quantitative outputs based on the fused features. The primary innovation lies in leveraging the multi-layer self-attention mechanism of the transformer to extract higher-level and more abstract representations. At the same time, the multiscale convolution captures features at different hierarchical levels, enhancing feature abstraction capabilities. As a result, this approach significantly improves DDG prediction compared to existing methods. We validated TransABseq using three different cross-validation strategies on two existing datasets and one newly reconstructed dataset. The experimental results demonstrate that our two-stage network architecture and integrated feature fusion approach significantly enhances predictive performance. TransABseq consistently exhibited superior performance across all three datasets, achieving PCCs of 0.607, 0.843, and 0.794, and RMSEs of 1.166, 1.314, and 1.337 kcal/mol, respectively. Furthermore, its robustness extended to blind test datasets, where TransABseq also outperformed existing methods, achieving PCC of 0.721 and RMSE of 0.925 kcal/mol. TransABseq demonstrated higher predictive accuracy and robustness.

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


# Contact

If you are interested in our work, OR, if you have any suggestions/questions about our work, PLEASE contact with us. E-mail: [231210701110@stu.just.edu.cn](mailto:231210701110@stu.just.edu.cn)
