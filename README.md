# OptEmbed
This repository contains PyTorch Implementation of CIKM 2022 submission paper:
  - **OptEmbed**: Learning Optimal Embedding Table for Click-through Rate Prediction [paper](https://arxiv.org/abs/2208.04482).


### Run

Running OptEmbed requires the following three phases. First is supernet training:
```
python train.py --gpu 0 --dataset $YOUR_DATASET \
        --method $YOUR_METHOD --model $YOUR_MODEL \
        --batch_size 2048 --epoch 30 --latent_dim 64 \
        --mlp_dims [1024, 512, 256] --mlp_dropout 0.0 \
        --optimizer adam --lr $LR --wd $WD \
        --arch_lr $ARCH_LR --alpha $ALPHA --thre_init 0.0 \
```

Second is evolutionary search:
```
python evo.py --gpu 0 --dataset $YOUR_DATASET \
        --model $YOUR_MODEL \
        --batch_size 2048 --epoch 30 --latent_dim 64 \
        --mlp_dims [1024, 512, 256] --mlp_dropout 0.0 \
        --keep_num 0 --mutation_num 10 \
        --crossover_num 10 --m_prob 0.1 \ 
```

Third is retraining:
```
python train.py --gpu 0 --dataset $YOUR_DATASET --retrain \
        --method $YOUR_METHOD --model $YOUR_MODEL \
        --batch_size 2048 --epoch 30 --latent_dim 64 \
        --mlp_dims [1024, 512, 256] --mlp_dropout 0.0 \
        --optimizer adam --lr $LR --wd $WD \
```


### Hyperparameter Settings

#### Supernet Training

Here we list all the hyper-parameters we used in the supernet training stage for each model in the following table. 

| Model\Dataset | Criteo                                                       | Avazu                                                        | KDD12                                                        |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DeepFM        | _lr_=3e-5, l<sub>2</sub>=1e-3,  _lr_<sup>t</sup>=1e-4, $\alpha$=1e-4 | _lr_=3e-4, l<sub>2</sub>=1e-5,  _lr_<sup>t</sup>=1e-4, $\alpha$=1e-6 | _lr_=3e-5, l<sub>2</sub>=1e-5,  _lr_<sup>t</sup>=1e-4, $\alpha$=1e-5 |
| DCN           | _lr_=3e-4, l<sub>2</sub>=1e-5, _lr_<sup>t</sup>=1e-4, $\alpha$=1e-5 | _lr_=1e-4, l<sub>2</sub>=3e-5, _lr_<sup>t</sup>=1e-4, $\alpha$=1e-4 | _lr_=1e-5, l<sub>2</sub>=1e-6, _lr_<sup>t</sup>=1e-4, $\alpha$=1e-5 |
| FNN           | _lr_=3e-4, l<sub>2</sub>=1e-5, _lr_<sup>t</sup>=1e-4, $\alpha$=1e-5 | _lr_=1e-4, l<sub>2</sub>=3e-5, _lr_<sup>t</sup>=1e-4, $\alpha$=1e-4 | _lr_=1e-5, l<sub>2</sub>=1e-6, _lr_<sup>t</sup>=1e-4, $\alpha$=1e-5 |
| IPNN          | _lr_=3e-4, l<sub>2</sub>=1e-5, _lr_<sup>t</sup>=3e-5, $\alpha$=1e-6 | _lr_=1e-4, l<sub>2</sub>=3e-5, _lr_<sup>t</sup>=1e-4, $\alpha$=1e-4 | _lr_=1e-5, l<sub>2</sub>=1e-6, _lr_<sup>t</sup>=1e-4, $\alpha$=1e-5 |

The following procedure describes how we determine these hyper-parameters:

First, we determine the hyper-parameters of the basic models by grid search: learning ratio and l<sub>2</sub> regularization. We select the optimal learning ratio _lr_ from \{1e-3, 3e-4, 1e-4, 3e-5, 1e-5\} and l<sub>2</sub> regularization from \{1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6\}. Adam optimizer and Xavier initialization are adopted. We empirically set the batch size to be 2048, embedding dimension to be 64, MLP structure to be [1024, 512, 256].

Second, we tune the hyper-parameters introduced by the OptEmbed method: learning ratio for threshold _lr_<sup>t</sup>, threshold regularization $\alpha$. We select the optimal threshold learning ratio _lr_<sup>t</sup> from \{1e-2, 1e-3, 1e-4\} and threshold regularization $\alpha$ from \{1e-4, 3e-5, 1e-5, 3e-6, 1e-6\}. During tuning process, we fix the optimal learning ratio _lr_ and l<sub>2</sub> regularization determined in the first step. We select the optimal hyper-parameters based on the performance of supernet on validation set.



#### Evolutionary Search

For the evolutionary search stage, we adopt the same hyper-parameters from previous work\cite{One-shot}. For all experiments, mutation number $n<sub>m</sub> = 10$, crossover number $n<sub>c</sub> = 10$, max iteration $_T_ = 30$, mutation probability $_prob_ = 0.1$ and $_k_ =15$.



#### Retraining


For the retraining stage, we adopt the same learning ratio _lr_ and l<sub>2</sub> regularization from the supernet training stage.



