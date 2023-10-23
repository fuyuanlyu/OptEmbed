# OptEmbed
This repository contains PyTorch Implementation of CIKM 2022 research-track oral paper:
  - **OptEmbed**: Learning Optimal Embedding Table for Click-through Rate Prediction [paper](https://arxiv.org/abs/2208.04482).

Notes: This repository is under debugging. We do not guarantee the reproducibility of our result on current version of code. We are actively debugging the reproducibility issue. Please check our code later.

### Data Preprocessing

You can prepare the Criteo data in the following format. Avazu and KDD12 datasets can be preprocessed by calling its own python file.

```
python datatransform/criteo2tf.py --store_stat --stats PATH_TO_STORE_STATS
		--dataset RAW_DATASET_FILE --record PATH_TO_PROCESSED_DATASET \
		--threshold 2 --ratio 0.8 0.1 0.1 \
```

Then you can find a `stats` folder under the `PATH_TO_STORE_STATS` folder and your processed files in the tfrecord format under the `PATH_TO_PROCESSED_DATASET` folder. You should update line 181-190 in `train.py` and line 200-209 in `evo.py` corresponding.


### Run

Running OptEmbed requires the following three phases. First is supernet training:
```
python supernet.py --gpu 0 --dataset $YOUR_DATASET --model $YOUR_MODEL \
        --batch_size 2048 --epoch 30 --latent_dim 64 \
        --mlp_dims [1024, 512, 256] --mlp_dropout 0.0 \
        --optimizer adam --lr $LR --wd $WD \
        --t_lr $LR_T --alpha $ALPHA \
```

Second is evolutionary search:
```
python evolution.py --gpu 0 --dataset $YOUR_DATASET --model $YOUR_MODEL \
        --batch_size 2048 --epoch 30 --latent_dim 64 \
        --mlp_dims [1024, 512, 256] --mlp_dropout 0.0 \
        --keep_num 0 --mutation_num 10 \
        --crossover_num 10 --m_prob 0.1 \ 
```

Third is retraining:
```
python retrain.py --gpu 0 --dataset $YOUR_DATASET --model $YOUR_MODEL \
        --batch_size 2048 --epoch 30 --latent_dim 64 \
        --mlp_dims [1024, 512, 256] --mlp_dropout 0.0 \
        --optimizer adam --lr $LR --wd $WD \
```


### Hyperparameter Settings

Notes: Due to the sensitivity of OptEmbed, we do not guarantee that the following hyper-parameters will be 100% optimal in your own preprocessed dataset. Kindly tune the hyper-parameters a little bit. If you encounter any problems regarding hyper-parameter tuning, you are welcomed to contact the first author directly.

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


### Bibliography

Kindly cite our paper using the following bibliography:
```
@inproceedings{OptEmbed,
  author       = {Fuyuan Lyu and
                  Xing Tang and
                  Hong Zhu and
                  Huifeng Guo and
                  Yingxue Zhang and
                  Ruiming Tang and
                  Xue Liu},
  title        = {OptEmbed: Learning Optimal Embedding Table for Click-through Rate
                  Prediction},
  booktitle    = {Proceedings of the 31st {ACM} International Conference on Information
                  {\&} Knowledge Management},
  pages        = {1399--1409},
  address      = {Atlanta, GA, USA},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3511808.3557411},
  doi          = {10.1145/3511808.3557411},
  timestamp    = {Mon, 26 Jun 2023 20:40:13 +0200},
  biburl       = {https://dblp.org/rec/conf/cikm/Lyu0ZG0TL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


