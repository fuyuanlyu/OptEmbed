# OptEmbed
This repository contains PyTorch Implementation of CIKM 2022 paper: OptEmbed: Learning Optimal Embedding Table for Click-through Rate Prediction.





### Hyperparameter Settings

#### Supernet Training

Here we list all the hyper-parameters we used in the supernet training stage for each model in the following table. The following procedure describes how we determine these hyper-parameters. 

| Model\Dataset | Criteo                                                       | Avazu                                    | KDD12                                    |
| ------------- | ------------------------------------------------------------ | ---------------------------------------- | ---------------------------------------- |
| DeepFM        | lr=3e-5, l2=1e-3,  lr^t=1e-4, alpha=1e-4                     | lr=3e-4, l2=1e-5,  lr^t=1e-4, alpha=1e-6 | lr=3e-5, l2=1e-5,  lr^t=1e-4, alpha=1e-5 |
| DCN           | lr=3e-4, l2=1e-5, lr^t=1e-4, alpha=1e-5                      | lr=1e-4, l2=3e-5, lr^t=1e-4, alpha=1e-4  | lr=1e-5, l2=1e-6, lr^t=1e-4, alpha=1e-5  |
| FNN           | lr=3e-4, l2=1e-5, lr^t=1e-4, alpha=1e-5                      | lr=1e-4, l2=3e-5, lr^t=1e-4, alpha=1e-4  | lr=1e-5, l2=1e-6, lr^t=1e-4, alpha=1e-5  |
| IPNN          | _lr_=3e-4, l<sub>2</sub>=1e-5, _lr_<sup>t</sup>=3e-5, $\alpha$=1e-6 | lr=1e-4, l2=3e-5, lr^t=1e-4, alpha=1e-4  | lr=1e-5, l2=1e-6, lr^t=1e-4, alpha=1e-5  |

First, we determine the hyper-parameters of the basic models by grid search: learning ratio and $l_2$ regularization. We select the optimal learning ratio _lr_ from \{1e-3, 3e-4, 1e-4, 3e-5, 1e-5\} and $l_2$ regularization from \{1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6\}. Adam optimizer and Xavier initialization are adopted. We empirically set the batch size to be 2048, embedding dimension to be 64, MLP structure to be [1024, 512, 256].

Second, we tune the hyper-parameters introduced by the OptEmbed method: learning ratio for threshold lr^t, threshold learning ratio decay $\gamma$, threshold regularization $\alpha$. We select the optimal threshold learning ratio $\text{lr}^\text{t}$ from \{1e-2, 1e-3, 1e-4\} and threshold regularization $\alpha$ from \{1e-4, 3e-5, 1e-5, 3e-6, 1e-6\}. During tuning process, we fix the optimal learning ratio and $l_2$ regularization determined in the first step. We select the optimal hyper-parameters based on the performance of supernet on validation set.



#### Evolutionary Searching

Additionally, for the evolutionary search stage, we adopt the same hyper-parameters from previous work\cite{One-shot}. For all experiments, mutation number $n_m = 10$, crossover number $n_c = 10$, max iteration $T=30$, mutation probability $prob = 0.1$ and $k=15$.



#### Retraining


Finally, for the retraining stage, we adopt the same learning rate and weight decay from Table \ref{Table:param}.





