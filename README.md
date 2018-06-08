# Adversarial Ranking for Language Generation

## Introcution
This is a tensorflow implementation of [Adversarial Ranking for Language Generation](https://arxiv.org/pdf/1705.11001.pdf) by Kevin Lin<sup>\*</sup>, Dianqi Li<sup>\*</sup>, Xiaodong He, Zhengyou Zhang, Ming-Ting Sun, NIPS 2017. 

## Environment
The code is based on python2.7 and tensorflow 1.2 version. The code is developed and tested using one NVIDIA M40 GPU. 

## Run
```
python main.py
```

## Note
* For a fair comparison, we used same LSTMs units, pre-train and test configurations in [SeqGan](https://github.com/LantaoYu/SeqGAN). If you use `tf.contrib.rnn.LSTMCell` instead of their LSTMs implementation, you will obtain different training results.
* `save/target_params.pkl` is the parameter for the oracle model from [SeqGan](https://github.com/LantaoYu/SeqGAN). `log` folder stores the log of your model training. 
* Ideally, you will receive a nll loss between `8.00-8.50`. However, adversarial training sometimes depends on the quality of pre-train model. 
* More evaluation metrics for adversarial text generation can be refered to: [paper](https://arxiv.org/pdf/1802.01886.pdf) and [repo](https://github.com/geek-ai/Texygen).

## Citing RankGan
if you find RankGan is useful in your research, please consider citing: 
```
@inproceedings{lin2017adversarial,
  title={Adversarial ranking for language generation},
  author={Lin, Kevin and Li, Dianqi and He, Xiaodong and Zhang, Zhengyou and Sun, Ming-Ting},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3155--3165},
  year={2017}
}
```

## Acknowledgements
This code is based on [SeqGan](https://github.com/LantaoYu/SeqGAN). Many thanks for the authors!
