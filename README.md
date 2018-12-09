# 중간 과제 - 마감 기한, 2018년 5월 17일 목요일까지 

순환 신경망을 사용하여 [이것](https://github.com/niderhoff/nlp-datasets)과 같이 문장 데이터를 분류하세요. 사용처가 흥미로울수록 더 좋습니다. 훌륭한 문서 처리는 추가 점수를 받습니다. 깃허브 링크를 댓글란에 제출하세요. 행운을 빕니다! 

## 개요

이 코드는 Siraj Raval이 유튜브에 올린 '이력서를 위한 인공지능' [강의 영상](https://youtu.be/p3SKx5C04qg)을 위한 것입니다. 이 CNN을 사용하여 [여기](http://barbizonmodeling.com/resumes/)에 있는 이력서 데이터를 분류할 수 있습니다.

**[이 코드는 "텐서플로를 활용한 문장 분류 CNN 구현" 블로그 글에 귀속됩니다.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

이는 텐서플로에 있는 Kim의 [문장 분류를 위한 순환 신경망](http://arxiv.org/abs/1408.5882) 연구를 살짝 간단하게 변형한 형태입니다.

## 필요 조건

- Python 3
- Tensorflow > 0.12
- Numpy

## 학습

변수 출력:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

학습:

```bash
./train.py
```

## 평가

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Checkpoint dir을 학습에서 나온 출력으로 교체해 보세요. 여러분의 데이터를 사용하려면, `eval.py` 스크립트를 변경하여 여러분의 데이터를 불러오세요.


## 참고 문헌

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

## Credits

이 코드의 저작권은 [dennybritz](https://github.com/dennybritz/cnn-text-classification-tf)에게 있습니다. 저는 사람들이 좀 더 쉽게 시작할 수 있도록 정리했을 뿐입니다.
