---
layout: post
title: "BERT Classifier Running in Goole Colab (TPU & GPU)"
math: true
---

## Introduction

I recently took part of the [Jigsaw Unintended Bias in Toxicity Classification ](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) Kaggle competition.

My final solution was a combined ensemble of multiple BERT and GPT2 models built using the popular [PyTorch implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT). All training was done either using cloud P100s or on my local 1070 Ti GPU. I had some remaining free credits on Google Cloud so I did not mind, but, after the competition, it got me thinking on how I could have done the same training while reducing the overall compute cost and time (if possible). That is what I will explore in this post.

I will compare the performances of my local GPU, Colab using a GPU (a slow Tesla K80) and Colab using a TPU. What I am really interested about is this free TPU option. How fast is it? Should I have been using it during the competition to save time?

Additionnally, I will present a clean and minimal implementation of a classifier built on top of BERT. It will be based on the [official Google repo](https://github.com/google-research/bert), however, I will remove all the stuff not needed and clarifiy the implementation.

At the moment of this writing, TPUs are not available yet for TensorFlow 2.0, therefore, version 1.14.0 will be used.

Both implementations (GPU and TPU) are available [here](https://github.com/ostamand/bert-classifier)

## Data

To get the data (csv format), navigate to the [Kaggle competition site](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). The `comment_text` and `target` columns contain respectively comments collected from the web and the associated label indicating if it was judged toxic.

## Model

The first step is to create the `model_fn` function. The actual graph is defined in the `build_model` function. It uses [TensorFlow Hub](https://www.tensorflow.org/hub) to get a pretrained BERT model and define a fully connected layer on top of the BERT pooling layer.

```python
def model_fn(features, labels, mode, params):
  input_ids = features["input_ids"]
  input_mask = features["input_mask"]
  segment_ids = features["segment_ids"]
  label_ids = features["label_ids"]

  loss, train_op, eval_op, accuracy = build_model(
      params['config'],
      input_ids,
      input_mask,
      segment_ids,
      label_ids
  )

  if mode == tf.estimator.ModeKeys.TRAIN:
    spec = tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op
    )
  elif mode == tf.estimator.ModeKeys.EVAL:
    spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_op
    )

  return spec
```

An `EstimatorSpec` is returned based on the current mode (eval, train).

## Input

To manage inputs we use the `input_fn` function. It returns a `dataset` based on the running mode.

```python
def input_fn(features, seq_length, batch_size, mode):
  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  num_examples = len(features)

  dataset = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  elif mode == tf.estimator.ModeKeys.EVAL:
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

  return dataset
```

## Estimator

Finally, we can create the `Estimator` by passing in our `model_fn`. Some running configuration are also provided.

```python
run_config = tf.estimator.RunConfig(
    log_step_count_steps=10,
    save_summary_steps=10
)

classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={
        'config': config},
    model_dir='tmp'
)
```

## Train

To start training, we simply call the `train` providing the `input_fn`.

```python
%%time
classifier.train(
    input_fn=lambda: input_fn(features, config.maxlen, config.bs, tf.estimator.ModeKeys.TRAIN),
    max_steps=config.num_train_steps
)
```

## Results

Below is a comparison of the run time for a GPU and TPU on Google Colab. The difference is huge with minimal changes required to the code between.

| Env.      | Wall time |
| --------- | --------- |
| GPU Colab | 6036 sec  |
| TPU Colab | 850 sec   |
