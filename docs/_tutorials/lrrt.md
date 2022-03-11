---
title: "Learning Rate Range Test"
tags: training learning-rate
---
This tutorial shows how to use to perform Learning Rate range tests in PyTorch.

## Learning Rate Range Test (LRRT)

Learning rate range test ( [LRRT](https://arxiv.org/abs/1803.09820) ) is a
method for discovering the largest learning rate values that can be used to
train a model without divergence. Data scientists are often interested in this
information because  large learning rates lead to faster model convergence than
a small learning rates.  Moreover, large learning rates are crucial in learning
rate schedules such as [CLR](https://arxiv.org/abs/1506.01186)  and
[1Cycle](https://arxiv.org/abs/1803.09820), which are used to train effectively
with large batch sizes. DeepSpeed provides LRRT for model training in PyTorch
frameworks.

## Prerequisites

To use DeepSpeed's LRRT, you must satisfy the following two conditions:

1. Integrate DeepSpeed into your training script using the [Getting
Started](/getting-started/) guide.
2. Add the parameters to configure LRRT to the parameters of your model. The
LRRT parameters are defined below.

## LRRT Parameters

LRRT works by linearly increasing the learning rate by a predefined amount, at
predefined intervals. Thus, LRRT is a form of learning rate schedule because it
defines how and when the learning rate should change during model training.  To
configure LRRT, you will need to set these parameters:

1. `lr_range_test_min_lr` : The initial learning rate for training `(float)`
2. `lr_range_test_step_size`: The interval for scaling up learning rate,
defined in training steps `(integer)`
3. `lr_range_test_step_rate`: The scaling factor for increasing learning rate
`(float)`
4. `lr_range_test_staircase`: If true, learning rate is changed every
`lr_range_test_step_size` training steps, otherwise learning rate is changed at
every training step `(boolean)`

## Required Model Configuration Changes

We will illustrate the required model configuration changes an example LRRT
schedule that:

1. Starts training with an initial learning rate of 0.0001
2. Uses a scaling rate of 5
3. Uses a scaling interval of 200 training steps
4. Scales learning rate at every training step, i.e., does not use staircase

### PyTorch

For PyTorch models, LRRT is implemented as a [learning rate
scheduler](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html),
a feature that is available in PyTorch versions 1.0.1 and newer. Thus, you can
add a `"scheduler"` entry of type `"LRRangeTest"` into your model configuration
as illustrated below:

```json
"scheduler": {
    "type": "LRRangeTest",
    "params": {
        "lr_range_test_min_lr": 0.0001,
        "lr_range_test_step_size": 200,
        "lr_range_test_step_rate": 5,
        "lr_range_test_staircase": false
    }
}
```


## Example: Tuning for Large Batch Sizes

We illustrate how LRRT can benefit data scientists with a snippet of our
experience of tuning an internal production model to converge efficiently on
larger batch sizes, as we scaled from one GPU (batch size 512) to four GPUs
(batch size 2048). Our goal was to train the model with the larger batch size
to match the performance of the smaller batch size using the same amount of
data samples. The challenge here is the well known problem of slow convergence
of large batch size training. Our approach was to use a
[1Cycle](/tutorials/1Cycle/) schedule in DeepSpeed to tackle
this problem, and we used LRRT to configure the schedule.

In the plots below, we illustrate using LRRT to discover the maximum learning
rates for effective training with batch size 2048. The plot on the left shows
the impact of large learning rates on validation loss over the first 9000
batches of training. The plot on the right shows the learning rate values
during the same period of training.  Using grid search we discover that the
best fixed learning rate for the batch size 2048 is 0.0002. The blue line
(`lr=0.0002`) represents training with this fixed learning rate. We compare the
two LRRT schedules with this fixed learning rate. The orange
(`lr_range_test_step_rate=5`) and gray (`lr_range_test_step_rate=50`) lines
represent training with similar LRRT schedules that differ only in
`lr_range_test_step_rate` values. Although the LRRT schedules start from the
same base learning rate, the gray line's learning rate grows about 10 times
faster than the orange line. Also, the learning rates of the LRRT schedules had
grown larger than that of the blue line in the presented data points. We
subsequently refer to the gray line as "fast growing", and the orange line as
"slow growing" LRRT schedules respectively.

![validation_loss](/assets/images/loss_and_lr.png)

We make the following observations from this small example.

1. Larger learning rates clearly benefit model performance, up to some point.
The fast growing LRRT schedule achieves validation loss of 0.46 after 3000
batches, which the fixed learning rate does not achieve with 9000 batches. The
slow growing LRRT does not match that score until after 6000 batches, however
it maintains an increasing performance advantage over the fixed learning rate.

2. There is an upper bound on learning rate values that are useful for training
the model. The fast growing LRRT schedule hits this boundary quickly and
diverges, while the slow growing LRRT will later diverge for the same reason.
LRRT helped us discover these boundaries quickly,  using less than 2% of the
training data. These boundaries are useful information for constructing
learning rate schedules.

These observations from LRRT helped us to configure the learning rate
boundaries and the cycle span for a 1Cycle schedule that solves the problem, as
shown below.

```json
"OneCycle": {
    "cycle_min_lr": 0.002,
    "cycle_max_lr": 0.005,
    "cycle_first_step_size": 2000,
    "cycle_second_step_size": 2000,
    ...
}
```

In our experience these are four most critical parameters of 1Cycle schedules.

1. We chose to use the slower LRRT schedule (`lr_range_test_step_rate=5`) to
set `cycle_min_lr` because it achieves the best loss and the faster schedule
diverges fairly quickly.
2. We set `cycle_min_lr` to 0.005 even though the plot shows that performance
was still improving at slightly higher learning rate. This is because we
observed that if we wait till the maximum learning rate, the model could be at
the point of divergence and impossible to recover.
3. Since it takes 8000 batches for the learning rate to become 0.005, we set
`cycle_first_step_size` and (`cycle_second_step_size`) to 2000 which is the
number of steps that it takes for four GPUs to process 8000 batches.

We hope this brief example sparks your imagination on using LRRT for your own
unique tuning challenges.
