# SRP (ICLR'22)

<div align="center">
    <a><img src="figs/smile.png"  height="90px" ></a>
    &nbsp &nbsp
    <a><img src="figs/neu.png"  height="90px" ></a>
</div>

This repository is for a new network pruning method (`Aligned Structured Sparsity Learning, ASSL`) for efficient single image super-resolution (SR), introduced in our ICLR 2022 poster paper:
> **Learning Efficient Image Super-Resolution Networks via Structure-Regularized Pruning [[Camera Ready](https://openreview.net/pdf?id=AjGC97Aofee)]** \
> [Yulun Zhang*](http://yulunzhang.com/), [Huan Wang*](http://huanwang.tech/), [Can Qin](http://canqin.tech/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/) (*equal contribution) \
> Northeastern University, Boston, MA, USA


## Introduction
Several image super-resolution (SR) networks have been proposed of late for efficient SR, achieving promising results. However, they are still not lightweight enough and neglect to be extended to larger networks. At the same time, model compression techniques, like neural architecture search and knowledge distillation, typically consume considerable computation resources. In contrast, network pruning is a cheap and effective model compression technique. However, it is hard to be applied to SR networks directly because filter pruning for residual blocks is well-known tricky. To address the above issues, we propose structure-regularized pruning (SRP), which imposes regularization on the pruned structure to ensure the locations of pruned filters are aligned across different layers. Specifically, for the layers connected by the same residual, we select the filters of the same indices as unimportant filters. To transfer the expressive power in the unimportant filters to the rest of the network, we employ L2 regularization to drive the weights towards zero so that eventually, their absence will cause minimal performance degradation. We apply SRP to train efficient image SR networks, resulting in a lightweight network SRPN-Lite and a very deep one SRPN. We conduct extensive comparisons with both lightweight and larger networks. SRPN-Lite and SRPN perform favorably against other recent efficient SR approaches quantitatively and visually.

## TODO
Code and trained models will be released soon! Stay tuned!


## Acknowledgements
We refer to the following implementations when we develop this code: [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch), [RCAN](https://github.com/yulunzhang/RCAN), [Regularization-Pruning](https://github.com/MingSun-Tse/Regularization-Pruning). Great thanks to them!
