# Invertible Attention + iResNet 

This is the implementation code repo for the second experiment in the paper [Invertible Attention](https://arxiv.org/abs/2106.09003). In this experiment, we embed invertible attention module into [i-ResNet](https://arxiv.org/abs/1811.00995). 

This model is trained to learn the distribution of CIFAR10, and generate samples following the learned distribution. 

## Embed Location

The invertible attention is embedded into "scale_block" defined in the models/conv_iResNet.py, line 155-166.
The invertible attention module itself is defined in models/inv_attention.py


## Other settings
Apart from the embedded attention, all the other settings such as optimizer, model framework, learning rate etc. are kept the same as the official implementation of [i-ResNet](https://github.com/jhjacobsen/invertible-resnet). One extra chnage is that, as we need to embed invertible attention and it requires more video memory, the trainig batch size has to be shrunk from 128 to 64 to fit the model into a single RTX 3090.

## Execution Shell Command

```shell
# i-ResNet + invertible concatenation style attention
python CIFAR_main.py --doAttention True False False --AttentionType concat
--save_dir results/iResNet_concat_gen/ --gen True

# i-ResNet + invertible dot-product style attention
python CIFAR_main.py --doAttention True False False --AttentionType dot
--save_dir results/iResNet_dot_gen/ --gen True

# vanilla i-ResNet
python CIFAR_main.py --doAttention False False False --AttentionType NoAttention
--save_dir results/iResNet_gen/ --gen True

```

### Expected Training Time
It takes about 3.5 days to finish training. 


### Acknowledgement
We need to thank the authors of [Invertible Residual Networks](https://arxiv.org/abs/1811.00995) for kindly providing their training code in full detail and sharing it with MIT License.


### Bibliography
```
@misc{zha2021invertible,
      title={Invertible Attention}, 
      author={Jiajun Zha and Yiran Zhong and Jing Zhang and Liang Zheng and Richard Hartley},
      year={2021},
      eprint={2106.09003},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```