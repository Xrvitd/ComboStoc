## ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models<br><sub>Official Code</sub>



![ComboStoc samples](visuals/teaser.png)

This repo contains the image diffusion models and training/sampling code for our paper exploring 
the Combinatorial Stochasticity for Diffusion Generative Models. 

#### We will add the image and structured shape generation code soon.

#### Pls cite our paper:
```
@article{xu2024combostoc,
      title={ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models}, 
      author={Rui Xu and Jiepeng Wang and Hao Pan and Yang Liu and Xin Tong and Shiqing Xin and Changhe Tu and Taku Komura and Wenping Wang},
      year={2024},
      eprint={2405.13729},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



In this paper we study an under-explored but important factor of diffusion generative models, i.e., the combinatorial complexity. 
Data samples are generally high-dimensional, and for various structured generation tasks there are additional attributes which are combined to associate with data samples.
We show that the space spanned by the combination of dimensions and attributes is insufficiently sampled by existing training scheme of diffusion generative models, causing degraded test time performance.
We present a simple fix to this problem by constructing stochastic processes that fully exploit the combinatorial structures, hence the name ComboStoc.
Using this simple strategy, we show that network training is significantly accelerated across diverse data modalities, including images and 3D structured shapes.
Moreover, ComboStoc enables a new way of test time generation which uses insynchronized time steps for different dimensions and attributes, thus allowing for varying degrees of control over them.


![More ComboStoc samples](visuals/yingwu.drawio.png)






