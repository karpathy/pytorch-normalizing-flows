# pytorch-normalizing-flows

Implementions of normalizing flows (NICE, RealNVP, MAF, IAF, Neural Splines Flows, etc) in PyTorch.

![Normalizing Flow fitting a 2D dataset](https://github.com/karpathy/pytorch-normalizing-flows/blob/master/assets/moon_flow.png)

**todos**
- TODO: make work on GPU
- TODO: 2D -> ND: get (flat) using MNIST
- TODO: ND -> images (multi-scale architectures, Glow nets, etc) on MNIST/CIFAR/ImageNet
- TODO: more stable residual-like IAF-style updates (tried but didn't work too well)
- TODO: parallel wavenet
- TODO: radial/planar 2D flows from Rezende Mohamed 2015?
