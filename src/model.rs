use burn::nn::{Linear, LinearConfig, conv::*, pool::*};
use burn::prelude::*;
use burn::tensor::activation::{leaky_relu, sigmoid};

#[derive(Module, Debug)]
pub struct CoTipNet<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    pool: AvgPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
}

#[allow(dead_code)]
#[derive(Config, Debug)]
pub struct CoTipNetConfig {}

impl<B: Backend> CoTipNet<B> {
    pub fn init(device: &B::Device) -> CoTipNet<B> {
        CoTipNet {
            conv1: Conv2dConfig::new([1, 4], [3, 3]).init(device),
            conv2: Conv2dConfig::new([4, 4], [3, 3]).init(device),
            conv3: Conv2dConfig::new([4, 8], [3, 3]).init(device),
            conv4: Conv2dConfig::new([8, 8], [3, 3]).init(device),
            pool: AvgPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            linear1: LinearConfig::new(32, 32).init(device),
            linear2: LinearConfig::new(32, 1).init(device),
        }
    }
}

impl<B: Backend> CoTipNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = leaky_relu(self.conv1.forward(x), 0.1);
        let x = leaky_relu(self.conv2.forward(x), 0.1);
        let x = self.pool.forward(x);
        let x = leaky_relu(self.conv3.forward(x), 0.1);
        let x = leaky_relu(self.conv4.forward(x), 0.1);
        let [batch, _, _, _] = x.dims();
        let x = x.reshape([batch, 32]);
        let x = leaky_relu(self.linear1.forward(x), 0.1);
        sigmoid(self.linear2.forward(x))
    }
}
