use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

use crate::dataset::TipImageItem;
use crate::preprocess::load_normal_image;

/// A batch of tip images ready for the model.
#[derive(Debug, Clone)]
pub struct TipBatch<B: Backend> {
    /// Shape: (batch, 1, 16, 16)
    pub images: Tensor<B, 4>,
    /// Shape: (batch, 1) — binary labels
    pub targets: Tensor<B, 2>,
}

/// Converts raw `TipImageItem`s into batched tensors.
/// When `augment` is true, each image produces 8 variants
/// (4 rotations × 2 flips).
pub struct TipBatcher {
    pub augment: bool,
}

impl TipBatcher {
    pub fn train() -> Self {
        Self { augment: true }
    }

    pub fn valid() -> Self {
        Self { augment: false }
    }
}

impl<B: Backend> Batcher<B, TipImageItem, TipBatch<B>> for TipBatcher {
    fn batch(&self, items: Vec<TipImageItem>, device: &B::Device) -> TipBatch<B> {
        let mut images = Vec::new();
        let mut targets = Vec::new();

        for item in &items {
            let pixels = load_normal_image(&item.path);
            // Shape: (1, 16, 16) — single channel
            let img = Tensor::<B, 1>::from_floats(pixels.as_slice(), device).reshape([1, 16, 16]);

            if self.augment {
                let augmented = augment_img(img);
                targets.extend(std::iter::repeat_n(item.label, augmented.len()));
                images.extend(augmented);
            } else {
                images.push(img);
                targets.push(item.label);
            }
        }

        let images = Tensor::stack(images, 0);
        let targets =
            Tensor::<B, 1>::from_floats(targets.as_slice(), device).reshape([targets.len(), 1]); // (batch, 1)

        TipBatch { images, targets }
    }
}

fn augment_img<B: Backend>(img: Tensor<B, 3>) -> Vec<Tensor<B, 3>> {
    let original = img.clone();
    let rot_90 = img.clone().swap_dims(1, 2).flip([2]);
    let rot_180 = img.clone().flip([1, 2]);
    let rot_270 = img.clone().swap_dims(1, 2).flip([1]);
    let flip_original = original.clone().flip([1]);
    let flip_rot_90 = rot_90.clone().flip([1]);
    let flip_rot_180 = rot_180.clone().flip([1]);
    let flip_rot_270 = rot_270.clone().flip([1]);

    vec![
        original,
        rot_90,
        rot_180,
        rot_270,
        flip_original,
        flip_rot_90,
        flip_rot_180,
        flip_rot_270,
    ]
}
