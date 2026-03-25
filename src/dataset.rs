use std::fs;
use std::path::{Path, PathBuf};

use burn::data::dataset::Dataset;

/// A single labeled image: path + binary label (1.0 = good, 0.0 = bad).
#[derive(Debug, Clone)]
pub struct TipImageItem {
    pub path: PathBuf,
    pub label: f32,
}

/// Dataset of labeled tip images loaded from a directory structure:
///   base_dir/{train,valid,test}/{goods,bads}/*.png
pub struct TipImageDataset {
    items: Vec<TipImageItem>,
}

impl TipImageDataset {
    pub fn from_dir(split_dir: &Path) -> Self {
        let mut items = collect_pngs(&split_dir.join("goods"), 1.0);
        items.extend(collect_pngs(&split_dir.join("bads"), 0.0));
        TipImageDataset { items }
    }
}

fn collect_pngs(dir: &Path, label: f32) -> Vec<TipImageItem> {
    fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? == "png" {
                Some(TipImageItem { path, label })
            } else {
                None
            }
        })
        .collect()
}

impl Dataset<TipImageItem> for TipImageDataset {
    fn get(&self, index: usize) -> Option<TipImageItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
