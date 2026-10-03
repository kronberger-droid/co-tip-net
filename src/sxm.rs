//! Reader for Nanonis `.sxm` scan files.
//!
//! Layout: an ASCII header of `:KEY:` blocks terminated by `:SCANIT_END:`,
//! then the bytes `0x1A 0x04`, then big-endian `f32` frames. Frames follow
//! the channel order of `DATA_INFO`; a channel with direction `both` stores
//! a forward frame followed by a backward frame. Each frame is
//! `SCAN_PIXELS` wide and high, stored in acquisition order.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum SxmError {
    Io(std::io::Error),
    Format(String),
}

impl fmt::Display for SxmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SxmError::Io(e) => write!(f, "{e}"),
            SxmError::Format(msg) => write!(f, "invalid sxm: {msg}"),
        }
    }
}

impl std::error::Error for SxmError {}

impl From<std::io::Error> for SxmError {
    fn from(e: std::io::Error) -> Self {
        SxmError::Io(e)
    }
}

fn format_err(msg: impl Into<String>) -> SxmError {
    SxmError::Format(msg.into())
}

/// One frame of a scan, oriented so row 0 is the top of the image.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Row-major values in the channel's unit; NaN where the scan was aborted.
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Physical size of the frame in nanometres (x, y).
    pub range_nm: (f32, f32),
}

/// Parsed `.sxm` file: scan geometry, frame layout and the raw frame data.
pub struct Sxm {
    width: usize,
    height: usize,
    range_nm: (f32, f32),
    scan_up: bool,
    /// (channel name, is forward) for every stored frame, in file order.
    frames: Vec<(String, bool)>,
    data: Vec<u8>,
}

impl Sxm {
    pub fn open(path: &Path) -> Result<Self, SxmError> {
        Self::parse(&std::fs::read(path)?)
    }

    pub fn parse(raw: &[u8]) -> Result<Self, SxmError> {
        let end = find(raw, b":SCANIT_END:").ok_or_else(|| format_err("no :SCANIT_END:"))?;
        let header_text: String = raw[..end].iter().map(|&b| b as char).collect();
        let header = parse_header(&header_text);

        let start = find(&raw[end..], b"\x1a\x04")
            .map(|i| end + i + 2)
            .ok_or_else(|| format_err("no data marker after header"))?;

        let pair = |key: &str| -> Result<(f64, f64), SxmError> {
            let v = header
                .get(key)
                .ok_or_else(|| format_err(format!("missing {key}")))?;
            let mut it = v.split_whitespace().map(|s| s.parse::<f64>());
            match (it.next(), it.next()) {
                (Some(Ok(a)), Some(Ok(b))) => Ok((a, b)),
                _ => Err(format_err(format!("bad {key}: {v:?}"))),
            }
        };
        let (nx, ny) = pair("SCAN_PIXELS")?;
        let (rx, ry) = pair("SCAN_RANGE")?;

        let info = header
            .get("DATA_INFO")
            .ok_or_else(|| format_err("missing DATA_INFO"))?;
        let mut frames = Vec::new();
        for line in info.lines().skip(1) {
            let cols: Vec<&str> = line.trim().split('\t').map(str::trim).collect();
            if cols.len() < 4 || cols[1].is_empty() {
                continue;
            }
            frames.push((cols[1].to_string(), true));
            if cols[3] == "both" {
                frames.push((cols[1].to_string(), false));
            }
        }

        Ok(Sxm {
            width: nx as usize,
            height: ny as usize,
            range_nm: ((rx * 1e9) as f32, (ry * 1e9) as f32),
            scan_up: header.get("SCAN_DIR").is_some_and(|d| d == "up"),
            frames,
            data: raw[start..].to_vec(),
        })
    }

    /// Names of the stored channels, in file order, without duplicates.
    pub fn channels(&self) -> Vec<&str> {
        let mut names: Vec<&str> = Vec::new();
        for (name, _) in &self.frames {
            if !names.contains(&name.as_str()) {
                names.push(name);
            }
        }
        names
    }

    /// The forward or backward frame of `channel`, top row first and with
    /// the backward frame mirrored to match the forward orientation.
    pub fn frame(&self, channel: &str, forward: bool) -> Result<Frame, SxmError> {
        let index = self
            .frames
            .iter()
            .position(|(name, fwd)| name == channel && *fwd == forward)
            .ok_or_else(|| {
                format_err(format!(
                    "no {} frame for channel {channel:?}, have {:?}",
                    if forward { "forward" } else { "backward" },
                    self.channels()
                ))
            })?;

        let n = self.width * self.height;
        let bytes = self
            .data
            .get(index * n * 4..(index + 1) * n * 4)
            .ok_or_else(|| format_err("file truncated"))?;
        let mut data: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_be_bytes(*b))
            .collect();

        if !forward {
            for row in data.chunks_exact_mut(self.width) {
                row.reverse();
            }
        }
        if self.scan_up {
            let flipped: Vec<f32> = data
                .chunks_exact(self.width)
                .rev()
                .flatten()
                .copied()
                .collect();
            data = flipped;
        }

        Ok(Frame {
            data,
            width: self.width,
            height: self.height,
            range_nm: self.range_nm,
        })
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Split the header into `KEY -> value` blocks. Values keep their inner
/// newlines (tables like `DATA_INFO` span several lines).
fn parse_header(text: &str) -> HashMap<String, String> {
    let mut header = HashMap::new();
    let mut key: Option<String> = None;
    let mut value = String::new();
    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.len() > 2 && trimmed.starts_with(':') && trimmed.ends_with(':') {
            if let Some(k) = key.take() {
                header.insert(k, value.trim().to_string());
            }
            key = Some(trimmed[1..trimmed.len() - 1].to_string());
            value.clear();
        } else {
            value.push_str(line);
            value.push('\n');
        }
    }
    if let Some(k) = key {
        header.insert(k, value.trim().to_string());
    }
    header
}

impl Frame {
    /// Keep the longest run of rows without NaN. Aborted scans leave the
    /// rows after the abort filled with NaN.
    pub fn complete_rows(self) -> Frame {
        let complete: Vec<bool> = self
            .data
            .chunks_exact(self.width)
            .map(|row| row.iter().all(|v| v.is_finite()))
            .collect();

        let (mut best, mut run_start) = ((0, 0), 0);
        for (i, &ok) in complete.iter().chain([&false]).enumerate() {
            if !ok {
                if i - run_start > best.1 - best.0 {
                    best = (run_start, i);
                }
                run_start = i + 1;
            }
        }

        let (r0, r1) = best;
        let height = r1 - r0;
        let range_y = self.range_nm.1 * height as f32 / self.height as f32;
        Frame {
            data: self.data[r0 * self.width..r1 * self.width].to_vec(),
            width: self.width,
            height,
            range_nm: (self.range_nm.0, range_y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal file with a `Z` channel stored both ways and a forward-only
    /// `Current` channel. Frame values encode (frame, row, col).
    fn synthetic(scan_dir: &str, nan_rows_from: usize) -> Vec<u8> {
        let (w, h) = (4usize, 3usize);
        let header = format!(
            ":NANONIS_VERSION:\n2\n:SCAN_PIXELS:\n       {w}       {h}\n\
             :SCAN_RANGE:\n           4.000000E-9           3.000000E-9\n\
             :SCAN_DIR:\n{scan_dir}\n:DATA_INFO:\n\
             \tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n\
             \t14\tZ\tm\tboth\t1.0E+0\t0.0E+0\n\
             \t0\tCurrent\tA\tforward\t1.0E+0\t0.0E+0\n\n\
             :SCANIT_END:\n\n\n"
        );
        let mut raw = header.into_bytes();
        raw.extend_from_slice(b"\x1a\x04");
        for frame in 0..3 {
            for r in 0..h {
                for c in 0..w {
                    let v = if r >= nan_rows_from {
                        f32::NAN
                    } else {
                        (frame * 100 + r * 10 + c) as f32
                    };
                    raw.extend_from_slice(&v.to_be_bytes());
                }
            }
        }
        raw
    }

    #[test]
    fn reads_header_and_channels() {
        let sxm = Sxm::parse(&synthetic("down", 99)).unwrap();
        assert_eq!(sxm.channels(), vec!["Z", "Current"]);
        let z = sxm.frame("Z", true).unwrap();
        assert_eq!((z.width, z.height), (4, 3));
        assert_eq!(z.range_nm, (4.0, 3.0));
        assert_eq!(z.data[..4], [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn backward_frame_is_mirrored_and_channels_are_offset() {
        let sxm = Sxm::parse(&synthetic("down", 99)).unwrap();
        let bwd = sxm.frame("Z", false).unwrap();
        assert_eq!(bwd.data[..4], [103.0, 102.0, 101.0, 100.0]);
        let current = sxm.frame("Current", true).unwrap();
        assert_eq!(current.data[0], 200.0);
        assert!(sxm.frame("Current", false).is_err());
    }

    #[test]
    fn scan_up_is_flipped_to_top_row_first() {
        let sxm = Sxm::parse(&synthetic("up", 99)).unwrap();
        let z = sxm.frame("Z", true).unwrap();
        assert_eq!(z.data[..4], [20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn complete_rows_drops_aborted_part() {
        let sxm = Sxm::parse(&synthetic("down", 2)).unwrap();
        let z = sxm.frame("Z", true).unwrap().complete_rows();
        assert_eq!(z.height, 2);
        assert_eq!(z.range_nm, (4.0, 2.0));
        assert!(z.data.iter().all(|v| v.is_finite()));
    }
}
