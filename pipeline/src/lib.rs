//! `svgit-pipeline` — owned raster-to-vector pipeline stages (Level 2).
//!
//! This crate is where svgit gradually takes ownership of the tracing pipeline
//! described in the project plan: preprocess → color quantization → segmentation
//! → boundary extraction → simplification → curve fitting → layering →
//! serialization.
//!
//! Today it implements the first and highest-leverage stage — **color
//! quantization** (k-means in CIELAB space). It runs as a pre-processing pass
//! before the VTracer core; later stages will replace more of that core.

pub mod color;
pub mod quantize;

pub use quantize::{quantize_rgba, QuantizeConfig};
