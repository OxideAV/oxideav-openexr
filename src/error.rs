//! Crate-local error type used by `oxideav-openexr`'s standalone (no
//! `oxideav-core`) public API.
//!
//! When the `registry` feature is enabled, [`ExrError`] gains a
//! `From<ExrError> for oxideav_core::Error` impl (defined in
//! [`crate::registry`]) so the trait-side surface (`Decoder` /
//! `Encoder`) can keep returning `oxideav_core::Result<T>` while the
//! underlying parse/encode functions stay framework-free.

use core::fmt;

/// `Result` alias scoped to `oxideav-openexr`.
pub type Result<T> = core::result::Result<T, ExrError>;

/// Error variants returned by `oxideav-openexr`'s standalone API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExrError {
    /// Byte stream malformed (bad magic, truncated header, channel list
    /// missing, attribute payload runs past the declared size, line
    /// offset table inconsistent with dataWindow, etc.).
    InvalidData(String),
    /// Byte stream uses a feature this crate doesn't implement yet
    /// (PIZ / RLE / B44 / DWAA / DWAB compression; tiled format;
    /// multi-part files; deep data; UINT pixel type; subsampled
    /// channels with sampling != 1).
    Unsupported(String),
}

impl ExrError {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
}

impl fmt::Display for ExrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for ExrError {}
