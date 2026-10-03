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

/// Contract alias for [`ExrError`] (image-crate API contract).
pub type Error = ExrError;

/// Error variants returned by `oxideav-openexr`'s standalone API.
#[derive(Debug)]
#[non_exhaustive]
pub enum ExrError {
    /// Byte stream malformed (bad magic, truncated header, channel list
    /// missing, attribute payload runs past the declared size, line
    /// offset table inconsistent with dataWindow, etc.), or a
    /// caller-assembled image has inconsistent geometry.
    InvalidData(String),
    /// Byte stream uses a feature this crate doesn't implement, or a
    /// part has no colour view (deep data; a channel set without an
    /// RGB(A) / `Y` / `Y RY BY` mapping; a layout the encoder cannot
    /// carry).
    Unsupported(String),
    /// The header declares a picture larger than the caller-configured
    /// [`crate::DecodeOptions`] limits. Raised before any pixel plane is
    /// allocated. The string carries the dimension that tripped the
    /// limit and the limit value.
    LimitExceeded(String),
    /// A read ([`crate::decode_from`]) or write ([`crate::encode_to`])
    /// failed.
    Io(std::io::Error),
}

impl ExrError {
    /// Construct an [`ExrError::InvalidData`] from a stringy message.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }
    /// Construct an [`ExrError::Unsupported`] from a stringy message.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
    /// Construct an [`ExrError::LimitExceeded`] from a stringy message.
    pub fn limit(msg: impl Into<String>) -> Self {
        Self::LimitExceeded(msg.into())
    }
}

impl fmt::Display for ExrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::LimitExceeded(s) => write!(f, "limit exceeded: {s}"),
            Self::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for ExrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ExrError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
