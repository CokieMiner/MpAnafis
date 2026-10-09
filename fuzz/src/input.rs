//! Independent category, operation, sign, scalar, and operand controls.

pub struct Input<'data> {
    pub category: u8,
    pub operation: u8,
    pub flags: u8,
    pub parameter: u16,
    pub left: &'data [u8],
    pub right: &'data [u8],
    pub modulus: &'data [u8],
}

impl<'data> Input<'data> {
    /// Decodes seven control bytes followed by three variable-length magnitudes.
    /// Empty magnitudes represent zero; all remaining bytes belong to an operand.
    pub fn parse(data: &'data [u8]) -> Option<Self> {
        let header = data.get(..7)?;
        let payload = &data[7..];
        let left_len = payload.len().checked_mul(usize::from(header[5]))? / 255;
        let (left, rest) = payload.split_at(left_len);
        let right_len = rest.len().checked_mul(usize::from(header[6]))? / 255;
        let (right, modulus) = rest.split_at(right_len);
        Some(Self {
            category: header[0] % 11,
            operation: header[1],
            flags: header[2],
            parameter: u16::from_le_bytes([header[3], header[4]]),
            left,
            right,
            modulus,
        })
    }
}
