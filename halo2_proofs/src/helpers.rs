use crate::poly::Polynomial;
use ff::PrimeField;
use halo2curves::{pairing::Engine, serde::SerdeObject, CurveAffine};
use itertools::Itertools;
use maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter},
    path::Path,
};

/// This enum specifies how various types are serialized and deserialized.
#[derive(Clone, Copy, Debug)]
pub enum SerdeFormat {
    /// Curve elements are serialized in compressed form.
    /// Field elements are serialized in standard form, with endianness specified by the
    /// `PrimeField` implementation.
    Processed,
    /// Curve elements are serialized in uncompressed form. Field elements are serialized
    /// in their internal Montgomery representation.
    /// When deserializing, checks are performed to ensure curve elements indeed lie on the curve and field elements
    /// are less than modulus.
    RawBytes,
    /// Serialization is the same as `RawBytes`, but no checks are performed.
    RawBytesUnchecked,
}

// Keep this trait for compatibility with IPA serialization
pub(crate) trait CurveRead: CurveAffine {
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut compressed = Self::Repr::default();
        reader.read_exact(compressed.as_mut())?;
        Option::from(Self::from_bytes(&compressed))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Invalid point encoding in proof"))
    }
}
impl<C: CurveAffine> CurveRead for C {}

pub trait SerdeCurveAffine: CurveAffine + SerdeObject {
    /// Reads an element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompress it
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    /// Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    /// does not perform any checks
    fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> Self {
        match format {
            SerdeFormat::Processed => <Self as CurveRead>::read(reader).unwrap(),
            SerdeFormat::RawBytes => <Self as SerdeObject>::read_raw(reader).unwrap(),
            SerdeFormat::RawBytesUnchecked => <Self as SerdeObject>::read_raw_unchecked(reader),
        }
    }
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) {
        match format {
            SerdeFormat::Processed => writer.write_all(self.to_bytes().as_ref()).unwrap(),
            _ => self.write_raw(writer).unwrap(),
        }
    }
}
impl<C: CurveAffine + SerdeObject> SerdeCurveAffine for C {}

pub trait SerdePrimeField: PrimeField + SerdeObject {
    /// Reads a field element as bytes from the buffer according to the `format`:
    /// - `Processed`: Reads a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads a field element from raw bytes in its internal Montgomery representations,
    /// and checks that the element is less than the modulus.
    /// - `RawBytesUnchecked`: Reads a field element in Montgomery form and performs no checks.
    fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> Self {
        match format {
            SerdeFormat::Processed => {
                let mut compressed = Self::Repr::default();
                reader.read_exact(compressed.as_mut()).unwrap();
                Option::from(Self::from_repr(compressed))
                    .unwrap_or_else(|| panic!("Invalid prime field point encoding"))
            }
            SerdeFormat::RawBytes => <Self as SerdeObject>::read_raw(reader).unwrap(),
            SerdeFormat::RawBytesUnchecked => <Self as SerdeObject>::read_raw_unchecked(reader),
        }
    }

    /// Writes a field element as bytes to the buffer according to the `format`:
    /// - `Processed`: Writes a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation.
    /// - Otherwise: Writes a field element into raw bytes in its internal Montgomery representation,
    /// WITHOUT performing the expensive Montgomery reduction.
    fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) {
        match format {
            SerdeFormat::Processed => writer.write_all(self.to_repr().as_ref()).unwrap(),
            _ => self.write_raw(writer).unwrap(),
        }
    }
}
impl<F: PrimeField + SerdeObject> SerdePrimeField for F {}

/// Convert a slice of `bool` into a `u8`.
///
/// Panics if the slice has length greater than 8.
pub fn pack(bits: &[bool]) -> u8 {
    let mut value = 0u8;
    assert!(bits.len() <= 8);
    for (bit_index, bit) in bits.iter().enumerate() {
        value |= (*bit as u8) << bit_index;
    }
    value
}

/// Writes the first `bits.len()` bits of a `u8` into `bits`.
pub fn unpack(byte: u8, bits: &mut [bool]) {
    for (bit_index, bit) in bits.iter_mut().enumerate() {
        *bit = (byte >> bit_index) & 1 == 1;
    }
}

/// Reads a vector of polynomials from buffer
pub(crate) fn read_polynomial_vec<R: io::Read, F: SerdePrimeField, B>(
    reader: &mut R,
    format: SerdeFormat,
) -> Vec<Polynomial<F, B>> {
    let mut len = [0u8; 4];
    reader.read_exact(&mut len).unwrap();
    let len = u32::from_be_bytes(len);

    (0..len)
        .map(|_| Polynomial::<F, B>::read(reader, format))
        .collect()
}

/// Reads a vector of polynomials from buffer
pub(crate) async fn multi_thread_read_polynomial_vec<
    F: SerdePrimeField,
    B: Send + Sync + 'static,
>(
    pk_prefix_path: impl AsRef<Path>,
    format: SerdeFormat,
    n: usize,
) -> Vec<Polynomial<F, B>> {
    const BUFFER_SIZE: usize = 1024 * 1024;
    let join_handles = (0..n)
        .map(|i| {
            let mut poly_path = pk_prefix_path
                .as_ref()
                .clone()
                .to_path_buf()
                .into_os_string();
            poly_path.push(format!("_{i}"));
            let mut reader = BufReader::with_capacity(BUFFER_SIZE, File::open(poly_path).unwrap());
            tokio::spawn(async move { Polynomial::<F, B>::read(&mut reader, format) })
        })
        .collect_vec();
    let mut ret = Vec::with_capacity(join_handles.len());
    for join_handle in join_handles {
        ret.push(join_handle.await.unwrap());
    }
    ret
}

/// Writes a slice of polynomials to buffer
pub(crate) fn write_polynomial_slice<W: io::Write, F: SerdePrimeField, B>(
    slice: &[Polynomial<F, B>],
    writer: &mut W,
    format: SerdeFormat,
) {
    writer
        .write_all(&(slice.len() as u32).to_be_bytes())
        .unwrap();
    for poly in slice.iter() {
        poly.write(writer, format);
    }
}

/// Writes a slice of polynomials to buffer
pub(crate) fn multi_thread_write_polynomial_slice<F: SerdePrimeField, B: Send + Sync>(
    slice: &[Polynomial<F, B>],
    pk_prefix_path: impl AsRef<Path>,
    format: SerdeFormat,
) {
    const BUFFER_SIZE: usize = 1024 * 1024;
    let poly_path = slice
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut poly_path = pk_prefix_path
                .as_ref()
                .clone()
                .to_path_buf()
                .into_os_string();
            poly_path.push(format!("_{i}"));
            poly_path
        })
        .collect_vec();
    slice
        .par_iter()
        .zip_eq(poly_path.par_iter())
        .for_each(|(poly, poly_path)| {
            let mut writer =
                BufWriter::with_capacity(BUFFER_SIZE, File::create(poly_path).unwrap());
            poly.write(&mut writer, format);
        });
}

/// Gets the total number of bytes of a slice of polynomials, assuming all polynomials are the same length
pub(crate) fn polynomial_slice_byte_length<F: PrimeField, B>(slice: &[Polynomial<F, B>]) -> usize {
    let field_len = F::default().to_repr().as_ref().len();
    4 + slice.len() * (4 + field_len * slice.get(0).map(|poly| poly.len()).unwrap_or(0))
}
