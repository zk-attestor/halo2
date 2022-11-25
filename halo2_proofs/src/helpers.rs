use std::io;

use ff::PrimeField;
use halo2curves::CurveAffine;

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

pub(crate) trait SerdePrimeField: PrimeField {
    /// Reads a field element as bytes from the buffer using `from_repr`.
    /// Endianness is specified by `PrimeField` implementation.
    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut compressed = Self::Repr::default();
        reader.read_exact(compressed.as_mut())?;
        Option::from(Self::from_repr(compressed)).ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "Invalid prime field point encoding")
        })
    }

    /// Writes a field element as bytes to the buffer using `to_repr`.
    /// Endianness is specified by `PrimeField` implementation.
    fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.to_repr().as_ref())
    }
}

impl<F: PrimeField> SerdePrimeField for F {}
