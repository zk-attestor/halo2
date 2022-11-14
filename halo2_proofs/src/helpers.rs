use std::io;

use ff::PrimeField;
use halo2curves::CurveAffine;
use serde::{de::Error, Deserialize, Serialize};

pub(crate) trait CurveRead: CurveAffine {
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut compressed = Self::Repr::default();
        reader.read_exact(compressed.as_mut())?;
        Option::from(Self::from_bytes(&compressed))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "invalid point encoding in proof"))
    }
}

impl<C: CurveAffine> CurveRead for C {}

#[derive(Clone, Debug)]
pub struct SerdePrimeField<F: PrimeField>(F);

impl<F: PrimeField> serde_with::SerializeAs<F> for SerdePrimeField<F> {
    fn serialize_as<S>(source: &F, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Serialize::serialize(
            serde_bytes::Bytes::new(source.to_repr().as_ref()),
            serializer,
        )
    }
}

impl<'de, F: PrimeField> serde_with::DeserializeAs<'de, F> for SerdePrimeField<F> {
    fn deserialize_as<D>(deserializer: D) -> Result<F, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut compressed = F::Repr::default();
        let bytes: serde_bytes::ByteBuf = Deserialize::deserialize(deserializer)?;
        compressed.as_mut().copy_from_slice(&bytes);
        Option::from(F::from_repr(compressed))
            .ok_or_else(|| D::Error::custom("invalid prime field point encoding"))
    }
}
