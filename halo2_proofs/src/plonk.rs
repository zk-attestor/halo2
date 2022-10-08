//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use blake2b_simd::Params as Blake2bParams;
use group::ff::Field;
use serde::{Deserialize, Serialize};

use crate::arithmetic::{CurveAffine, FieldExt};
use crate::helpers::CurveRead;
use crate::poly::{
    commitment::Params, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
    PinnedEvaluationDomain, Polynomial,
};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript};

mod assigned;
mod circuit;
mod error;
mod evaluation;
mod keygen;
mod lookup;
pub(crate) mod permutation;
mod vanishing;

mod prover;
mod verifier;

pub use assigned::*;
pub use circuit::*;
pub use error::*;
pub use keygen::*;
pub use prover::*;
pub use verifier::*;

use evaluation::Evaluator;
use std::io;

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<C: CurveAffine> {
    domain: EvaluationDomain<C::Scalar>,
    fixed_commitments: Vec<C>,
    permutation: permutation::VerifyingKey<C>,
    cs: ConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: C::Scalar,
    selectors: Vec<Vec<bool>>,
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Writes a verifying key to a buffer.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write(&(self.fixed_commitments.len() as u32).to_be_bytes());
        for commitment in &self.fixed_commitments {
            writer.write_all(commitment.to_bytes().as_ref())?;
        }
        self.permutation.write(writer)?;

        // write self.selectors
        for selector in &self.selectors {
            let mut selector_bytes = vec![0u8; selector.len() / 8 + 1];
            for i in 0..selector.len() {
                let byte_index = i / 8;
                let bit_index = i % 8;
                selector_bytes[byte_index] |= (selector[i] as u8) << bit_index;
            }
            writer.write_all(&selector_bytes)?;
        }

        Ok(())
    }

    /// Reads a verification key from a buffer.
    pub fn read<'params, R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        params: &impl Params<'params, C>,
    ) -> io::Result<Self> {
        let (domain, cs, _) = keygen::create_domain::<C, ConcreteCircuit>(params.k());
        let mut num_fixed_columns_be_bytes = [0u8; 4];
        reader.read(&mut num_fixed_columns_be_bytes)?;
        let num_fixed_columns = u32::from_be_bytes(num_fixed_columns_be_bytes);

        let fixed_commitments: Vec<_> = (0..num_fixed_columns)
            .map(|_| C::read(reader))
            .collect::<Result<_, _>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation)?;

        // read selectors
        let selectors: Vec<Vec<bool>> = vec![vec![false; params.n() as usize]; cs.num_selectors]
            .into_iter()
            .map(|mut selector| {
                let mut selector_bytes = vec![0u8; selector.len() / 8 + 1];
                reader
                    .read_exact(&mut selector_bytes)
                    .expect("unable to read selector bytes");
                for i in 0..selector.len() {
                    let byte_index = i / 8;
                    let bit_index = i % 8;
                    selector[i] = (selector_bytes[byte_index] >> bit_index) & 1 == 1;
                }
                Ok(selector)
            })
            .collect::<Result<Vec<Vec<bool>>, &str>>()
            .unwrap();
        let (cs, _) = cs.compress_selectors(selectors.clone());

        Ok(Self::from_parts(
            domain,
            fixed_commitments,
            permutation,
            cs,
            selectors,
        ))
    }

    fn from_parts(
        domain: EvaluationDomain<C::Scalar>,
        fixed_commitments: Vec<C>,
        permutation: permutation::VerifyingKey<C>,
        cs: ConstraintSystem<C::Scalar>,
        selectors: Vec<Vec<bool>>,
    ) -> Self {
        // Compute cached values.
        let cs_degree = cs.degree();

        let mut vk = Self {
            domain,
            fixed_commitments,
            permutation,
            cs,
            cs_degree,
            // Temporary, this is not pinned.
            transcript_repr: C::Scalar::zero(),
            selectors,
        };

        let mut hasher = Blake2bParams::new()
            .hash_length(64)
            .personal(b"Halo2-Verify-Key")
            .to_state();

        let s = format!("{:?}", vk.pinned());

        hasher.update(&(s.len() as u64).to_le_bytes());
        hasher.update(s.as_bytes());

        // Hash in final Blake2bState
        vk.transcript_repr = C::Scalar::from_bytes_wide(hasher.finalize().as_array());

        vk
    }

    /// Hashes a verification key into a transcript.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, C> {
        PinnedVerificationKey {
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }

    /// Returns commitments of fixed polynomials
    pub fn fixed_commitments(&self) -> &Vec<C> {
        &self.fixed_commitments
    }

    /// Returns `VerifyingKey` of permutation
    pub fn permutation(&self) -> &permutation::VerifyingKey<C> {
        &self.permutation
    }

    /// Returns `ConstraintSystem`
    pub fn cs(&self) -> &ConstraintSystem<C::Scalar> {
        &self.cs
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, C: CurveAffine> {
    base_modulus: &'static str,
    scalar_modulus: &'static str,
    domain: PinnedEvaluationDomain<'a, C::Scalar>,
    cs: PinnedConstraintSystem<'a, C::Scalar>,
    fixed_commitments: &'a Vec<C>,
    permutation: &'a permutation::VerifyingKey<C>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<C: CurveAffine> {
    vk: VerifyingKey<C>,
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_last: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_active_row: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    fixed_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    permutation: permutation::ProvingKey<C>,
    ev: Evaluator<C>,
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<C> {
        &self.vk
    }
    /// Writes a proving key to a buffer.
    /// Does so by first writing the verifying key and then serializing the rest of the data (in the form of field polynomials)
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.vk.write(writer)?;
        let partial_pkey = ProvingKeyWithoutVerifyingKey {
            l0: self.l0.clone(),
            l_last: self.l_last.clone(),
            l_active_row: self.l_active_row.clone(),
            fixed_values: self.fixed_values.clone(),
            fixed_polys: self.fixed_polys.clone(),
            fixed_cosets: self.fixed_cosets.clone(),
            permutation: self.permutation.clone(),
            ev: self.ev.clone(),
        };
        let pkey_serialized =
            bincode::serialize(&partial_pkey).expect("should be able to serialize pkey");
        writer.write_all(&pkey_serialized)
    }
    /// Reads a proving key from a buffer.
    /// Does so by reading verification key first, and then deserializing the rest of the file into the remaining proving key data.
    pub fn read<'params, R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        params: &impl Params<'params, C>,
    ) -> io::Result<Self> {
        let vk = VerifyingKey::<C>::read::<R, ConcreteCircuit>(reader, params)?;
        let mut buf = vec![];
        reader.read_to_end(&mut buf)?;
        let partial_pk: ProvingKeyWithoutVerifyingKey<C> =
            bincode::deserialize(&buf).expect("should be able to deserialize pkey");
        Ok(Self {
            vk,
            l0: partial_pk.l0,
            l_last: partial_pk.l_last,
            l_active_row: partial_pk.l_active_row,
            fixed_values: partial_pk.fixed_values,
            fixed_polys: partial_pk.fixed_polys,
            fixed_cosets: partial_pk.fixed_cosets,
            permutation: partial_pk.permutation,
            ev: partial_pk.ev,
        })
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<C::Scalar> {
        &self.domain
    }
}

#[allow(missing_docs)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: CurveAffine")]
struct ProvingKeyWithoutVerifyingKey<C: CurveAffine> {
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_last: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_active_row: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    fixed_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    permutation: permutation::ProvingKey<C>,
    ev: Evaluator<C>,
}

#[derive(Clone, Copy, Debug)]
struct Theta;
type ChallengeTheta<F> = ChallengeScalar<F, Theta>;

#[derive(Clone, Copy, Debug)]
struct Beta;
type ChallengeBeta<F> = ChallengeScalar<F, Beta>;

#[derive(Clone, Copy, Debug)]
struct Gamma;
type ChallengeGamma<F> = ChallengeScalar<F, Gamma>;

#[derive(Clone, Copy, Debug)]
struct Y;
type ChallengeY<F> = ChallengeScalar<F, Y>;

#[derive(Clone, Copy, Debug)]
struct X;
type ChallengeX<F> = ChallengeScalar<F, X>;
