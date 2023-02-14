use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use rand_core::RngCore;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::iter::{self, ExactSizeIterator};

use super::super::{circuit::Any, ChallengeBeta, ChallengeGamma, ChallengeX};
use super::{Argument, ProvingKey};
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine, FieldExt},
    plonk::{self, Error},
    poly::{
        self,
        commitment::{Blind, Params},
        Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

pub(crate) struct CommittedSet<C: CurveAffine> {
    pub(crate) permutation_product_poly: Polynomial<C::Scalar, Coeff>,
    pub(crate) permutation_product_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    // permutation_product_blind: Blind<C::Scalar>,
}

pub(crate) struct Committed<C: CurveAffine> {
    pub(crate) sets: Vec<CommittedSet<C>>,
}

pub struct ConstructedSet<C: CurveAffine> {
    permutation_product_poly: Polynomial<C::Scalar, Coeff>,
    // permutation_product_blind: Blind<C::Scalar>,
}

pub(crate) struct Constructed<C: CurveAffine> {
    sets: Vec<ConstructedSet<C>>,
}

pub(crate) struct Evaluated<C: CurveAffine> {
    constructed: Constructed<C>,
}

impl Argument {
    pub(in crate::plonk) fn commit<
        'params,
        C: CurveAffine,
        P: Params<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
        const ZK: bool,
    >(
        &self,
        params: &P,
        pk: &plonk::ProvingKey<C>,
        pkey: &ProvingKey<C>,
        advice: &[Polynomial<C::Scalar, LagrangeCoeff>],
        fixed: &[Polynomial<C::Scalar, LagrangeCoeff>],
        instance: &[Polynomial<C::Scalar, LagrangeCoeff>],
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        let domain = &pk.vk.domain;

        // How many columns can be included in a single permutation polynomial?
        // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
        // will never underflow because of the requirement of at least a degree
        // 3 circuit for the permutation argument.
        assert!(pk.vk.cs_degree >= 2 + ZK as usize);
        let chunk_len = if ZK {
            pk.vk.cs_degree - 2
        } else {
            pk.vk.cs_degree - 1
        };
        let num_chunks = pkey.permutations.chunks(chunk_len).count();
        let blinding_factors = pk.vk.cs.blinding_factors::<ZK>();

        // Each column gets its own delta power.
        let mut deltaomega = C::Scalar::one();

        // Track the "last" value from the previous column set
        let mut last_z = C::Scalar::one();

        let mut zs = Vec::with_capacity(num_chunks);
        let mut products = Vec::with_capacity(num_chunks);
        for (columns, permutations) in self
            .columns
            .chunks(chunk_len)
            .zip(pkey.permutations.chunks(chunk_len))
        {
            // Goal is to compute the products of fractions
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where p_j(X) is the jth column in this permutation,
            // and i is the ith row of the column.

            let mut modified_values = Vec::with_capacity(params.n() as usize);
            #[allow(clippy::uninit_vec)]
            unsafe {
                modified_values.set_len(params.n() as usize);
            }

            // Iterate over each column of the permutation
            for (col_idx, (&column, permuted_column_values)) in
                columns.iter().zip(permutations.iter()).enumerate()
            {
                let values = match column.column_type() {
                    Any::Advice(_) => advice,
                    Any::Fixed => fixed,
                    Any::Instance => instance,
                };
                parallelize(&mut modified_values, |modified_values, start| {
                    for ((modified_values, value), permuted_value) in modified_values
                        .iter_mut()
                        .zip(values[column.index()][start..].iter())
                        .zip(permuted_column_values[start..].iter())
                    {
                        if col_idx == 0 {
                            *modified_values = *beta * permuted_value + &*gamma + value;
                        } else {
                            *modified_values *= *beta * permuted_value + &*gamma + value;
                        }
                    }
                });
            }

            // Invert to obtain the denominator for the permutation product polynomial
            modified_values.batch_invert();

            // Iterate over each column again, this time finishing the computation
            // of the entire fraction by computing the numerators
            for &column in columns.iter() {
                let omega = domain.get_omega();
                let values = match column.column_type() {
                    Any::Advice(_) => advice,
                    Any::Fixed => fixed,
                    Any::Instance => instance,
                };
                parallelize(&mut modified_values, |modified_values, start| {
                    let mut deltaomega = deltaomega * &omega.pow_vartime([start as u64, 0, 0, 0]);
                    for (modified_values, value) in modified_values
                        .iter_mut()
                        .zip(values[column.index()][start..].iter())
                    {
                        // Multiply by p_j(\omega^i) + \delta^j \omega^i \beta
                        *modified_values *= &(deltaomega * &*beta + &*gamma + value);
                        deltaomega *= &omega;
                    }
                });
                deltaomega *= &C::Scalar::DELTA;
            }

            // The modified_values vector is a vector of products of fractions
            // of the form
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where i is the index into modified_values, for the jth column in
            // the permutation

            // Compute the evaluations of the permutation product polynomial
            // over our domain, starting with z[0] = 1
            if ZK {
                let mut z = Vec::with_capacity(params.n() as usize);
                z.push(last_z);
                for row in 1..(params.n() as usize) {
                    let tmp = modified_values[row - 1] * z.last().unwrap();
                    z.push(tmp);
                }
                let mut z = domain.lagrange_from_vec(z);

                // Set blinding factors
                for z in &mut z[params.n() as usize - blinding_factors..] {
                    *z = C::Scalar::random(&mut rng);
                }
                // Set new last_z
                last_z = z[params.n() as usize - (blinding_factors + 1)];

                zs.push(z);
            } else {
                products.push(modified_values);
            }
        }

        if !ZK && !products.is_empty() {
            #[cfg(feature = "profile")]
            let start = std::time::Instant::now();
            let num_columns = products.len();
            let last_column = products.len() - 1;
            let num_threads = rayon::current_num_threads();
            let n = params.n() as usize;
            let row_chunk_size = (n + num_threads - 1) / num_threads;
            let zs_chunks = crossbeam::scope(|s| {
                let products = &products;
                #[allow(clippy::needless_collect)]
                let handles = (0..n)
                    .step_by(row_chunk_size)
                    .map(|first_row| {
                        let last_row = n.min(first_row + row_chunk_size) - 1;
                        s.spawn(move |_| {
                            let mut zs = vec![Vec::with_capacity(row_chunk_size); num_columns];
                            zs[0].push(if first_row == 0 {
                                C::Scalar::one()
                            } else {
                                products[last_column][first_row - 1]
                            });
                            for idx in first_row..last_row {
                                // last().unwrap() is idx - first_row
                                for col in 0..last_column {
                                    let tmp = products[col][idx] * zs[col].last().unwrap();
                                    zs[col + 1].push(tmp);
                                }
                                // zs[0][idx + 1 - first_row]
                                let tmp =
                                    products[last_column][idx] * zs[last_column].last().unwrap();
                                zs[0].push(tmp);
                            }
                            for col in 0..last_column {
                                // zs[col + 1][last_row - first_row]
                                let tmp = products[col][last_row] * zs[col].last().unwrap();
                                zs[col + 1].push(tmp);
                            }
                            zs
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            })
            .unwrap();

            let zs_new = zs_chunks
                .into_par_iter()
                .reduce_with(|mut zs1, mut zs2| {
                    let multiplier = zs1.last().unwrap().last().unwrap();
                    zs2.par_iter_mut()
                        .flat_map(|z2| z2.par_iter_mut())
                        .for_each(|z2| {
                            *z2 *= multiplier;
                        });
                    zs1.par_iter_mut()
                        .zip(zs2.par_iter_mut())
                        .for_each(|(z1, z2)| {
                            z1.append(z2);
                        });
                    zs1
                })
                .unwrap();
            zs = zs_new
                .into_iter()
                .map(|z| domain.lagrange_from_vec(z))
                .collect();
            assert_eq!(
                zs[0][0],
                zs[last_column][n - 1] * products[last_column][n - 1]
            );
            #[cfg(feature = "profile")]
            println!(
                "Compute permutation polys for non-zk zig-zag (cpu): {:?}",
                start.elapsed()
            );
        }

        let sets: Vec<_> = zs
            .into_iter()
            .map(|z| {
                // let blind = Blind(C::Scalar::random(&mut rng));

                let permutation_product_commitment_projective = params.commit_lagrange(&z);
                // let permutation_product_blind = blind;
                let z = domain.lagrange_to_coeff(z);
                let permutation_product_poly = z.clone();

                let permutation_product_coset = domain.coeff_to_extended(&z);

                let permutation_product_commitment =
                    permutation_product_commitment_projective.to_affine();

                // Hash the permutation product commitment
                transcript
                    .write_point(permutation_product_commitment)
                    .unwrap();

                CommittedSet {
                    permutation_product_poly,
                    permutation_product_coset,
                    // permutation_product_blind,
                }
            })
            .collect();

        Ok(Committed { sets })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn construct(self) -> Constructed<C> {
        Constructed {
            sets: self
                .sets
                .iter()
                .map(|set| ConstructedSet {
                    permutation_product_poly: set.permutation_product_poly.clone(),
                    // permutation_product_blind: set.permutation_product_blind,
                })
                .collect(),
        }
    }
}

impl<C: CurveAffine> super::ProvingKey<C> {
    pub(in crate::plonk) fn open(
        &self,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'_, C>> + Clone {
        self.polys.iter().map(move |poly| ProverQuery {
            point: *x,
            poly,
            // blind: Blind::default(),
        })
    }

    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        &self,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<(), Error> {
        // Hash permutation evals
        for eval in self.polys.iter().map(|poly| eval_polynomial(poly, *x)) {
            transcript.write_scalar(eval)?;
        }

        Ok(())
    }
}

impl<C: CurveAffine> Constructed<C> {
    pub(in crate::plonk) fn evaluate<
        E: EncodedChallenge<C>,
        T: TranscriptWrite<C, E>,
        const ZK: bool,
    >(
        self,
        pk: &plonk::ProvingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &pk.vk.domain;
        let blinding_factors = pk.vk.cs.blinding_factors::<ZK>();

        {
            let mut sets = self.sets.iter();

            if ZK {
                while let Some(set) = sets.next() {
                    let permutation_product_eval =
                        eval_polynomial(&set.permutation_product_poly, *x);

                    let permutation_product_next_eval = eval_polynomial(
                        &set.permutation_product_poly,
                        domain.rotate_omega(*x, Rotation::next()),
                    );

                    // Hash permutation product evals
                    for eval in iter::empty()
                        .chain(Some(&permutation_product_eval))
                        .chain(Some(&permutation_product_next_eval))
                    {
                        transcript.write_scalar(*eval)?;
                    }

                    // If we have any remaining sets to process, evaluate this set at omega^u
                    // so we can constrain the last value of its running product to equal the
                    // first value of the next set's running product, chaining them together.
                    if sets.len() > 0 {
                        let permutation_product_last_eval = eval_polynomial(
                            &set.permutation_product_poly,
                            domain.rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32))),
                        );

                        transcript.write_scalar(permutation_product_last_eval)?;
                    }
                }
            } else {
                if let Some(set) = sets.next() {
                    let permutation_product_eval =
                        eval_polynomial(&set.permutation_product_poly, *x);

                    let permutation_product_next_eval = eval_polynomial(
                        &set.permutation_product_poly,
                        domain.rotate_omega(*x, Rotation::next()),
                    );

                    // Hash permutation product evals
                    for eval in iter::empty()
                        .chain(Some(&permutation_product_eval))
                        .chain(Some(&permutation_product_next_eval))
                    {
                        transcript.write_scalar(*eval)?;
                    }
                }
                for set in sets {
                    let permutation_product_eval =
                        eval_polynomial(&set.permutation_product_poly, *x);
                    transcript.write_scalar(permutation_product_eval)?;
                }
            }
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a, const ZK: bool>(
        &'a self,
        pk: &'a plonk::ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let blinding_factors = pk.vk.cs.blinding_factors::<ZK>();
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());
        let x_last = pk
            .vk
            .domain
            .rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32)));

        iter::empty()
            .chain(
                self.constructed
                    .sets
                    .iter()
                    .enumerate()
                    .flat_map(move |(idx, set)| {
                        iter::empty()
                            // Open permutation product commitments at x and \omega x
                            .chain(Some(ProverQuery {
                                point: *x,
                                poly: &set.permutation_product_poly,
                                // blind: set.permutation_product_blind,
                            }))
                            .chain((ZK || idx == 0).then_some(ProverQuery {
                                point: x_next,
                                poly: &set.permutation_product_poly,
                                // blind: set.permutation_product_blind,
                            }))
                    }),
            )
            // Open it at \omega^{last} x for all but the last set. This rotation is only
            // sensical for the first row, but we only use this rotation in a constraint
            // that is gated on l_0.
            .chain(if ZK {
                self.constructed
                    .sets
                    .iter()
                    .rev()
                    .skip(1)
                    .flat_map(move |set| {
                        Some(ProverQuery {
                            point: x_last,
                            poly: &set.permutation_product_poly,
                            // blind: set.permutation_product_blind,
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            })
    }
}
