use ff::Field;
use group::Curve;
use halo2curves::CurveExt;
use rand_core::RngCore;
use std::collections::BTreeSet;
use std::env::var;
use std::marker::PhantomData;
use std::ops::RangeTo;
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;
use std::{collections::HashMap, iter, mem, sync::atomic::Ordering};

use super::{
    circuit::{
        sealed::{self, SealedPhase},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, FirstPhase, Fixed,
        FloorPlanner, Instance, Selector,
    },
    lookup, permutation, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
    ChallengeY, Error, Expression, ProvingKey,
};
use crate::poly::batch_invert_assigned_ref;
use crate::poly::commitment::ParamsProver;
use crate::transcript::Transcript;
use crate::{
    arithmetic::{eval_polynomial, CurveAffine, FieldExt},
    circuit::Value,
    plonk::Assigned,
    poly::{
        self,
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Basis, Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, ProverQuery,
    },
};
use crate::{
    poly::batch_invert_assigned,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use group::prime::PrimeCurveAffine;

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<
    'params,
    'a,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + 'a,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&[&'a [Scheme::Scalar]]],
    mut rng: R,
    mut transcript: &'a mut T,
) -> Result<(), Error> {
    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;
    let mut meta = ConstraintSystem::default();
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    struct InstanceSingle<C: CurveAffine> {
        pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        pub instance_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    }

    let instance: Vec<InstanceSingle<Scheme::Curve>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<Scheme::Curve>, Error> {
            let instance_values = instance
                .iter()
                .map(|values| {
                    let mut poly = domain.empty_lagrange();
                    assert_eq!(poly.len(), params.n() as usize);
                    if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                        panic!("Error::InstanceTooLarge");
                    }
                    for (poly, value) in poly.iter_mut().zip(values.iter()) {
                        *poly = *value;
                    }
                    poly
                })
                .collect::<Vec<_>>();

            let instance_polys: Vec<_> = instance_values
                .iter()
                .map(|poly| {
                    let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
                    domain.lagrange_to_coeff(lagrange_vec)
                })
                .collect();

            Ok(InstanceSingle {
                instance_values,
                instance_polys,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[derive(Clone)]
    struct AdviceSingle<C: CurveAffine, B: Basis> {
        pub advice_polys: Vec<Polynomial<C::Scalar, B>>,
        pub advice_blinds: Vec<Blind<C::Scalar>>,
    }

    struct WitnessCollection<'params, 'a, 'b, Scheme, P, C, E, R, T>
    where
        Scheme: CommitmentScheme<Curve = C>,
        P: Prover<'params, Scheme>,
        C: CurveAffine,
        E: EncodedChallenge<C>,
        R: RngCore + 'a,
        T: TranscriptWrite<C, E>,
    {
        params: &'params Scheme::ParamsProver,
        current_phase: sealed::Phase,
        advice: Vec<Polynomial<Assigned<C::Scalar>, LagrangeCoeff>>,
        challenges: &'b mut HashMap<usize, C::Scalar>,
        instances: &'b [&'a [C::Scalar]],
        usable_rows: RangeTo<usize>,
        advice_single: AdviceSingle<C, LagrangeCoeff>,
        instance_single: &'b InstanceSingle<C>,
        rng: &'b mut R,
        transcript: &'b mut &'a mut T,
        column_indices: [Vec<usize>; 3],
        challenge_indices: [Vec<usize>; 3],
        unusable_rows_start: usize,
        _marker: PhantomData<(P, E)>,
    }

    impl<'params, 'a, 'b, F, Scheme, P, C, E, R, T> Assignment<F>
        for WitnessCollection<'params, 'a, 'b, Scheme, P, C, E, R, T>
    where
        F: FieldExt,
        Scheme: CommitmentScheme<Curve = C>,
        P: Prover<'params, Scheme>,
        C: CurveAffine<ScalarExt = F>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about regions in this context.
        }

        fn exit_region(&mut self) {
            // Do nothing; we don't care about regions in this context.
        }

        fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.params.k()));
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(Error::BoundsFailure)
        }

        fn assign_advice<'r, 'v>(
            //<V, VR, A, AR>(
            &'r mut self,
            //_: A,
            column: Column<Advice>,
            row: usize,
            to: Value<Assigned<F>>,
        ) -> Result<Value<&'v Assigned<F>>, Error> {
            // TODO: better to assign all at once, deal with phases later
            // Ignore assignment of advice column in different phase than current one.
            if self.current_phase != column.column_type().phase {
                return Ok(Value::unknown());
            }

            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.params.k()));
            }

            let advice_get_mut = self
                .advice
                .get_mut(column.index())
                .expect("Not enough advice columns")
                .get_mut(row)
                .expect("Not enough rows");
            // We can get another 3-4% decrease in witness gen time by using the following unsafe code, but this skips all array bound checks so we should use it only if the performance gain is really necessary:
            /*
            let advice_get_mut = unsafe {
                self.advice
                    .get_unchecked_mut(column.index())
                    .get_unchecked_mut(row)
            };
            */
            *advice_get_mut = to
                .assign()
                .expect("No Value::unknown() in advice column allowed during create_proof");
            let immutable_raw_ptr = advice_get_mut as *const Assigned<F>;
            Ok(Value::known(unsafe { &*immutable_raw_ptr }))
        }

        fn assign_fixed(&mut self, _: Column<Fixed>, _: usize, _: Assigned<F>) {
            // We only care about advice columns here
        }

        fn copy(&mut self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) {
            // We only care about advice columns here
        }

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Assigned<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&mut self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn next_phase(&mut self) {
            let phase = self.current_phase.to_u8() as usize;
            if phase == 0 {
                // Absorb instances into transcript.
                // Do this here and not earlier in case we want to be able to mutate
                // the instances during synthesize in FirstPhase in the future
                if !P::QUERY_INSTANCE {
                    for values in self.instances.iter() {
                        for value in values.iter() {
                            self.transcript
                                .common_scalar(*value)
                                .expect("Absorbing instance value to transcript failed");
                        }
                    }
                } else {
                    let instance_commitments_projective: Vec<_> = self
                        .instance_single
                        .instance_values
                        .iter()
                        .map(|poly| self.params.commit_lagrange(poly, Blind::default()))
                        .collect();
                    let mut instance_commitments =
                        vec![C::identity(); instance_commitments_projective.len()];
                    C::CurveExt::batch_normalize(
                        &instance_commitments_projective,
                        &mut instance_commitments,
                    );
                    let instance_commitments = instance_commitments;
                    drop(instance_commitments_projective);

                    for commitment in &instance_commitments {
                        self.transcript
                            .common_point(*commitment)
                            .expect("Absorbing instance commitment to transcript failed");
                    }
                }
            }
            // Commit the advice columns in the current phase
            let mut advice_values = batch_invert_assigned_ref::<F>(
                self.column_indices
                    .get(phase)
                    .expect("The API only supports 3 phases right now")
                    .iter()
                    .map(|column_index| &self.advice[*column_index])
                    .collect(),
            );
            // Add blinding factors to advice columns
            for advice_values in &mut advice_values {
                for cell in &mut advice_values[self.unusable_rows_start..] {
                    *cell = F::random(&mut self.rng);
                }
            }
            // Compute commitments to advice column polynomials
            let blinds: Vec<_> = advice_values
                .iter()
                .map(|_| Blind(F::random(&mut self.rng)))
                .collect();
            let advice_commitments_projective: Vec<_> = advice_values
                .iter()
                .zip(blinds.iter())
                .map(|(poly, blind)| self.params.commit_lagrange(poly, *blind))
                .collect();
            let mut advice_commitments = vec![C::identity(); advice_commitments_projective.len()];
            C::CurveExt::batch_normalize(&advice_commitments_projective, &mut advice_commitments);
            let advice_commitments = advice_commitments;
            drop(advice_commitments_projective);

            for commitment in &advice_commitments {
                self.transcript
                    .write_point(*commitment)
                    .expect("Absorbing advice commitment to transcript failed");
            }
            for ((column_index, advice_poly), blind) in self.column_indices[phase]
                .iter()
                .zip(advice_values)
                .zip(blinds)
            {
                self.advice_single.advice_polys[*column_index] = advice_poly;
                self.advice_single.advice_blinds[*column_index] = blind;
            }
            for challenge_index in self.challenge_indices[phase].iter() {
                let existing = self.challenges.insert(
                    *challenge_index,
                    *self.transcript.squeeze_challenge_scalar::<()>(),
                );
                assert!(existing.is_none());
            }
            self.current_phase = self.current_phase.next();
        }
    }

    let mut column_indices = [(); 3].map(|_| vec![]);
    for (index, phase) in meta.advice_column_phase.iter().enumerate() {
        column_indices[phase.to_u8() as usize].push(index);
    }
    let mut challenge_indices = [(); 3].map(|_| vec![]);
    for (index, phase) in meta.challenge_phase.iter().enumerate() {
        challenge_indices[phase.to_u8() as usize].push(index);
    }

    let (advice, challenges) = {
        let mut advice = Vec::with_capacity(instances.len());
        let mut challenges = HashMap::<usize, Scheme::Scalar>::with_capacity(meta.num_challenges);

        let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);
        let phases = pk.vk.cs.phases().collect::<Vec<_>>();
        let num_phases = phases.len();
        // WARNING: this will currently not work if `circuits` has more than 1 circuit
        // because the original API squeezes the challenges for a phase after running all circuits
        // once in that phase.
        assert_eq!(
            circuits.len(),
            1,
            "New challenge API doesn't work with multiple circuits yet"
        );
        for ((circuit, instances), instance_single) in
            circuits.iter().zip(instances).zip(instance.iter())
        {
            let mut witness: WitnessCollection<Scheme, P, _, E, _, _> = WitnessCollection {
                params,
                current_phase: phases[0],
                advice: vec![domain.empty_lagrange_assigned(); meta.num_advice_columns],
                instances,
                challenges: &mut challenges,
                // The prover will not be allowed to assign values to advice
                // cells that exist within inactive rows, which include some
                // number of blinding factors and an extra row for use in the
                // permutation argument.
                usable_rows: ..unusable_rows_start,
                advice_single: AdviceSingle::<Scheme::Curve, LagrangeCoeff> {
                    advice_polys: vec![domain.empty_lagrange(); meta.num_advice_columns],
                    advice_blinds: vec![Blind::default(); meta.num_advice_columns],
                },
                instance_single,
                rng: &mut rng,
                transcript: &mut transcript,
                column_indices: column_indices.clone(),
                challenge_indices: challenge_indices.clone(),
                unusable_rows_start,
                _marker: PhantomData,
            };

            // while loop is for compatibility with circuits that do not use the new `next_phase` API to manage phases
            // If the circuit uses the new API, then the while loop will only execute once
            while witness.current_phase.to_u8() < num_phases as u8 {
                // Synthesize the circuit to obtain the witness and other information.
                ConcreteCircuit::FloorPlanner::synthesize(
                    &mut witness,
                    circuit,
                    config.clone(),
                    meta.constants.clone(),
                )
                .unwrap();
                if witness.current_phase.to_u8() < num_phases as u8 {
                    witness.next_phase();
                }
            }
            advice.push(witness.advice_single);
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    let lookups: Vec<Vec<lookup::prover::Permuted<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();

    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    // Commit to permutations.
    let permutations: Vec<permutation::prover::Committed<Scheme::Curve>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_polys,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let lookups: Vec<Vec<lookup::prover::Committed<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| lookup.commit_product(pk, params, beta, gamma, &mut rng, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript)?;

    // Obtain challenge for keeping all separate gates linearly independent
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    // Calculate the advice polys
    let advice: Vec<AdviceSingle<Scheme::Curve, Coeff>> = advice
        .into_iter()
        .map(
            |AdviceSingle {
                 advice_polys,
                 advice_blinds,
             }| {
                AdviceSingle {
                    advice_polys: advice_polys
                        .into_iter()
                        .map(|poly| domain.lagrange_to_coeff(poly))
                        .collect::<Vec<_>>(),
                    advice_blinds,
                }
            },
        )
        .collect();

    // Evaluate the h(X) polynomial
    let h_poly = pk.ev.evaluate_h(
        pk,
        &advice
            .iter()
            .map(|a| a.advice_polys.as_slice())
            .collect::<Vec<_>>(),
        &instance
            .iter()
            .map(|i| i.instance_polys.as_slice())
            .collect::<Vec<_>>(),
        &challenges,
        *y,
        *beta,
        *gamma,
        *theta,
        &lookups,
        &permutations,
    );

    // Construct the vanishing argument's h(X) commitments
    let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;

    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow(&[params.n(), 0, 0, 0]);

    if P::QUERY_INSTANCE {
        // Compute and hash instance evals for each circuit instance
        for instance in instance.iter() {
            // Evaluate polynomials at omega^i x
            let instance_evals: Vec<_> = meta
                .instance_queries
                .iter()
                .map(|&(column, at)| {
                    eval_polynomial(
                        &instance.instance_polys[column.index()],
                        domain.rotate_omega(*x, at),
                    )
                })
                .collect();

            // Hash each instance column evaluation
            for eval in instance_evals.iter() {
                transcript.write_scalar(*eval)?;
            }
        }
    }

    // Compute and hash advice evals for each circuit instance
    for advice in advice.iter() {
        // Evaluate polynomials at omega^i x
        let advice_evals: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(
                    &advice.advice_polys[column.index()],
                    domain.rotate_omega(*x, at),
                )
            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    }

    // Compute and hash fixed evals (shared across all circuit instances)
    let fixed_evals: Vec<_> = meta
        .fixed_queries
        .iter()
        .map(|&(column, at)| {
            eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(*x, at))
        })
        .collect();

    // Hash each fixed column evaluation
    for eval in fixed_evals.iter() {
        transcript.write_scalar(*eval)?;
    }

    let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;

    // Evaluate common permutation data
    pk.permutation.evaluate(x, transcript)?;

    // Evaluate the permutations, if any, at omega^i x.
    let permutations: Vec<permutation::prover::Evaluated<Scheme::Curve>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.construct().evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;

    // Evaluate the lookups, if any, at omega^i x.
    let lookups: Vec<Vec<lookup::prover::Evaluated<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let instances = instance
        .iter()
        .zip(advice.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .flat_map(|(((instance, advice), permutation), lookups)| {
            iter::empty()
                .chain(
                    P::QUERY_INSTANCE
                        .then_some(pk.vk.cs.instance_queries.iter().map(move |&(column, at)| {
                            ProverQuery {
                                point: domain.rotate_omega(*x, at),
                                poly: &instance.instance_polys[column.index()],
                                blind: Blind::default(),
                            }
                        }))
                        .into_iter()
                        .flatten(),
                )
                .chain(
                    pk.vk
                        .cs
                        .advice_queries
                        .iter()
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(*x, at),
                            poly: &advice.advice_polys[column.index()],
                            blind: advice.advice_blinds[column.index()],
                        }),
                )
                .chain(permutation.open(pk, x))
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)).into_iter())
        })
        .chain(
            pk.vk
                .cs
                .fixed_queries
                .iter()
                .map(|&(column, at)| ProverQuery {
                    point: domain.rotate_omega(*x, at),
                    poly: &pk.fixed_polys[column.index()],
                    blind: Blind::default(),
                }),
        )
        .chain(pk.permutation.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x));

    let prover = P::new(params);
    prover
        .create_proof(&mut rng, transcript, instances)
        .map_err(|_| Error::ConstraintSystemFailure)
}
