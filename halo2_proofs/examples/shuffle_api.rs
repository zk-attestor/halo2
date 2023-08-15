use std::{hash::Hash, marker::PhantomData, vec};

use ff::{FromUniformBytes, WithSmallOrderMulGroup};
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
        ConstraintSystem, Error, Fixed, Selector,
    },
    poly::{
        commitment::ParamsProver,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
        Rotation,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::{bn256::Bn256, serde::SerdeObject, CurveAffine};
use pairing::MultiMillerLoop;
use rand_core::OsRng;

struct ShuffleChip<F: Field> {
    config: ShuffleConfig,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
struct ShuffleConfig {
    input_0: Column<Advice>,
    input_1: Column<Fixed>,
    shuffle_0: Column<Advice>,
    shuffle_1: Column<Advice>,
    s_input: Selector,
    s_shuffle: Selector,
}

impl<F: Field> ShuffleChip<F> {
    fn construct(config: ShuffleConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        input_0: Column<Advice>,
        input_1: Column<Fixed>,
        shuffle_0: Column<Advice>,
        shuffle_1: Column<Advice>,
    ) -> ShuffleConfig {
        let s_shuffle = meta.complex_selector();
        let s_input = meta.complex_selector();
        meta.shuffle("shuffle", |meta| {
            let s_input = meta.query_selector(s_input);
            let s_shuffle = meta.query_selector(s_shuffle);
            let input_0 = meta.query_advice(input_0, Rotation::cur());
            let input_1 = meta.query_fixed(input_1, Rotation::cur());
            let shuffle_0 = meta.query_advice(shuffle_0, Rotation::cur());
            let shuffle_1 = meta.query_advice(shuffle_1, Rotation::cur());
            vec![
                (s_input.clone() * input_0, s_shuffle.clone() * shuffle_0),
                (s_input * input_1, s_shuffle * shuffle_1),
            ]
        });
        ShuffleConfig {
            input_0,
            input_1,
            shuffle_0,
            shuffle_1,
            s_input,
            s_shuffle,
        }
    }
}

#[derive(Default)]
struct MyCircuit<F: Field> {
    input_0: Vec<Value<F>>,
    input_1: Vec<F>,
    shuffle_0: Vec<Value<F>>,
    shuffle_1: Vec<Value<F>>,
}

impl<F: Field> Circuit<F> for MyCircuit<F> {
    // Since we are using a single chip for everything, we can just reuse its config.
    type Config = ShuffleConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    fn params(&self) -> Self::Params {}

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let input_0 = meta.advice_column();
        let input_1 = meta.fixed_column();
        let shuffle_0 = meta.advice_column();
        let shuffle_1 = meta.advice_column();
        ShuffleChip::configure(meta, input_0, input_1, shuffle_0, shuffle_1)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let ch = ShuffleChip::<F>::construct(config);
        layouter.assign_region(
            || "load inputs & shuffles",
            |mut region| {
                for (i, (input_0, input_1)) in
                    self.input_0.iter().zip(self.input_1.iter()).enumerate()
                {
                    region.assign_advice(ch.config.input_0, i, *input_0);
                    region.assign_fixed(ch.config.input_1, i, *input_1);
                    ch.config.s_input.enable(&mut region, i)?;
                }

                for (i, (shuffle_0, shuffle_1)) in
                    self.shuffle_0.iter().zip(self.shuffle_1.iter()).enumerate()
                {
                    region.assign_advice(ch.config.shuffle_0, i, *shuffle_0);
                    region.assign_advice(ch.config.shuffle_1, i, *shuffle_1);
                    ch.config.s_shuffle.enable(&mut region, i)?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn test_prover<E: MultiMillerLoop>(k: u32, circuit: MyCircuit<E::Fr>, expected: bool)
where
    E::Fr: Hash + FromUniformBytes<64> + WithSmallOrderMulGroup<3>,
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + SerdeObject,
    E::G2Affine: CurveAffine + SerdeObject,
{
    let params = ParamsKZG::<E>::new(k);
    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk, &circuit).unwrap();

    let proof = {
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

        create_proof::<KZGCommitmentScheme<E>, ProverSHPLONK<E>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[]],
            OsRng,
            &mut transcript,
        )
        .expect("proof generation should not fail");

        transcript.finalize()
    };

    let accepted = {
        let strategy = SingleStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

        verify_proof::<KZGCommitmentScheme<E>, VerifierSHPLONK<E>, _, _, _>(
            &params,
            pk.get_vk(),
            strategy,
            &[&[]],
            &mut transcript,
        )
        .is_ok()
    };

    assert_eq!(accepted, expected);
}

fn main() {
    use halo2_proofs::dev::MockProver;
    use halo2curves::bn256::Fr;
    const K: u32 = 4;
    let input_0 = [1, 2, 4, 1]
        .map(|e: u64| Value::known(Fr::from(e)))
        .to_vec();
    let input_1 = [10, 20, 40, 10].map(Fr::from).to_vec();
    let shuffle_0 = [4, 1, 1, 2]
        .map(|e: u64| Value::known(Fr::from(e)))
        .to_vec();
    let shuffle_1 = [40, 10, 10, 20]
        .map(|e: u64| Value::known(Fr::from(e)))
        .to_vec();
    let circuit = MyCircuit {
        input_0,
        input_1,
        shuffle_0,
        shuffle_1,
    };
    let prover = MockProver::run(K, &circuit, vec![]).unwrap();
    prover.assert_satisfied();
    test_prover::<Bn256>(K, circuit, true);
}
