use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
};

use ff::Field;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
        ConstraintSystem, Error, Fixed, Instance, ProvingKey,
    },
    poly::{
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::SingleStrategy,
        },
        Rotation,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
    SerdeFormat,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use rand_core::OsRng;

#[derive(Clone, Copy)]
struct StandardPlonkConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    q_a: Column<Fixed>,
    q_b: Column<Fixed>,
    q_c: Column<Fixed>,
    q_ab: Column<Fixed>,
    constant: Column<Fixed>,
    #[allow(dead_code)]
    instance: Column<Instance>,
}

impl StandardPlonkConfig {
    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        let [a, b, c] = [(); 3].map(|_| meta.advice_column());
        let [q_a, q_b, q_c, q_ab, constant] = [(); 5].map(|_| meta.fixed_column());
        let instance = meta.instance_column();

        [a, b, c].map(|column| meta.enable_equality(column));

        meta.create_gate(
            "q_a·a + q_b·b + q_c·c + q_ab·a·b + constant + instance = 0",
            |meta| {
                let [a, b, c] = [a, b, c].map(|column| meta.query_advice(column, Rotation::cur()));
                let [q_a, q_b, q_c, q_ab, constant] = [q_a, q_b, q_c, q_ab, constant]
                    .map(|column| meta.query_fixed(column, Rotation::cur()));
                let instance = meta.query_instance(instance, Rotation::cur());
                Some(
                    q_a * a.clone()
                        + q_b * b.clone()
                        + q_c * c
                        + q_ab * a * b
                        + constant
                        + instance,
                )
            },
        );

        StandardPlonkConfig {
            a,
            b,
            c,
            q_a,
            q_b,
            q_c,
            q_ab,
            constant,
            instance,
        }
    }
}

#[derive(Clone, Default)]
struct StandardPlonk(Fr);

impl Circuit<Fr> for StandardPlonk {
    type Config = StandardPlonkConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        StandardPlonkConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "",
            |mut region| {
                region.assign_advice(config.a, 0, Value::known(self.0));
                region.assign_fixed(config.q_a, 0, -Fr::one());

                region.assign_advice(config.a, 1, Value::known(-Fr::from(5u64)));
                for (idx, column) in (1..).zip([
                    config.q_a,
                    config.q_b,
                    config.q_c,
                    config.q_ab,
                    config.constant,
                ]) {
                    region.assign_fixed(column, 1, Fr::from(idx as u64));
                }

                let a = region.assign_advice(config.a, 2, Value::known(Fr::one()));
                a.copy_advice(&mut region, config.b, 3);
                a.copy_advice(&mut region, config.c, 4);
                Ok(())
            },
        )
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 24)]
async fn main() -> std::io::Result<()> {
    let k = 22;

    let buf_size = 1024 * 1024;

    // let pk_path = "/home/ubuntu/playground/serialization-test.pk";
    // let f = File::open(pk_path)?;
    // let mut reader = BufReader::with_capacity(buf_size, f);
    // let start = std::time::Instant::now();
    // let pk = ProvingKey::<G1Affine>::read::<_, StandardPlonk>(&mut reader, SerdeFormat::RawBytes)
    //     .unwrap();
    // println!("SerdeFormat::RawBytes pk read time: {:?}", start.elapsed());

    // let pk_folder = "/home/ubuntu/playground/serialization-test/";
    let pk_folder = "/mnt/ramdisk/serialization-test";
    // let start = std::time::Instant::now();
    // pk.multi_thread_write(pk_folder, SerdeFormat::RawBytes)?;
    // println!(
    //     "SerdeFormat::RawBytes pk multi thread write time: {:?}",
    //     start.elapsed()
    // );

    let start = std::time::Instant::now();
    ProvingKey::<G1Affine>::multi_thread_read::<StandardPlonk>(pk_folder, SerdeFormat::RawBytes)
        .await?;
    println!(
        "SerdeFormat::RawBytes pk multi thread read time: {:?}",
        start.elapsed()
    );

    Ok(())
    // let circuit = StandardPlonk(Fr::random(OsRng));
    // let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    // let vk = keygen_vk(&params, &circuit).expect("vk should not fail");
    // let pk = keygen_pk(&params, vk, &circuit).expect("pk should not fail");

    // for buf_size in [1024, 8 * 1024, 1024 * 1024, 1024 * 1024 * 1024] {
    //     println!("buf_size: {buf_size}");
    //     // Using halo2_proofs serde implementation
    //     let f = File::create("serialization-test.pk")?;
    //     let mut writer = BufWriter::with_capacity(buf_size, f);
    //     let start = std::time::Instant::now();
    //     pk.write(&mut writer, SerdeFormat::RawBytes)?;
    //     writer.flush().unwrap();
    //     println!("SerdeFormat::RawBytes pk write time: {:?}", start.elapsed());

    //     let f = File::open("serialization-test.pk")?;
    //     let mut reader = BufReader::with_capacity(buf_size, f);
    //     let start = std::time::Instant::now();
    //     let pk =
    //         ProvingKey::<G1Affine>::read::<_, StandardPlonk>(&mut reader, SerdeFormat::RawBytes)
    //             .unwrap();
    //     println!("SerdeFormat::RawBytes pk read time: {:?}", start.elapsed());

    //     let metadata = fs::metadata("serialization-test.pk")?;
    //     let file_size = metadata.len();
    //     println!("The size of the file is {} bytes", file_size);
    //     std::fs::remove_file("serialization-test.pk")?;

    //     // Using bincode
    //     let f = File::create("serialization-test.pk")?;
    //     let mut writer = BufWriter::with_capacity(buf_size, f);
    //     let start = std::time::Instant::now();
    //     bincode::serialize_into(&mut writer, &pk).unwrap();
    //     writer.flush().unwrap();
    //     println!("bincode pk write time: {:?}", start.elapsed());

    //     let f = File::open("serialization-test.pk").unwrap();
    //     let mut reader = BufReader::with_capacity(buf_size, f);
    //     let start = std::time::Instant::now();
    //     let pk: ProvingKey<G1Affine> = bincode::deserialize_from(&mut reader).unwrap();
    //     println!("bincode pk read time: {:?}", start.elapsed());

    //     let metadata = fs::metadata("serialization-test.pk")?;
    //     let file_size = metadata.len();
    //     println!("The size of the file is {} bytes", file_size);
    //     std::fs::remove_file("serialization-test.pk").unwrap();

    //     let instances: &[&[Fr]] = &[&[circuit.clone().0]];
    //     let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    //     create_proof::<
    //         KZGCommitmentScheme<Bn256>,
    //         ProverGWC<'_, Bn256>,
    //         Challenge255<G1Affine>,
    //         _,
    //         Blake2bWrite<Vec<u8>, G1Affine, Challenge255<_>>,
    //         _,
    //     >(
    //         &params,
    //         &pk,
    //         &[circuit.clone()],
    //         &[instances],
    //         OsRng,
    //         &mut transcript,
    //     )
    //     .expect("prover should not fail");
    //     let proof = transcript.finalize();

    //     let strategy = SingleStrategy::new(&params);
    //     let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    //     assert!(verify_proof::<
    //         KZGCommitmentScheme<Bn256>,
    //         VerifierGWC<'_, Bn256>,
    //         Challenge255<G1Affine>,
    //         Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
    //         SingleStrategy<'_, Bn256>,
    //     >(
    //         &params,
    //         pk.get_vk(),
    //         strategy,
    //         &[instances],
    //         &mut transcript
    //     )
    //     .is_ok());
    // }
    // Ok(())
}
