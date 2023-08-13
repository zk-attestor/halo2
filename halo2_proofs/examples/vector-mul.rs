use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::Field,
    circuit::{AssignedCell, Chip, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{Advice, Assigned, Circuit, Column, ConstraintSystem, Error, Instance, Selector},
    poly::Rotation,
};

// ANCHOR: instructions
trait NumericInstructions<F: Field>: Chip<F> {
    /// Variable representing a number.
    type Num<'a>;

    /// Loads a number into the circuit as a private input.
    fn load_private<'a>(&self, layouter: impl Layouter<F>, a: &[Value<F>]) -> Vec<Self::Num<'a>>;

    /// Returns `c = a * b`. The caller is responsible for ensuring that `a.len() == b.len()`.
    fn mul<'a>(
        &self,
        layouter: impl Layouter<F>,
        a: &[Self::Num<'a>],
        b: &[Self::Num<'a>],
    ) -> Vec<Self::Num<'a>>;

    /// Exposes a number as a public input to the circuit.
    fn expose_public<'a>(&self, layouter: impl Layouter<F>, num: &Self::Num<'a>, row: usize);
}
// ANCHOR_END: instructions

// ANCHOR: chip
/// The chip that will implement our instructions! Chips store their own
/// config, as well as type markers if necessary.
struct FieldChip<F: Field> {
    config: FieldConfig,
    _marker: PhantomData<F>,
}
// ANCHOR_END: chip

// ANCHOR: chip-config
/// Chip state is stored in a config struct. This is generated by the chip
/// during configuration, and then stored inside the chip.
#[derive(Clone, Debug)]
struct FieldConfig {
    /// For this chip, we will use two advice columns to implement our instructions.
    /// These are also the columns through which we communicate with other parts of
    /// the circuit.
    advice: [Column<Advice>; 3],

    /// This is the public input (instance) column.
    instance: Column<Instance>,

    // We need a selector to enable the multiplication gate, so that we aren't placing
    // any constraints on cells where `NumericInstructions::mul` is not being used.
    // This is important when building larger circuits, where columns are used by
    // multiple sets of instructions.
    s_mul: Selector,
}

impl<F: Field> FieldChip<F> {
    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 3],
        instance: Column<Instance>,
    ) -> <Self as Chip<F>>::Config {
        meta.enable_equality(instance);
        for column in &advice {
            meta.enable_equality(*column);
        }
        let s_mul = meta.selector();

        // Define our multiplication gate!
        meta.create_gate("mul", |meta| {
            // To implement multiplication, we need three advice cells and a selector
            // cell. We arrange them like so:
            //
            // | a0  | a1  | a2  | s_mul |
            // |-----|-----|-----|-------|
            // | lhs | rhs | out | s_mul |
            //
            // Gates may refer to any relative offsets we want, but each distinct
            // offset adds a cost to the proof. The most common offsets are 0 (the
            // current row), 1 (the next row), and -1 (the previous row), for which
            // `Rotation` has specific constructors.
            let lhs = meta.query_advice(advice[0], Rotation::cur());
            let rhs = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[2], Rotation::cur());
            let s_mul = meta.query_selector(s_mul);

            // Finally, we return the polynomial expressions that constrain this gate.
            // For our multiplication gate, we only need a single polynomial constraint.
            //
            // The polynomial expressions returned from `create_gate` will be
            // constrained by the proving system to equal zero. Our expression
            // has the following properties:
            // - When s_mul = 0, any value is allowed in lhs, rhs, and out.
            // - When s_mul != 0, this constrains lhs * rhs = out.
            vec![s_mul * (lhs * rhs - out)]
        });

        FieldConfig {
            advice,
            instance,
            s_mul,
        }
    }
}
// ANCHOR_END: chip-config

// ANCHOR: chip-impl
impl<F: Field> Chip<F> for FieldChip<F> {
    type Config = FieldConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}
// ANCHOR_END: chip-impl

// ANCHOR: instructions-impl
/// A variable representing a number.
#[derive(Clone, Debug)]
struct Number<'a, F: Field>(AssignedCell<&'a Assigned<F>, F>);

impl<F: Field> NumericInstructions<F> for FieldChip<F> {
    type Num<'a> = Number<'a, F>;

    fn load_private<'a>(
        &self,
        mut layouter: impl Layouter<F>,
        values: &[Value<F>],
    ) -> Vec<Self::Num<'a>> {
        let config = self.config();

        layouter
            .assign_region(
                || "load private",
                |mut region| {
                    Ok(values
                        .iter()
                        .enumerate()
                        .map(|(i, value)| region.assign_advice(config.advice[0], i, *value))
                        .map(Number)
                        .collect())
                },
            )
            .unwrap()
    }

    fn mul<'a>(
        &self,
        mut layouter: impl Layouter<F>,
        a: &[Self::Num<'a>],
        b: &[Self::Num<'a>],
    ) -> Vec<Self::Num<'a>> {
        let config = self.config();
        assert_eq!(a.len(), b.len());

        #[cfg(feature = "thread-safe-region")]
        {
            use rayon::prelude::{
                IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
            };
            layouter.assign_region(
                || "mul",
                |region: Region<'_, F>| {
                    let thread_safe_region = std::sync::Mutex::new(region);
                    a.par_iter()
                        .zip(b.par_iter())
                        .enumerate()
                        .map(|(i, (a, b))| {
                            let mut region = thread_safe_region.lock().unwrap();

                            config.s_mul.enable(&mut region, i)?;

                            a.0.copy_advice(|| "lhs", &mut region, config.advice[0], i)?;
                            b.0.copy_advice(|| "rhs", &mut region, config.advice[1], i)?;

                            let value = a.0.value().copied() * b.0.value();

                            // Finally, we do the assignment to the output, returning a
                            // variable to be used in another part of the circuit.
                            region
                                .assign_advice(|| "lhs * rhs", config.advice[2], i, || value)
                                .map(Number)
                        })
                        .collect()
                },
            )
        }

        #[cfg(not(feature = "thread-safe-region"))]
        layouter
            .assign_region(
                || "mul",
                |mut region: Region<'_, F>| {
                    Ok(a.iter()
                        .zip(b.iter())
                        .enumerate()
                        .map(|(i, (a, b))| {
                            config.s_mul.enable(&mut region, i).unwrap();

                            a.0.copy_advice(&mut region, config.advice[0], i);
                            b.0.copy_advice(&mut region, config.advice[1], i);

                            let value = a.0.value().copied().copied() * b.0.value().copied();

                            // Finally, we do the assignment to the output, returning a
                            // variable to be used in another part of the circuit.
                            Number(region.assign_advice(config.advice[2], i, value))
                        })
                        .collect())
                },
            )
            .unwrap()
    }

    fn expose_public(&self, mut layouter: impl Layouter<F>, num: &Self::Num<'_>, row: usize) {
        let config = self.config();

        layouter.constrain_instance(*num.0.cell(), config.instance, row)
    }
}
// ANCHOR_END: instructions-impl

// ANCHOR: circuit
/// The full circuit implementation.
///
/// In this struct we store the private input variables. We use `Option<F>` because
/// they won't have any value during key generation. During proving, if any of these
/// were `None` we would get an error.
#[derive(Default)]
struct MyCircuit<F: Field> {
    a: Vec<Value<F>>,
    b: Vec<Value<F>>,
}

impl<F: Field> Circuit<F> for MyCircuit<F> {
    // Since we are using a single chip for everything, we can just reuse its config.
    type Config = FieldConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // We create the three advice columns that FieldChip uses for I/O.
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // We also need an instance column to store public inputs.
        let instance = meta.instance_column();

        FieldChip::configure(meta, advice, instance)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let field_chip = FieldChip::<F>::construct(config);

        // Load our private values into the circuit.
        let a = field_chip.load_private(layouter.namespace(|| "load a"), &self.a);
        let b = field_chip.load_private(layouter.namespace(|| "load b"), &self.b);

        let ab = field_chip.mul(layouter.namespace(|| "a * b"), &a, &b);

        for (i, c) in ab.iter().enumerate() {
            // Expose the result as a public input to the circuit.
            field_chip.expose_public(layouter.namespace(|| "expose c"), c, i);
        }
        Ok(())
    }
}
// ANCHOR_END: circuit

fn main() {
    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::Fp;

    const N: usize = 20000;
    // ANCHOR: test-circuit
    // The number of rows in our circuit cannot exceed 2^k. Since our example
    // circuit is very small, we can pick a very small value here.
    let k = 16;

    // Prepare the private and public inputs to the circuit!
    let a = [Fp::from(2); N];
    let b = [Fp::from(3); N];
    let c: Vec<Fp> = a.iter().zip(b).map(|(&a, b)| a * b).collect();

    // Instantiate the circuit with the private inputs.
    let circuit = MyCircuit {
        a: a.iter().map(|&x| Value::known(x)).collect(),
        b: b.iter().map(|&x| Value::known(x)).collect(),
    };

    // Arrange the public input. We expose the multiplication result in row 0
    // of the instance column, so we position it there in our public inputs.
    let mut public_inputs = c;

    let start = std::time::Instant::now();
    // Given the correct public input, our circuit will verify.
    let prover = MockProver::run(k, &circuit, vec![public_inputs.clone()]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
    println!("positive test took {:?}", start.elapsed());

    // If we try some other public input, the proof will fail!
    let start = std::time::Instant::now();
    public_inputs[0] += Fp::one();
    let prover = MockProver::run(k, &circuit, vec![public_inputs]).unwrap();
    assert!(prover.verify().is_err());
    println!("negative test took {:?}", start.elapsed());
    // ANCHOR_END: test-circuit
}
