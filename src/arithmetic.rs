//! This module provides common utilities, traits and structures for group and
//! field arithmetic.
//!
//! This module is temporary, and the extension traits defined here are expected to be
//! upstreamed into the `ff` and `group` crates after some refactoring.

use std::hash::Hash;

use subtle::{Choice, ConditionallySelectable, CtOption};

pub trait CurveAffineExt: pasta_curves::arithmetic::CurveAffine {
    fn batch_add<const COMPLETE: bool, const LOAD_POINTS: bool>(
        points: &mut [Self],
        output_indices: &[u32],
        num_points: usize,
        offset: usize,
        bases: &[Self],
        base_positions: &[u32],
    );

    /// Unlike the `Coordinates` trait, this just returns the raw affine coordinantes without checking `is_on_curve`
    fn into_coordinates(self) -> (Self::Base, Self::Base);
}

pub trait BigPrimeField: pasta_curves::arithmetic::FieldExt + From<u128> + Hash {
    fn from_u64_digits(val: Vec<u64>) -> Self;

    /// Returns the base 2^32 little endian representation of the prime field element
    ///
    /// Basically same as `to_repr` but does not go further into bytes
    fn to_u32_digits(&self) -> Vec<u32>;

    /// Returns the base `2^bit_len` little endian representation of the prime field element
    /// up to `num_limbs` number of limbs (truncates any extra limbs)
    ///
    /// Basically same as `to_repr` but does not go further into bytes
    ///
    /// Undefined behavior if `bit_len > 64`
    fn to_u64_limbs(&self, num_limbs: usize, bit_len: usize) -> Vec<u64>;

    /// Returns the base `2^bit_len` little endian representation of the prime field element
    /// up to `num_limbs` number of limbs (truncates any extra limbs)
    ///
    /// Basically same as `to_repr` but does not go further into bytes
    ///
    /// Undefined behavor if `bit_len <= 64` or `bit_len > 128`
    fn to_u128_limbs(&self, num_limbs: usize, bit_len: usize) -> Vec<u128>;

    fn to_i128(&self) -> i128;
}

pub(crate) fn sqrt_tonelli_shanks<F: ff::PrimeField, S: AsRef<[u64]>>(
    f: &F,
    tm1d2: S,
) -> CtOption<F> {
    use subtle::ConstantTimeEq;

    // w = self^((t - 1) // 2)
    let w = f.pow_vartime(tm1d2);

    let mut v = F::S;
    let mut x = w * f;
    let mut b = x * w;

    // Initialize z as the 2^S root of unity.
    let mut z = F::root_of_unity();

    for max_v in (1..=F::S).rev() {
        let mut k = 1;
        let mut tmp = b.square();
        let mut j_less_than_v: Choice = 1.into();

        for j in 2..max_v {
            let tmp_is_one = tmp.ct_eq(&F::one());
            let squared = F::conditional_select(&tmp, &z, tmp_is_one).square();
            tmp = F::conditional_select(&squared, &tmp, tmp_is_one);
            let new_z = F::conditional_select(&z, &squared, tmp_is_one);
            j_less_than_v &= !j.ct_eq(&v);
            k = u32::conditional_select(&j, &k, tmp_is_one);
            z = F::conditional_select(&z, &new_z, j_less_than_v);
        }

        let result = x * z;
        x = F::conditional_select(&result, &x, b.ct_eq(&F::one()));
        z = z.square();
        b *= z;
        v = k;
    }

    CtOption::new(
        x,
        (x * x).ct_eq(f), // Only return Some if it's the square root.
    )
}

/// Compute a + b + carry, returning the result and the new carry over.
/// The carry input must be 0 or 1; otherwise the behavior is undefined.
/// The carryOut output is guaranteed to be 0 or 1.
#[inline(always)]
pub(crate) const fn adc(a: u64, b: u64, carry: bool) -> (u64, bool) {
    a.carrying_add(b, carry)
}

/// Compute a - b - borrow, returning the result and the new borrow.
#[inline(always)]
pub(crate) const fn sbb(a: u64, b: u64, borrow: bool) -> (u64, bool) {
    a.borrowing_sub(b, borrow)
}

/// Compute a + (b * c), returning the result and the new carry over.
// Alias madd1
#[inline(always)]
pub(crate) const fn macx(a: u64, b: u64, c: u64) -> (u64, u64) {
    b.carrying_mul(c, a)
}

/// Compute a + (b * c) + carry, returning the result and the new carry over.
// Alias madd2
#[inline(always)]
pub(crate) const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let (lo, hi) = b.carrying_mul(c, a);
    let (lo, carry) = lo.overflowing_add(carry);
    (lo, hi + carry as u64)
}

/// Compute a * b, returning the result.
#[inline(always)]
pub(crate) fn mul_512(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
    let (r0, carry) = macx(0, a[0], b[0]);
    let (r1, carry) = macx(carry, a[0], b[1]);
    let (r2, carry) = macx(carry, a[0], b[2]);
    let (r3, carry_out) = macx(carry, a[0], b[3]);

    let (r1, carry) = macx(r1, a[1], b[0]);
    let (r2, carry) = mac(r2, a[1], b[1], carry);
    let (r3, carry) = mac(r3, a[1], b[2], carry);
    let (r4, carry_out) = mac(carry_out, a[1], b[3], carry);

    let (r2, carry) = macx(r2, a[2], b[0]);
    let (r3, carry) = mac(r3, a[2], b[1], carry);
    let (r4, carry) = mac(r4, a[2], b[2], carry);
    let (r5, carry_out) = mac(carry_out, a[2], b[3], carry);

    let (r3, carry) = macx(r3, a[3], b[0]);
    let (r4, carry) = mac(r4, a[3], b[1], carry);
    let (r5, carry) = mac(r5, a[3], b[2], carry);
    let (r6, carry_out) = mac(carry_out, a[3], b[3], carry);

    [r0, r1, r2, r3, r4, r5, r6, carry_out]
}

#[inline(always)]
pub(crate) fn decompose_u64_digits_to_limbs(
    e: impl IntoIterator<Item = u64>,
    number_of_limbs: usize,
    bit_len: usize,
) -> Vec<u64> {
    debug_assert!(bit_len <= 64);

    let mut e = e.into_iter();
    let mask: u64 = (1u64 << bit_len) - 1u64;
    let mut u64_digit = e.next().unwrap();
    let mut rem = 64;
    (0..number_of_limbs)
        .map(|_| match rem.cmp(&bit_len) {
            core::cmp::Ordering::Greater => {
                let limb = u64_digit & mask;
                u64_digit >>= bit_len;
                rem -= bit_len;
                limb
            }
            core::cmp::Ordering::Equal => {
                let limb = u64_digit & mask;
                u64_digit = e.next().unwrap_or(0);
                rem = 64;
                limb
            }
            core::cmp::Ordering::Less => {
                let mut limb = u64_digit;
                u64_digit = e.next().unwrap_or(0);
                limb |= (u64_digit & ((1 << (bit_len - rem)) - 1)) << rem;
                u64_digit >>= bit_len - rem;
                rem += 64 - bit_len;
                limb
            }
        })
        .collect()
}

pub(crate) fn u64_digits_to_u128_limbs(
    e: impl IntoIterator<Item = u64>,
    num_limbs: usize,
    bit_len: usize,
) -> Vec<u128> {
    debug_assert!(bit_len > 64 && bit_len <= 128);

    let mut e = e.into_iter();
    let mut limb0 = e.next().unwrap_or(0) as u128;
    let mut rem = bit_len - 64;
    let mut u64_digit = e.next().unwrap_or(0);
    limb0 |= ((u64_digit & ((1 << rem) - 1)) as u128) << 64;
    u64_digit >>= rem;

    core::iter::once(limb0)
        .chain((1..num_limbs).map(|_| {
            let mut limb: u128 = u64_digit.into();
            let mut bits = rem;
            u64_digit = e.next().unwrap_or(0);
            // only possible if bit_len > 96
            if bit_len - bits >= 64 {
                limb |= (u64_digit as u128) << bits;
                u64_digit = e.next().unwrap_or(0);
                bits += 64;
            }
            rem = bit_len - bits;
            limb |= ((u64_digit & ((1 << rem) - 1)) as u128) << bits;
            u64_digit >>= rem;
            rem = 64 - rem;
            limb
        }))
        .collect()
}
