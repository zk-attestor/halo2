#[macro_export]
macro_rules! field_common {
    (
        $field:ident,
        $modulus:ident,
        $inv:ident,
        $modulus_str:ident,
        $two_inv:ident,
        $root_of_unity_inv:ident,
        $delta:ident,
        $zeta:ident,
        $r:ident,
        $r2:ident,
        $r3:ident
    ) => {
        impl $field {
            /// Returns zero, the additive identity.
            #[inline]
            pub const fn zero() -> $field {
                $field([0, 0, 0, 0])
            }

            /// Returns one, the multiplicative identity.
            #[inline]
            pub const fn one() -> $field {
                $r
            }

            fn from_u512(limbs: [u64; 8]) -> $field {
                // We reduce an arbitrary 512-bit number by decomposing it into two 256-bit digits
                // with the higher bits multiplied by 2^256. Thus, we perform two reductions
                //
                // 1. the lower bits are multiplied by R^2, as normal
                // 2. the upper bits are multiplied by R^2 * 2^256 = R^3
                //
                // and computing their sum in the field. It remains to see that arbitrary 256-bit
                // numbers can be placed into Montgomery form safely using the reduction. The
                // reduction works so long as the product is less than R=2^256 multiplied by
                // the modulus. This holds because for any `c` smaller than the modulus, we have
                // that (2^256 - 1)*c is an acceptable product for the reduction. Therefore, the
                // reduction always works so long as `c` is in the field; in this case it is either the
                // constant `R2` or `R3`.
                let d0 = $field([limbs[0], limbs[1], limbs[2], limbs[3]]);
                let d1 = $field([limbs[4], limbs[5], limbs[6], limbs[7]]);
                // Convert to Montgomery form
                d0 * $r2 + d1 * $r3
            }

            /// Converts from an integer represented in little endian
            /// into its (congruent) `$field` representation.
            pub fn from_raw(val: [u64; 4]) -> Self {
                (&$field(val)).mul(&$r2)
            }

            /// Converts from an integer represented in little endian
            /// into its (congruent) `$field` representation.
            pub const fn const_from_raw(val: [u64; 4]) -> Self {
                (&$field(val)).const_mul(&$r2)
            }

            /// Attempts to convert a little-endian byte representation of
            /// a scalar into a `Fr`, failing if the input is not canonical.
            pub fn from_bytes(bytes: &[u8; 32]) -> CtOption<$field> {
                <Self as ff::PrimeField>::from_repr(*bytes)
            }

            /// Converts an element of `Fr` into a byte representation in
            /// little-endian byte order.
            pub fn to_bytes(&self) -> [u8; 32] {
                <Self as ff::PrimeField>::to_repr(self)
            }
        }

        impl Group for $field {
            type Scalar = Self;

            fn group_zero() -> Self {
                Self::zero()
            }
            fn group_add(&mut self, rhs: &Self) {
                *self += *rhs;
            }
            fn group_sub(&mut self, rhs: &Self) {
                *self -= *rhs;
            }
            fn group_scale(&mut self, by: &Self::Scalar) {
                *self *= *by;
            }
        }

        impl fmt::Debug for $field {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let tmp = self.to_repr();
                write!(f, "0x")?;
                for &b in tmp.iter().rev() {
                    write!(f, "{:02x}", b)?;
                }
                Ok(())
            }
        }

        impl Default for $field {
            #[inline]
            fn default() -> Self {
                Self::zero()
            }
        }

        impl From<bool> for $field {
            fn from(bit: bool) -> $field {
                if bit {
                    $field::one()
                } else {
                    $field::zero()
                }
            }
        }

        impl From<u64> for $field {
            fn from(val: u64) -> $field {
                $field([val, 0, 0, 0]) * $r2
            }
        }

        impl From<u128> for $field {
            fn from(v: u128) -> $field {
                $field::from_raw([v as u64, (v >> 64) as u64, 0, 0])
            }
        }

        impl ConstantTimeEq for $field {
            fn ct_eq(&self, other: &Self) -> Choice {
                self.0[0].ct_eq(&other.0[0])
                    & self.0[1].ct_eq(&other.0[1])
                    & self.0[2].ct_eq(&other.0[2])
                    & self.0[3].ct_eq(&other.0[3])
            }
        }

        impl core::cmp::Ord for $field {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                match self.0[3].cmp(&other.0[3]) {
                    core::cmp::Ordering::Equal => {}
                    ne => return ne,
                }
                match self.0[2].cmp(&other.0[2]) {
                    core::cmp::Ordering::Equal => {}
                    ne => return ne,
                }
                match self.0[1].cmp(&other.0[1]) {
                    core::cmp::Ordering::Equal => {}
                    ne => return ne,
                }
                self.0[0].cmp(&other.0[0])
            }
        }

        impl core::cmp::PartialOrd for $field {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl ConditionallySelectable for $field {
            fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
                $field([
                    u64::conditional_select(&a.0[0], &b.0[0], choice),
                    u64::conditional_select(&a.0[1], &b.0[1], choice),
                    u64::conditional_select(&a.0[2], &b.0[2], choice),
                    u64::conditional_select(&a.0[3], &b.0[3], choice),
                ])
            }
        }

        impl<'a> Neg for &'a $field {
            type Output = $field;

            #[inline]
            fn neg(self) -> $field {
                self.neg()
            }
        }

        impl Neg for $field {
            type Output = $field;

            #[inline]
            fn neg(self) -> $field {
                -&self
            }
        }

        impl<'a, 'b> Sub<&'b $field> for &'a $field {
            type Output = $field;

            #[inline]
            fn sub(self, rhs: &'b $field) -> $field {
                self.sub(rhs)
            }
        }

        impl<'a, 'b> Add<&'b $field> for &'a $field {
            type Output = $field;

            #[inline]
            fn add(self, rhs: &'b $field) -> $field {
                self.add(rhs)
            }
        }

        impl<'a, 'b> Mul<&'b $field> for &'a $field {
            type Output = $field;

            #[inline]
            fn mul(self, rhs: &'b $field) -> $field {
                self.mul(rhs)
            }
        }

        impl From<$field> for [u64; 4] {
            fn from(elt: $field) -> [u64; 4] {
                // Turn into canonical form by computing
                // (a.R) / R = a
                #[cfg(feature = "asm")]
                let tmp = $field::montgomery_reduce(&[
                    elt.0[0], elt.0[1], elt.0[2], elt.0[3], 0, 0, 0, 0,
                ]);

                #[cfg(not(feature = "asm"))]
                let tmp = $field::montgomery_reduce_short(elt.0[0], elt.0[1], elt.0[2], elt.0[3]);

                tmp.0
            }
        }

        impl From<$field> for [u8; 32] {
            fn from(value: $field) -> [u8; 32] {
                value.to_repr()
            }
        }

        impl<'a> From<&'a $field> for [u8; 32] {
            fn from(value: &'a $field) -> [u8; 32] {
                value.to_repr()
            }
        }

        impl FieldExt for $field {
            const MODULUS: &'static str = $modulus_str;
            const TWO_INV: Self = $two_inv;
            const ROOT_OF_UNITY_INV: Self = $root_of_unity_inv;
            const DELTA: Self = $delta;
            const ZETA: Self = $zeta;

            fn from_u128(v: u128) -> Self {
                $field::from_raw([v as u64, (v >> 64) as u64, 0, 0])
            }

            /// Converts a 512-bit little endian integer into
            /// a `$field` by reducing by the modulus.
            fn from_bytes_wide(bytes: &[u8; 64]) -> $field {
                $field::from_u512([
                    u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                    u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                    u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
                    u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
                    u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
                    u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
                    u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
                    u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
                ])
            }

            fn get_lower_128(&self) -> u128 {
                let tmp =
                    $field::montgomery_reduce_short(self.0[0], self.0[1], self.0[2], self.0[3]);

                u128::from(tmp.0[0]) | (u128::from(tmp.0[1]) << 64)
            }
        }

        impl BigPrimeField for $field {
            fn from_u64_digits(val: Vec<u64>) -> Self {
                debug_assert!(val.len() <= 4);
                let mut raw = [0u64; 4];
                raw[..val.len()].copy_from_slice(&val);
                Self::from_raw(raw)
            }

            fn to_u32_digits(&self) -> Vec<u32> {
                let tmp: [u64; 4] = (*self).into();
                tmp.iter()
                    .flat_map(|digit| [(digit & (u32::MAX as u64)) as u32, (digit >> 32) as u32])
                    .collect()
            }

            fn to_u64_limbs(&self, num_limbs: usize, bit_len: usize) -> Vec<u64> {
                let tmp: [u64; 4] = (*self).into();
                decompose_u64_digits_to_limbs(tmp, num_limbs, bit_len)
            }

            fn to_u128_limbs(&self, num_limbs: usize, bit_len: usize) -> Vec<u128> {
                let tmp: [u64; 4] = (*self).into();
                u64_digits_to_u128_limbs(tmp, num_limbs, bit_len)
            }

            fn to_i128(&self) -> i128 {
                let tmp: [u64; 4] = (*self).into();
                if tmp[2] == 0 && tmp[3] == 0 {
                    i128::from(tmp[0]) | (i128::from(tmp[1]) << 64)
                } else {
                    // modulus - tmp
                    let (a0, borrow) = $modulus.0[0].overflowing_sub(tmp[0]);
                    let (a1, _) = sbb($modulus.0[1], tmp[1], borrow);

                    -(i128::from(a0) | (i128::from(a1) << 64))
                }
            }
        }
    };
}

#[macro_export]
macro_rules! field_arithmetic {
    ($field:ident, $modulus:ident, $inv:ident, $field_type:ident) => {
        field_specific!($field, $modulus, $inv, $field_type);
        impl $field {
            /// Doubles this field element.
            #[inline]
            pub const fn double(&self) -> $field {
                self.add(self)
            }

            /// Multiplies `rhs` by `self`, returning the result.
            #[inline]
            pub const fn const_mul(&self, rhs: &Self) -> $field {
                // Schoolbook multiplication

                let (r0, carry) = self.0[0].widening_mul(rhs.0[0]);
                let (r1, carry) = macx(carry, self.0[0], rhs.0[1]);
                let (r2, carry) = macx(carry, self.0[0], rhs.0[2]);
                let (r3, r4) = macx(carry, self.0[0], rhs.0[3]);

                let (r1, carry) = macx(r1, self.0[1], rhs.0[0]);
                let (r2, carry) = mac(r2, self.0[1], rhs.0[1], carry);
                let (r3, carry) = mac(r3, self.0[1], rhs.0[2], carry);
                let (r4, r5) = mac(r4, self.0[1], rhs.0[3], carry);

                let (r2, carry) = macx(r2, self.0[2], rhs.0[0]);
                let (r3, carry) = mac(r3, self.0[2], rhs.0[1], carry);
                let (r4, carry) = mac(r4, self.0[2], rhs.0[2], carry);
                let (r5, r6) = mac(r5, self.0[2], rhs.0[3], carry);

                let (r3, carry) = macx(r3, self.0[3], rhs.0[0]);
                let (r4, carry) = mac(r4, self.0[3], rhs.0[1], carry);
                let (r5, carry) = mac(r5, self.0[3], rhs.0[2], carry);
                let (r6, r7) = mac(r6, self.0[3], rhs.0[3], carry);

                $field::montgomery_reduce(r0, r1, r2, r3, r4, r5, r6, r7)
            }

            /// Squares this element.
            #[inline]
            pub const fn square(&self) -> $field {
                let r0;
                let mut r1;
                let mut r2;
                let mut r3;
                let mut r4;
                let mut r5;
                let mut r6;
                let mut r7;
                let mut carry;
                let mut carry2;

                (r1, carry) = self.0[0].widening_mul(self.0[1]);
                (r2, carry) = self.0[0].carrying_mul(self.0[2], carry);
                (r3, r4) = self.0[0].carrying_mul(self.0[3], carry);

                (r3, carry) = macx(r3, self.0[1], self.0[2]);
                (r4, r5) = mac(r4, self.0[1], self.0[3], carry);

                (r5, r6) = macx(r5, self.0[2], self.0[3]);

                r7 = r6 >> 63;
                r6 = (r6 << 1) | (r5 >> 63);
                r5 = (r5 << 1) | (r4 >> 63);
                r4 = (r4 << 1) | (r3 >> 63);
                r3 = (r3 << 1) | (r2 >> 63);
                r2 = (r2 << 1) | (r1 >> 63);
                r1 <<= 1;

                (r0, carry) = self.0[0].widening_mul(self.0[0]);
                (r1, carry2) = r1.overflowing_add(carry);
                (r2, carry) = mac(r2, self.0[1], self.0[1], carry2 as u64);
                (r3, carry2) = r3.overflowing_add(carry);
                (r4, carry) = mac(r4, self.0[2], self.0[2], carry2 as u64);
                (r5, carry2) = r5.overflowing_add(carry);
                (r6, carry) = mac(r6, self.0[3], self.0[3], carry2 as u64);
                r7 = r7.wrapping_add(carry);

                $field::montgomery_reduce(r0, r1, r2, r3, r4, r5, r6, r7)
            }

            /// Subtracts `rhs` from `self`, returning the result.
            #[inline]
            pub const fn sub(&self, rhs: &Self) -> Self {
                let mut d0;
                let mut d1;
                let mut d2;
                let mut d3;
                let mut borrow;

                (d0, borrow) = self.0[0].overflowing_sub(rhs.0[0]);
                (d1, borrow) = sbb(self.0[1], rhs.0[1], borrow);
                (d2, borrow) = sbb(self.0[2], rhs.0[2], borrow);
                (d3, borrow) = sbb(self.0[3], rhs.0[3], borrow);

                // If underflow occurred on the final limb, add the modulus.
                if borrow {
                    (d0, borrow) = d0.overflowing_add($modulus.0[0]);
                    (d1, borrow) = d1.carrying_add($modulus.0[1], borrow);
                    (d2, borrow) = d2.carrying_add($modulus.0[2], borrow);
                    (d3, _) = d3.carrying_add($modulus.0[3], borrow);
                }
                $field([d0, d1, d2, d3])
            }

            /// Negates `self`.
            #[inline]
            pub const fn neg(&self) -> Self {
                if self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0 {
                    return $field([0, 0, 0, 0]);
                }
                // Subtract `self` from `MODULUS` to negate. Ignore the final
                // borrow because it cannot underflow; self is guaranteed to
                // be in the field.
                let (d0, borrow) = $modulus.0[0].overflowing_sub(self.0[0]);
                let (d1, borrow) = sbb($modulus.0[1], self.0[1], borrow);
                let (d2, borrow) = sbb($modulus.0[2], self.0[2], borrow);
                let d3 = $modulus.0[3] - (self.0[3] + borrow as u64);

                $field([d0, d1, d2, d3])
            }

            /// Montgomery reduce where last 4 registers are 0
            #[inline(always)]
            pub(crate) fn montgomery_reduce_short(
                mut r0: u64,
                mut r1: u64,
                mut r2: u64,
                mut r3: u64,
            ) -> $field {
                // The Montgomery reduction here is based on Algorithm 14.32 in
                // Handbook of Applied Cryptography
                // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
                let mut k;

                k = r0.wrapping_mul($inv);
                (_, r0) = macx(r0, k, $modulus.0[0]);
                (r1, r0) = mac(r1, k, $modulus.0[1], r0);
                (r2, r0) = mac(r2, k, $modulus.0[2], r0);
                (r3, r0) = mac(r3, k, $modulus.0[3], r0);

                k = r1.wrapping_mul($inv);
                (_, r1) = macx(r1, k, $modulus.0[0]);
                (r2, r1) = mac(r2, k, $modulus.0[1], r1);
                (r3, r1) = mac(r3, k, $modulus.0[2], r1);
                (r0, r1) = mac(r0, k, $modulus.0[3], r1);

                k = r2.wrapping_mul($inv);
                (_, r2) = macx(r2, k, $modulus.0[0]);
                (r3, r2) = mac(r3, k, $modulus.0[1], r2);
                (r0, r2) = mac(r0, k, $modulus.0[2], r2);
                (r1, r2) = mac(r1, k, $modulus.0[3], r2);

                k = r3.wrapping_mul($inv);
                (_, r3) = macx(r3, k, $modulus.0[0]);
                (r0, r3) = mac(r0, k, $modulus.0[1], r3);
                (r1, r3) = mac(r1, k, $modulus.0[2], r3);
                (r2, r3) = mac(r2, k, $modulus.0[3], r3);

                // Result may be within MODULUS of the correct value
                if !$field::is_less_than([r0, r1, r2, r3], $modulus.0) {
                    let mut borrow;
                    (r0, borrow) = r0.overflowing_sub($modulus.0[0]);
                    (r1, borrow) = sbb(r1, $modulus.0[1], borrow);
                    (r2, borrow) = sbb(r2, $modulus.0[2], borrow);
                    r3 = r3.wrapping_sub($modulus.0[3] + borrow as u64);
                }
                $field([r0, r1, r2, r3])
                // (&$field([r0, r1, r2, r3])).sub(&$modulus)
            }

            #[inline(always)]
            fn is_less_than(x: [u64; 4], y: [u64; 4]) -> bool {
                match x[3].cmp(&y[3]) {
                    core::cmp::Ordering::Less => return true,
                    core::cmp::Ordering::Greater => return false,
                    _ => {}
                }
                match x[2].cmp(&y[2]) {
                    core::cmp::Ordering::Less => return true,
                    core::cmp::Ordering::Greater => return false,
                    _ => {}
                }
                match x[1].cmp(&y[1]) {
                    core::cmp::Ordering::Less => return true,
                    core::cmp::Ordering::Greater => return false,
                    _ => {}
                }
                x[0].lt(&y[0])
            }
        }
    };
}

#[macro_export]
macro_rules! field_specific {
    ($field:ident, $modulus:ident, $inv:ident, sparse) => {
        impl $field {
            /// Adds `rhs` to `self`, returning the result.
            #[inline]
            pub const fn add(&self, rhs: &Self) -> Self {
                let (d0, carry) = self.0[0].overflowing_add(rhs.0[0]);
                let (d1, carry) = self.0[1].carrying_add(rhs.0[1], carry);
                let (d2, carry) = self.0[2].carrying_add(rhs.0[2], carry);
                // sparse means that the sum won't overflow the top register
                let d3 = self.0[3] + rhs.0[3] + carry as u64;

                // Attempt to subtract the modulus, to ensure the value
                // is smaller than the modulus.
                (&$field([d0, d1, d2, d3])).sub(&$modulus)
            }

            /// Multiplies `rhs` by `self`, returning the result.
            #[inline]
            pub fn mul(&self, rhs: &Self) -> $field {
                // When the highest bit in the top register of the modulus is 0 and the rest of the bits are not all 1, we can use an optimization from the gnark team: https://hackmd.io/@gnark/modular_multiplication

                // I think this is exactly the same as the previous `mul` implementation (see `const_mul`) with `montgomery_reduce` at the end (where `montgomery_reduce` is slightly cheaper in "sparse" setting)
                // Maybe the use of mutable variables is slightly more efficient?
                let mut r0;
                let mut r1;
                let mut t0;
                let mut t1;
                let mut t2;
                let mut t3;
                let mut k;

                (t0, r0) = self.0[0].widening_mul(rhs.0[0]);
                k = t0.wrapping_mul($inv);
                (_, r1) = macx(t0, k, $modulus.0[0]);
                (t1, r0) = self.0[0].carrying_mul(rhs.0[1], r0);
                (t0, r1) = mac(t1, k, $modulus.0[1], r1);
                (t2, r0) = self.0[0].carrying_mul(rhs.0[2], r0);
                (t1, r1) = mac(t2, k, $modulus.0[2], r1);
                (t3, r0) = self.0[0].carrying_mul(rhs.0[3], r0);
                (t2, r1) = mac(t3, k, $modulus.0[3], r1);
                t3 = r0 + r1;

                (t0, r0) = macx(t0, self.0[1], rhs.0[0]);
                k = t0.wrapping_mul($inv);
                (_, r1) = macx(t0, k, $modulus.0[0]);
                (t1, r0) = mac(t1, self.0[1], rhs.0[1], r0);
                (t0, r1) = mac(t1, k, $modulus.0[1], r1);
                (t2, r0) = mac(t2, self.0[1], rhs.0[2], r0);
                (t1, r1) = mac(t2, k, $modulus.0[2], r1);
                (t3, r0) = mac(t3, self.0[1], rhs.0[3], r0);
                (t2, r1) = mac(t3, k, $modulus.0[3], r1);
                t3 = r0 + r1;

                (t0, r0) = macx(t0, self.0[2], rhs.0[0]);
                k = t0.wrapping_mul($inv);
                (_, r1) = macx(t0, k, $modulus.0[0]);
                (t1, r0) = mac(t1, self.0[2], rhs.0[1], r0);
                (t0, r1) = mac(t1, k, $modulus.0[1], r1);
                (t2, r0) = mac(t2, self.0[2], rhs.0[2], r0);
                (t1, r1) = mac(t2, k, $modulus.0[2], r1);
                (t3, r0) = mac(t3, self.0[2], rhs.0[3], r0);
                (t2, r1) = mac(t3, k, $modulus.0[3], r1);
                t3 = r0 + r1;

                (t0, r0) = macx(t0, self.0[3], rhs.0[0]);
                k = t0.wrapping_mul($inv);
                (_, r1) = macx(t0, k, $modulus.0[0]);
                (t1, r0) = mac(t1, self.0[3], rhs.0[1], r0);
                (t0, r1) = mac(t1, k, $modulus.0[1], r1);
                (t2, r0) = mac(t2, self.0[3], rhs.0[2], r0);
                (t1, r1) = mac(t2, k, $modulus.0[2], r1);
                (t3, r0) = mac(t3, self.0[3], rhs.0[3], r0);
                (t2, r1) = mac(t3, k, $modulus.0[3], r1);
                t3 = r0 + r1;

                // Result may be within MODULUS of the correct value
                if !$field::is_less_than([t0, t1, t2, t3], $modulus.0) {
                    let mut borrow;
                    (t0, borrow) = t0.overflowing_sub($modulus.0[0]);
                    (t1, borrow) = sbb(t1, $modulus.0[1], borrow);
                    (t2, borrow) = sbb(t2, $modulus.0[2], borrow);
                    t3 = t3.wrapping_sub($modulus.0[3] + borrow as u64);
                }
                $field([t0, t1, t2, t3])

                //(&$field([t0, t1, t2, t3])).sub(&$modulus)
            }

            #[allow(clippy::too_many_arguments)]
            #[inline(always)]
            pub(crate) const fn montgomery_reduce(
                r0: u64,
                mut r1: u64,
                mut r2: u64,
                mut r3: u64,
                mut r4: u64,
                mut r5: u64,
                mut r6: u64,
                mut r7: u64,
            ) -> $field {
                // The Montgomery reduction here is based on Algorithm 14.32 in
                // Handbook of Applied Cryptography
                // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
                let mut k;
                let mut carry;
                let mut carry2;

                k = r0.wrapping_mul($inv);
                (_, carry) = macx(r0, k, $modulus.0[0]);
                (r1, carry) = mac(r1, k, $modulus.0[1], carry);
                (r2, carry) = mac(r2, k, $modulus.0[2], carry);
                (r3, carry) = mac(r3, k, $modulus.0[3], carry);
                (r4, carry2) = r4.overflowing_add(carry);

                k = r1.wrapping_mul($inv);
                (_, carry) = macx(r1, k, $modulus.0[0]);
                (r2, carry) = mac(r2, k, $modulus.0[1], carry);
                (r3, carry) = mac(r3, k, $modulus.0[2], carry);
                (r4, carry) = mac(r4, k, $modulus.0[3], carry);
                (r5, carry2) = adc(r5, carry, carry2);

                k = r2.wrapping_mul($inv);
                (_, carry) = macx(r2, k, $modulus.0[0]);
                (r3, carry) = mac(r3, k, $modulus.0[1], carry);
                (r4, carry) = mac(r4, k, $modulus.0[2], carry);
                (r5, carry) = mac(r5, k, $modulus.0[3], carry);
                (r6, carry2) = adc(r6, carry, carry2);

                k = r3.wrapping_mul($inv);
                (_, carry) = macx(r3, k, $modulus.0[0]);
                (r4, carry) = mac(r4, k, $modulus.0[1], carry);
                (r5, carry) = mac(r5, k, $modulus.0[2], carry);
                (r6, carry) = mac(r6, k, $modulus.0[3], carry);
                (r7, _) = adc(r7, carry, carry2);

                // Result may be within MODULUS of the correct value
                (&$field([r4, r5, r6, r7])).sub(&$modulus)
            }
        }
    };
    ($field:ident, $modulus:ident, $inv:ident, dense) => {
        impl $field {
            /// Adds `rhs` to `self`, returning the result.
            #[inline]
            pub const fn add(&self, rhs: &Self) -> Self {
                let (d0, carry) = self.0[0].overflowing_add(rhs.0[0]);
                let (d1, carry) = adc(self.0[1], rhs.0[1], carry);
                let (d2, carry) = adc(self.0[2], rhs.0[2], carry);
                let (d3, carry) = adc(self.0[3], rhs.0[3], carry);

                // Attempt to subtract the modulus, to ensure the value
                // is smaller than the modulus.
                let (s0, borrow) = d0.overflowing_sub($modulus.0[0]);
                let (s1, borrow) = sbb(d1, $modulus.0[1], borrow);
                let (s2, borrow) = sbb(d2, $modulus.0[2], borrow);
                let (s3, borrow) = sbb(d3, $modulus.0[3], borrow);

                // if underflow, then return original sum
                if !carry && borrow {
                    $field([d0, d1, d2, d3])
                } else {
                    // otherwise return difference
                    $field([s0, s1, s2, s3])
                }
            }

            #[inline(always)]
            pub const fn mul(&self, rhs: &Self) -> $field {
                $field::const_mul(self, rhs)
            }

            #[allow(clippy::too_many_arguments)]
            #[inline(always)]
            pub(crate) const fn montgomery_reduce(
                r0: u64,
                mut r1: u64,
                mut r2: u64,
                mut r3: u64,
                mut r4: u64,
                mut r5: u64,
                mut r6: u64,
                mut r7: u64,
            ) -> Self {
                // The Montgomery reduction here is based on Algorithm 14.32 in
                // Handbook of Applied Cryptography
                // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
                let mut k;
                let mut carry;
                let mut carry2;

                k = r0.wrapping_mul($inv);
                (_, carry) = macx(r0, k, $modulus.0[0]);
                (r1, carry) = mac(r1, k, $modulus.0[1], carry);
                (r2, carry) = mac(r2, k, $modulus.0[2], carry);
                (r3, carry) = mac(r3, k, $modulus.0[3], carry);
                (r4, carry2) = r4.overflowing_add(carry);

                k = r1.wrapping_mul($inv);
                (_, carry) = k.carrying_mul($modulus.0[0], r1);
                (r2, carry) = mac(r2, k, $modulus.0[1], carry);
                (r3, carry) = mac(r3, k, $modulus.0[2], carry);
                (r4, carry) = mac(r4, k, $modulus.0[3], carry);
                (r5, carry2) = adc(r5, carry, carry2);

                k = r2.wrapping_mul($inv);
                (_, carry) = macx(r2, k, $modulus.0[0]);
                (r3, carry) = mac(r3, k, $modulus.0[1], carry);
                (r4, carry) = mac(r4, k, $modulus.0[2], carry);
                (r5, carry) = mac(r5, k, $modulus.0[3], carry);
                (r6, carry2) = adc(r6, carry, carry2);

                k = r3.wrapping_mul($inv);
                (_, carry) = macx(r3, k, $modulus.0[0]);
                (r4, carry) = mac(r4, k, $modulus.0[1], carry);
                (r5, carry) = mac(r5, k, $modulus.0[2], carry);
                (r6, carry) = mac(r6, k, $modulus.0[3], carry);
                (r7, carry2) = adc(r7, carry, carry2);

                // Result may be within MODULUS of the correct value
                let mut borrow;
                (r4, borrow) = r4.overflowing_sub($modulus.0[0]);
                (r5, borrow) = sbb(r5, $modulus.0[1], borrow);
                (r6, borrow) = sbb(r6, $modulus.0[2], borrow);
                (r7, borrow) = sbb(r7, $modulus.0[3], borrow);

                if !carry2 && borrow {
                    (r4, borrow) = r4.overflowing_add($modulus.0[0]);
                    (r5, borrow) = adc(r5, $modulus.0[1], borrow);
                    (r6, borrow) = adc(r6, $modulus.0[2], borrow);
                    (r7, _) = adc(r7, $modulus.0[3], borrow);
                }
                $field([r4, r5, r6, r7])
            }
        }
    };
}
