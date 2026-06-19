#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    Bf16,
    Fp8E4M3,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Bf16 => 2,
            DType::Fp8E4M3 => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::Fp8E4M3 => "fp8e4m3",
        }
    }
}

pub trait Dtype: Copy + Clone + 'static {
    const SIZE_BYTES: usize;
    const NAME: &'static str;
    const DTYPE: DType;
    type Repr: Copy + Default;
}

macro_rules! define_dtype {
    ($marker:ident, $repr:ty, $bytes:expr, $name:literal, $tag:expr) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $marker;
        impl Dtype for $marker {
            const SIZE_BYTES: usize = $bytes;
            const NAME: &'static str = $name;
            const DTYPE: DType = $tag;
            type Repr = $repr;
        }
    };
}

define_dtype!(F32, f32, 4, "f32", DType::F32);
define_dtype!(F16, Half, 2, "f16", DType::F16);
define_dtype!(Bf16, BF16, 2, "bf16", DType::Bf16);
define_dtype!(Fp8E4M3, F8E4M3, 1, "fp8e4m3", DType::Fp8E4M3);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct BF16(pub u16);

impl BF16 {
    pub fn from_f32(x: f32) -> Self {
        let bits = x.to_bits();
        if (bits & 0x7fff_ffff) > 0x7f80_0000 {
            return BF16((bits >> 16) as u16 | 0x0040);
        }
        let rounding_bias = 0x7fff + ((bits >> 16) & 1);
        BF16(((bits + rounding_bias) >> 16) as u16)
    }

    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Half(pub u16);

impl Half {
    pub fn from_f32(x: f32) -> Self {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x007f_ffff;

        if exp == 0xff {
            let m = if mant != 0 { 0x0200 } else { 0 };
            return Half(sign | 0x7c00 | m);
        }
        let unbiased = exp - 127 + 15;
        if unbiased >= 0x1f {
            return Half(sign | 0x7c00);
        }
        if unbiased <= 0 {
            if unbiased < -10 {
                return Half(sign);
            }
            let mant = mant | 0x0080_0000;
            let shift = (14 - unbiased) as u32;
            let half_mant = (mant >> shift) as u16;
            let round_bit = (mant >> (shift - 1)) & 1;
            return Half(sign | (half_mant + round_bit as u16));
        }
        let half = sign | ((unbiased as u16) << 10) | ((mant >> 13) as u16);
        let round_bit = (mant >> 12) & 1;
        Half(half + round_bit as u16)
    }

    pub fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x03ff;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e = -1i32;
                let mut m = mant;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03ff;
                sign | (((127 - 15 + e) as u32) << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            sign | 0x7f80_0000 | (mant << 13)
        } else {
            sign | ((exp + (127 - 15)) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct F8E4M3(pub u8);

impl F8E4M3 {
    pub const MAX: f32 = 448.0;
    const EXP_BIAS: i32 = 7;

    pub fn to_f32(self) -> f32 {
        let b = self.0;
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((b >> 3) & 0x0f) as i32;
        let mant = (b & 0x07) as i32;

        if exp == 0x0f && mant == 0x07 {
            return f32::NAN;
        }
        if exp == 0 {
            return sign * (mant as f32) * 2f32.powi(1 - Self::EXP_BIAS - 3);
        }
        let frac = 1.0 + (mant as f32) / 8.0;
        sign * frac * 2f32.powi(exp - Self::EXP_BIAS)
    }

    pub fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return F8E4M3(0xff);
        }
        let sign_bit: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
        let ax = x.abs();
        if ax == 0.0 {
            return F8E4M3(sign_bit);
        }
        if ax >= Self::MAX {
            return F8E4M3(sign_bit | 0x7e);
        }

        let min_normal = 2f32.powi(1 - Self::EXP_BIAS);
        if ax < min_normal {
            let scale = 2f32.powi(1 - Self::EXP_BIAS - 3);
            let q = ax / scale;
            let mant = round_half_even(q).min(7.0) as u8;
            return F8E4M3(sign_bit | mant);
        }

        let mut exp = ax.log2().floor() as i32;
        if exp < 1 - Self::EXP_BIAS {
            exp = 1 - Self::EXP_BIAS;
        }
        let frac = ax / 2f32.powi(exp) - 1.0;
        let mut mant = round_half_even(frac * 8.0) as i32;
        let mut biased_exp = exp + Self::EXP_BIAS;
        if mant == 8 {
            mant = 0;
            biased_exp += 1;
        }
        if biased_exp >= 0x0f && mant >= 0x07 {
            return F8E4M3(sign_bit | 0x7e);
        }
        F8E4M3(sign_bit | ((biased_exp as u8) << 3) | (mant as u8))
    }
}

fn round_half_even(x: f32) -> f32 {
    let r = x.round();
    if (x - x.floor() - 0.5).abs() < f32::EPSILON {
        let down = x.floor();
        if (down as i64) % 2 == 0 {
            down
        } else {
            down + 1.0
        }
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(F32::SIZE_BYTES, 4);
        assert_eq!(Bf16::SIZE_BYTES, 2);
        assert_eq!(F16::SIZE_BYTES, 2);
        assert_eq!(Fp8E4M3::SIZE_BYTES, 1);
        assert_eq!(DType::Fp8E4M3.size_bytes(), 1);
    }

    #[test]
    fn bf16_roundtrip() {
        for &v in &[0.0f32, 1.0, -2.5, 3.1415927, 1e6, -1e-6] {
            let r = BF16::from_f32(v).to_f32();
            let tol = v.abs() * 0.01 + 1e-6;
            assert!((r - v).abs() <= tol, "bf16 {v} -> {r}");
        }
    }

    #[test]
    fn half_roundtrip() {
        for &v in &[0.0f32, 1.0, -2.5, 0.5, 100.0, -0.001] {
            let r = Half::from_f32(v).to_f32();
            let tol = v.abs() * 0.001 + 1e-4;
            assert!((r - v).abs() <= tol, "f16 {v} -> {r}");
        }
    }

    #[test]
    fn fp8_exact_values() {
        for &v in &[1.0f32, 2.0, 0.5, 4.0, -8.0, 256.0] {
            let r = F8E4M3::from_f32(v).to_f32();
            assert_eq!(r, v, "fp8 exact {v} -> {r}");
        }
    }

    #[test]
    fn fp8_saturates_and_nan() {
        assert_eq!(F8E4M3::from_f32(1e9).to_f32(), 448.0);
        assert_eq!(F8E4M3::from_f32(-1e9).to_f32(), -448.0);
        assert!(F8E4M3::from_f32(f32::NAN).to_f32().is_nan());
        assert_eq!(F8E4M3::from_f32(0.0).to_f32(), 0.0);
    }

    #[test]
    fn fp8_rounds_close() {
        for &v in &[3.0f32, 5.0, 100.0, 0.1, -0.3, 12.5] {
            let r = F8E4M3::from_f32(v).to_f32();
            let tol = v.abs() * 0.13 + 0.02;
            assert!((r - v).abs() <= tol, "fp8 {v} -> {r}");
        }
    }
}
