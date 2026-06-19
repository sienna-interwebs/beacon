pub trait Shape: Copy + 'static {
    const RANK: usize;
    const NUMEL: usize;
    const DIMS: &'static [usize];
}

#[derive(Clone, Copy, Debug, Default)]
pub struct S1<const A: usize>;
#[derive(Clone, Copy, Debug, Default)]
pub struct S2<const A: usize, const B: usize>;
#[derive(Clone, Copy, Debug, Default)]
pub struct S3<const A: usize, const B: usize, const C: usize>;
#[derive(Clone, Copy, Debug, Default)]
pub struct S4<const A: usize, const B: usize, const C: usize, const D: usize>;

impl<const A: usize> Shape for S1<A> {
    const RANK: usize = 1;
    const NUMEL: usize = A;
    const DIMS: &'static [usize] = &[A];
}

impl<const A: usize, const B: usize> Shape for S2<A, B> {
    const RANK: usize = 2;
    const NUMEL: usize = A * B;
    const DIMS: &'static [usize] = &[A, B];
}

impl<const A: usize, const B: usize, const C: usize> Shape for S3<A, B, C> {
    const RANK: usize = 3;
    const NUMEL: usize = A * B * C;
    const DIMS: &'static [usize] = &[A, B, C];
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> Shape for S4<A, B, C, D> {
    const RANK: usize = 4;
    const NUMEL: usize = A * B * C * D;
    const DIMS: &'static [usize] = &[A, B, C, D];
}

pub trait MatmulWith<Rhs: Shape>: Shape {
    type Output: Shape;
}

impl<const M: usize, const K: usize, const N: usize> MatmulWith<S2<K, N>> for S2<M, K> {
    type Output = S2<M, N>;
}

impl<const B: usize, const M: usize, const K: usize, const N: usize> MatmulWith<S2<K, N>>
    for S3<B, M, K>
{
    type Output = S3<B, M, N>;
}

impl<const B: usize, const M: usize, const K: usize, const N: usize> MatmulWith<S3<B, K, N>>
    for S3<B, M, K>
{
    type Output = S3<B, M, N>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_and_dims() {
        assert_eq!(<S1<50257> as Shape>::NUMEL, 50257);
        assert_eq!(<S2<768, 3072> as Shape>::NUMEL, 768 * 3072);
        assert_eq!(<S4<4, 12, 512, 64> as Shape>::NUMEL, 4 * 12 * 512 * 64);
        assert_eq!(<S2<768, 3072> as Shape>::DIMS, &[768, 3072]);
        assert_eq!(<S3<4, 512, 768> as Shape>::RANK, 3);
    }

    #[test]
    fn matmul_output_shape() {
        type Lhs = S2<2048, 768>;
        type Rhs = S2<768, 3072>;
        type Out = <Lhs as MatmulWith<Rhs>>::Output;
        assert_eq!(<Out as Shape>::DIMS, &[2048, 3072]);
    }
}
