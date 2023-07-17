use crate::Gf;

/*
 * Primitive polynomials - see Lin & Costello, Appendix A,
 * and  Lee & Messerschmitt, p. 453.
 */
const PP: &[u8; 9] = b"101110001";

// made this const, cuz she big and unchanging
pub const STATICS: Statics = Statics::new();

/*
 * To speed up computations, we have tables for logarithm, exponent and
 * inverse of a number.  We use a table for multiplication as well (it takes
 * 64K, no big deal even on a PDA, especially because it can be
 * pre-initialized an put into a ROM!), otherwhise we use a table of
 * logarithms. In any case the macro gf_mul(x,y) takes care of
 * multiplications.
 */
pub struct Statics {
    pub gf_exp: [Gf; 510],
    //gf_log: [i32; 256],
    inverse: [Gf; 256],
    gf_mul_table: [[Gf; 256]; 256],
}
impl Statics {
    pub const fn new() -> Self {
        let (gf_exp, gf_log, inverse) = Self::generate_gf();
        let gf_mul_table = Self::_init_mul_table(&gf_exp, &gf_log);
        Self {
            gf_exp,
            //gf_log,
            inverse,
            gf_mul_table,
        }
    }
    /// Initialize the data structures used for computations in GF
    const fn generate_gf() -> ([Gf; 510], [i32; 256], [Gf; 256]) {
        let mut gf_exp: [Gf; 510] = [0; 510];
        // only used in initializing other fields
        let mut gf_log: [i32; 256] = [0; 256];
        let mut inverse: [Gf; 256] = [0; 256];
        let mut mask: Gf;

        mask = 1; /* x ** 0 = 1 */
        gf_exp[8] = 0; /* will be updated at the end of the 1st loop */
        /*
         * first, generate the (polynomial representation of) powers of \alpha,
         * which are stored in gf_exp[i] = \alpha ** i .
         * At the same time build gf_log[gf_exp[i]] = i .
         * The first 8 powers are simply bits shifted to the left.
         */

        // for i in 0..8 {
        let mut i = 0;
        while i < 8 {
            gf_exp[i] = mask;
            gf_log[gf_exp[i] as usize] = i as i32;
            /*
             * If Pp[i] == 1 then \alpha ** i occurs in poly-repr
             * gf_exp[8] = \alpha ** 8
             */
            if PP[i] == b'1' {
                gf_exp[8] ^= mask;
            }
            mask <<= 1;
            i += 1;
        }
        /*
         * now gf_exp[8] = \alpha ** 8 is complete, so can also
         * compute its inverse.
         */
        gf_log[gf_exp[8] as usize] = 8;

        /*
         * Poly-repr of \alpha ** (i+1) is given by poly-repr of
         * \alpha ** i shifted left one-bit and accounting for any
         * \alpha ** 8 term that may occur when poly-repr of
         * \alpha ** i is shifted.
         */
        mask = 1 << 7;

        // for i in 9..255 {
        // increment so it begins at 9 this time
        i += 1;
        while i < 255 {
            if gf_exp[i - 1] >= mask {
                gf_exp[i] = gf_exp[8] ^ ((gf_exp[i - 1] ^ mask) << 1);
            } else {
                gf_exp[i] = gf_exp[i - 1] << 1;
            }
            gf_log[gf_exp[i] as usize] = i as i32;
            i += 1;
        }
        /*
         * log(0) is not defined, so use a special value
         */
        gf_log[0] = 255;
        /* set the extended gf_exp values for fast multiply */

        // for i in 0..255 {
        i = 0;
        while i < 255 {
            gf_exp[i + 255] = gf_exp[i];
            i += 1;
        }
        /*
         * again special cases. 0 has no inverse. This used to
         * be initialized to 255, but it should make no difference
         * since noone is supposed to read from here.
         */
        inverse[0] = 0;
        inverse[1] = 1;

        // for i in 2..=255 {
        i = 2;
        while i <= 255 {
            inverse[i] = gf_exp[255 - gf_log[i] as usize];
            i += 1;
        }

        (gf_exp, gf_log, inverse)
    }
    const fn _init_mul_table(gf_exp: &[Gf; 510], gf_log: &[i32; 256]) -> [[Gf; 256]; 256] {
        let mut gf_mul_table: [[Gf; 256]; 256] = [[0; 256]; 256];
        // for i in 0..256 {
        let mut i = 0;
        while i < 256 {
            let mut j = 0;
            // for j in 0..256 {
            while j < 256 {
                gf_mul_table[i][j] = gf_exp[Self::modnn(gf_log[i] + gf_log[j]) as usize];
                j += 1;
            }
            i += 1;
        }
        // for j in 0..256 {
        let mut j = 0;
        while j < 256 {
            gf_mul_table[j][0] = 0;
            gf_mul_table[0][j] = 0;
            j += 1;
        }
        gf_mul_table
    }
    pub const fn modnn(mut x: i32) -> Gf {
        while x >= 255 {
            x -= 255;
            x = (x >> 8) + (x & 255);
        }
        x as Gf
    }
    pub fn addmul(&self, dst: &mut [Gf], src: &[Gf], c: Gf, sz: usize) {
        // eprintln!("c: {:02x}, sz: {}", c, sz);
        // eprintln!("dst: {:02x?}", dst);
        // eprintln!("src: {:02x?}", src);
        if c != 0 {
            self._addmul1(dst, src, c, sz);
        }
    }
    fn _addmul1(&self, dst: &mut [Gf], src: &[Gf], c: Gf, sz: usize) {
        if !src.is_empty() {
            let mulc = self.gf_mul_table[c as usize];
            //let lim = &dst[sz - UNROLL + 1..];
            // they unroll, for now I'll just do it directly
            for i in 0..sz {
                dst[i] ^= mulc[src[i] as usize];
            }
            // eprintln!("dst: {:02x?}", dst);
        }
    }
    /*
     * computes C = AB where A is n*k, B is k*m, C is n*m
     */
    pub fn matmul(&self, a: &[Gf], b: &[Gf], c: &mut [Gf], n: usize, k: usize, m: usize) {
        // eprintln!("a: {:02x?}", a);
        // eprintln!("b: {:02x?}", b);
        // eprintln!("c: {:02x?}", c);
        for row in 0..n {
            for col in 0..m {
                let mut acc: Gf = 0;
                for i in 0..k {
                    let pa: Gf = a[(row * k) + i];
                    let pb: Gf = b[col + (i * m)];
                    acc ^= self.gf_mul(pa, pb);
                }
                c[row * m + col] = acc;
            }
        }
        // eprintln!("c: {:02x?}", c);
    }
    /*
     * fast code for inverting a vandermonde matrix.
     *
     * NOTE: It assumes that the matrix is not singular and _IS_ a vandermonde
     * matrix. Only uses the second column of the matrix, containing the p_i's.
     *
     * Algorithm borrowed from "Numerical recipes in C" -- sec.2.8, but largely
     * revised for my purposes.
     * p = coefficients of the matrix (p_i)
     * q = values of the polynomial (known)
     */
    pub fn invert_vdm(&self, src: &mut [Gf], k: usize) {
        /*
         * b holds the coefficient for the matrix inversion
         * c holds the coefficient of P(x) = Prod (x - p_i), i=0..k-1
         */
        let (mut b, mut c, mut p): (Vec<Gf>, Vec<Gf>, Vec<Gf>) =
            (vec![0; k], vec![0; k], vec![0; k]);
        let (mut t, mut xx): (Gf, Gf);

        /* degenerate case, matrix must be p^0 = 1 */
        if k == 1 {
            return;
        }
        let mut j = 1;
        for i in 0..k {
            c[i] = 0;
            p[i] = src[j];
            j += k;
        }

        /*
         * construct coeffs. recursively. We know c[k] = 1 (implicit)
         * and start P_0 = x - p_0, then at each stage multiply by
         * x - p_i generating P_i = x P_{i-1} - p_i P_{i-1}
         * After k steps we are done.
         */
        c[k - 1] = p[0]; /* really -p(0), but x = -x in GF(2^m) */
        (1..k).for_each(|i| {
            let p_i = p[i]; /* see above comment */
            for j in (k - 1 - (i - 1))..(k - 1) {
                c[j] ^= self.gf_mul(p_i, c[j + 1]);
            }
            c[k - 1] ^= p_i;
        });

        for row in 0..k {
            /*
             * synthetic division etc.
             */
            xx = p[row];
            t = 1;
            b[k - 1] = 1; /* this is in fact c[k] */
            for i in (1..=(k - 1)).rev() {
                b[i - 1] = c[i] ^ self.gf_mul(xx, b[i]);
                t = self.gf_mul(xx, t) ^ b[i - 1];
            }
            for col in 0..k {
                src[col * k + row] = self.gf_mul(self.inverse[t as usize], b[col]);
            }
        }
    }
    fn gf_mul(&self, x: Gf, y: Gf) -> Gf {
        self.gf_mul_table[x as usize][y as usize]
    }
    pub fn _invert_mat(&self, src: &mut [Gf], k: usize) {
        let mut c: Gf;
        let (mut irow, mut icol) = (0, 0);

        let mut indxc = vec![0; k];
        let mut indxr = vec![0; k];
        let mut ipiv = vec![0; k];
        let mut id_row = vec![0; k];

        /*
         * ipiv marks elements already used as pivots.
         */
        (0..k).for_each(|i| {
            ipiv[i] = 0;
        });

        for col in 0..k {
            let mut piv_found: bool = false;

            /*
             * Zeroing column 'col', look for a non-zero element.
             * First try on the diagonal, if it fails, look elsewhere.
             */
            if ipiv[col] != 1 && src[col * k + col] != 0 {
                irow = col;
                icol = col;
                // goto found_piv;
            }
            for row in 0..k {
                if ipiv[row] != 1 {
                    for ix in 0..k {
                        if ipiv[ix] == 0 {
                            if src[row * k + ix] != 0 {
                                irow = row;
                                icol = ix;
                                // goto found_piv;
                                piv_found = true;
                            }
                        } else {
                            assert!(ipiv[ix] <= 1);
                        }
                        if piv_found {
                            break;
                        }
                    }
                }
                if piv_found {
                    break;
                }
            }

            // found_piv:
            ipiv[icol] += 1;
            /*
             * swap rows irow and icol, so afterwards the diagonal
             * element will be correct. Rarely done, not worth
             * optimizing.
             */
            if irow != icol {
                for ix in 0..k {
                    // direct implementation is easiest solution for the "SWAP" macro
                    src.swap(irow * k + ix, icol * k + ix);
                }
            }
            indxr[col] = irow;
            indxc[col] = icol;
            let pivot_row = &mut src[icol * k..(icol + 1) * k];
            c = pivot_row[icol];
            assert!(c != 0);
            if c != 1 {
                /* otherwhise this is a NOP */
                /*
                 * this is done often , but optimizing is not so
                 * fruitful, at least in the obvious ways (unrolling)
                 */
                c = self.inverse[c as usize];
                pivot_row[icol] = 1;
                (0..k).for_each(|ix| {
                    pivot_row[ix] = self.gf_mul(c, pivot_row[ix]);
                });
            }
            /*
             * from all rows, remove multiples of the selected row
             * to zero the relevant entry (in fact, the entry is not zero
             * because we know it must be zero).
             * (Here, if we know that the pivot_row is the identity,
             * we can optimize the addmul).
             */
            id_row[icol] = 1;
            if pivot_row != id_row {
                // create a copy of pivot row, since we can't
                // have mut and immut references at the same time
                // if we know what size it'll be, might as well
                // start it there and save realloc time
                let mut pivot_clone = vec![0; pivot_row.len()];
                for val in pivot_row.iter() {
                    pivot_clone.push(*val);
                }

                for ix in 0..k {
                    let p = &mut src[ix * k..(ix + 1) * k];
                    if ix != icol {
                        c = p[icol];
                        p[icol] = 0;
                        // eprintln!("Loc 1");
                        self.addmul(p, &pivot_clone[k..], c, k);
                    }
                }
            }
            id_row[icol] = 0;
        } /* done all columns */
        for col in (1..=k).rev() {
            if indxr[col - 1] != indxc[col - 1] {
                for row in 0..k {
                    // direct implementation is easiest solution for the "SWAP" macro
                    src.swap(row * k + indxr[col - 1], row * k + indxc[col - 1]);
                }
            }
        }
    }
}
