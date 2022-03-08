mod tests;

/*
 * Primitive polynomials - see Lin & Costello, Appendix A,
 * and  Lee & Messerschmitt, p. 453.
 */
const PP: &[u8; 9] = b"101110001";

/* To make sure that we stay within cache in the inner loops of fec_encode().  (It would
probably help to also do this for fec_decode().*/
const STRIDE: usize = 8192;

static UNROLL: usize = 16; /* 1, 4, 8, 16 */
// TODO: Implement unrolling
// could be done at build time. Run some basic unrolling tests
// in the build.rs, whichever unrolling is fastest, have a macro
// that does the unrolling

//const FEC_MAGIC: u32 = 0xFECC0DEC;

#[derive(Debug)]
pub enum Error {
    ZeroK,
    ZeroM,
    BigN,
    KGtN,
    Tbd,
}

type Gf = u8;
//type Matrix<T> = Vec<Vec<T>>;
type Result<Fec> = std::result::Result<Fec, Error>;

/*
 * To speed up computations, we have tables for logarithm, exponent and
 * inverse of a number.  We use a table for multiplication as well (it takes
 * 64K, no big deal even on a PDA, especially because it can be
 * pre-initialized an put into a ROM!), otherwhise we use a table of
 * logarithms. In any case the macro gf_mul(x,y) takes care of
 * multiplications.
 */
struct Statics {
    gf_exp: [Gf; 510],
    gf_log: [i32; 256],
    inverse: [Gf; 256],
    gf_mul_table: [[Gf; 256]; 256],
}
impl Statics {
    pub fn new() -> Self {
        let mut gf_exp: [Gf; 510] = [0; 510];
        let mut gf_log: [i32; 256] = [0; 256];
        let mut inverse: [Gf; 256] = [0; 256];
        let mut gf_mul_table: [[Gf; 256]; 256] = [[0; 256]; 256];
        Self::generate_gf(&mut gf_exp, &mut gf_log, &mut inverse);
        Self::_init_mul_table(&mut gf_mul_table, &mut gf_exp, &mut gf_log);
        Self {
            gf_exp: gf_exp,
            gf_log: gf_log,
            inverse: inverse,
            gf_mul_table: gf_mul_table,
        }
    }
    /// Initialize the data structures used for computations in GF
    fn generate_gf(gf_exp: &mut [Gf; 510], gf_log: &mut [i32; 256], inverse: &mut [Gf; 256]) {
        let mut mask: Gf;

        mask = 1; /* x ** 0 = 1 */
        gf_exp[8] = 0; /* will be updated at the end of the 1st loop */
        /*
         * first, generate the (polynomial representation of) powers of \alpha,
         * which are stored in gf_exp[i] = \alpha ** i .
         * At the same time build gf_log[gf_exp[i]] = i .
         * The first 8 powers are simply bits shifted to the left.
         */
        for i in 0..8 {
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
        for i in 9..255 {
            if gf_exp[i - 1] >= mask {
                gf_exp[i] = gf_exp[8] ^ ((gf_exp[i - 1] ^ mask) << 1);
            } else {
                gf_exp[i] = gf_exp[i - 1] << 1;
            }
            gf_log[gf_exp[i] as usize] = i as i32;
        }
        /*
         * log(0) is not defined, so use a special value
         */
        gf_log[0] = 255;
        /* set the extended gf_exp values for fast multiply */
        for i in 0..255 {
            gf_exp[i + 255] = gf_exp[i];
        }
        /*
         * again special cases. 0 has no inverse. This used to
         * be initialized to 255, but it should make no difference
         * since noone is supposed to read from here.
         */
        inverse[0] = 0;
        inverse[1] = 1;
        for i in 2..=255 {
            inverse[i] = gf_exp[255 - gf_log[i] as usize];
        }
    }
    fn _init_mul_table(
        gf_mul_table: &mut [[Gf; 256]; 256],
        gf_exp: &[Gf; 510],
        gf_log: &[i32; 256],
    ) {
        for i in 0..256 {
            for j in 0..256 {
                gf_mul_table[i][j] = gf_exp[Self::modnn(gf_log[i] + gf_log[j]) as usize];
            }
        }
        for j in 0..256 {
            gf_mul_table[j][0] = 0;
            gf_mul_table[0][j] = 0;
        }
    }
    fn modnn(mut x: i32) -> Gf {
        while x >= 255 {
            x -= 255;
            x = (x >> 8) + (x & 255);
        }
        x as Gf
    }
    pub fn addmul(&self, dst: &mut [Gf], src: &[Gf], c: Gf, sz: usize) {
        eprintln!("c: {:02x}, sz: {}", c, sz);
        eprintln!("dst: {:02x?}", dst);
        eprintln!("src: {:02x?}", src);
        if c != 0 {
            self._addmul1(dst, src, c, sz);
        }
    }
    fn _addmul1(&self, dst: &mut [Gf], src: &[Gf], c: Gf, sz: usize) {
        if src.len() > 0 {
            let mulc = self.gf_mul_table[c as usize];
            //let lim = &dst[sz - UNROLL + 1..];
            // they unroll, for now I'll just do it directly
            for i in 0..sz {
                dst[i] ^= mulc[src[i] as usize];
            }
            eprintln!("dst: {:02x?}", dst);
        }
    }
    /*
     * computes C = AB where A is n*k, B is k*m, C is n*m
     */
    fn _matmul(&self, a: &[Gf], b: &[Gf], c: &mut [Gf], n: usize, k: usize, m: usize) {
        eprintln!("a: {:02x?}", a);
        eprintln!("b: {:02x?}", b);
        eprintln!("c: {:02x?}", c);
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
        eprintln!("c: {:02x?}", c);
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
    fn _invert_vdm(&self, src: &mut Vec<Gf>, k: usize) {
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
        for i in 1..k {
            let p_i = p[i]; /* see above comment */
            for j in (k - 1 - (i - 1))..(k - 1) {
                c[j] ^= self.gf_mul(p_i, c[j + 1]);
            }
            c[k - 1] ^= p_i;
        }

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

        // unnecessary for Rust
        // free(c);
        // free(b);
        // free(p);

        return;
    }
    fn gf_mul(&self, x: Gf, y: Gf) -> Gf {
        self.gf_mul_table[x as usize][y as usize]
    }
    fn _invert_mat(&self, src: &mut [Gf], k: usize) {
        let mut c: Gf;
        let (mut irow, mut icol) = (0, 0);

        let mut indxc = vec![0; k];
        let mut indxr = vec![0; k];
        let mut ipiv = vec![0; k];
        let mut id_row = vec![0; k];

        /*
         * ipiv marks elements already used as pivots.
         */
        for i in 0..k {
            ipiv[i] = 0;
        }

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
                    let tmp = src[irow * k + ix];
                    src[irow * k + ix] = src[icol * k + ix];
                    src[icol * k + ix] = tmp;
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
                for ix in 0..k {
                    pivot_row[ix] = self.gf_mul(c, pivot_row[ix]);
                }
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
                        eprintln!("Loc 1");
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
                    let tmp = src[row * k + indxr[col - 1]];
                    src[row * k + indxr[col - 1]] = src[row * k + indxc[col - 1]];
                    src[row * k + indxc[col - 1]] = tmp;
                }
            }
        }
    }
}

pub struct Fec {
    // magic: u64, // I'm not sure what magic does. It's never used except in new and free. My guess is it's some kind of way to make sure you're freeing the right memory?
    k: usize,
    m: usize,
    enc_matrix: Vec<Gf>,

    /*
     * To speed up computations, we have tables for logarithm, exponent and
     * inverse of a number.  We use a table for multiplication as well (it takes
     * 64K, no big deal even on a PDA, especially because it can be
     * pre-initialized an put into a ROM!), otherwhise we use a table of
     * logarithms. In any case the macro gf_mul(x,y) takes care of
     * multiplications.
     */
    // gf_exp: [Gf; 510],
    // gf_log: [u32; 256],
    // inverse: [Gf; 256],
    // gf_mul_table: [[Gf; 256]; 256],
    statics: Statics,
}
impl Fec {
    /*
     * This section contains the proper FEC encoding/decoding routines.
     * The encoding matrix is computed starting with a Vandermonde matrix,
     * and then transforming it into a systematic matrix.
     */
    /*
     * param k the number of blocks required to reconstruct
     * param m the total number of blocks created
     */
    pub fn new(k: usize, n: usize) -> Result<Fec> {
        //  eprintln!("Creating new - k: {}, n: {}", k, n);
        if k < 1 {
            return Err(Error::ZeroK);
        }
        if n < 1 {
            return Err(Error::ZeroM);
        }
        if n > 256 {
            return Err(Error::BigN);
        }
        if k > n {
            return Err(Error::KGtN);
        }
        let mut tmp_m: Vec<Gf> = vec![0; n * k];

        // let mut gf_exp = [0; 510];
        // let mut gf_log = [0; 256];
        // let mut inverse = [0; 256];
        // let mut gf_mul_table = [[0; 256]; 256];

        // Self::init_fec(&mut gf_mul_table, &mut gf_exp, &mut gf_log, &mut inverse);
        let statics = Statics::new();

        // m rows by k columns
        let mut enc_matrix: Vec<Gf> = vec![0; n * k];

        let mut ret_val = Fec {
            k: k,
            m: n,
            enc_matrix: vec![], // needs to be added in below
            // gf_exp: gf_exp,
            // gf_log: gf_log,
            // inverse: inverse,
            // // magic: (((FEC_MAGIC ^ k as u32) ^ m as u32) ^ (enc_matrix)) as u64,
            // gf_mul_table: gf_mul_table,
            statics: statics,
        };

        /*
         * fill the matrix with powers of field elements, starting from 0.
         * The first row is special, cannot be computed with exp. table.
         */
        tmp_m[0] = 1;
        for col in 1..k {
            tmp_m[col] = 0;
        }
        for row in 0..(n - 1) {
            //  eprintln!("row: {}", row);
            let p: &mut [u8] = &mut tmp_m[(row + 1) * k..];
            for col in 0..k {
                p[col] = ret_val.statics.gf_exp[Statics::modnn((row * col) as i32) as usize];
            }
        }

        /*
         * quick code to build systematic matrix: invert the top
         * k*k vandermonde matrix, multiply right the bottom n-k rows
         * by the inverse, and construct the identity matrix at the top.
         */
        eprintln!("tmp_m: {:02x?}", tmp_m);
        ret_val.statics._invert_vdm(&mut tmp_m, k); /* much faster than _invert_mat */
        ret_val.statics._matmul(
            &tmp_m[k * k..],
            &tmp_m[..],
            &mut enc_matrix[k * k..],
            n - k,
            k,
            k,
        );
        /*
         * the upper matrix is I so do not bother with a slow multiply
         */
        // the Vec is initialized to 0's when defined
        // memset(retval->enc_matrix, '\0', k * k * sizeof(gf));
        for i in 0..k {
            //  eprintln!("i: {}", i);
            enc_matrix[i * (k + 1)] = 1;
        }

        // unnecessary in Rust, tmp_m gets dropped
        // free(tmp_m);

        ret_val.enc_matrix = enc_matrix;

        Ok(ret_val)
    }
    fn chunk_size(&self, data_len: usize) -> usize {
        (data_len as f64 / self.k as f64).ceil() as usize
    }
    fn flatten(square: &mut Vec<Vec<u8>>) -> Vec<u8> {
        let mut ret_vec = vec![];
        for chunk in square {
            ret_vec.append(chunk);
        }
        ret_vec
    }
    // returns Vec of chunks
    pub fn encode(&mut self, data: &Vec<u8>) -> Result<Vec<Vec<u8>>> {
        // eprintln!("\nEncoding k: {}, m: {}", self.k, self.m);
        // clean side
        let chunk_size = self.chunk_size(data.len());
        // eprintln!("chunk_size: {}", chunk_size);
        let data_slice = &data[..];

        let mut chunks = vec![];

        // eprintln!("data: {:02x?}", data);
        // eprintln!("data len: {:02x?}", data.len());

        for i in 0..self.k {
            let mut temp_vec = vec![];
            if (i * chunk_size) >= data_slice.len() {
                eprintln!("empty chunk");
                temp_vec.append(&mut vec![0; chunk_size].to_vec());
            } else if ((i * chunk_size) < data_slice.len())
                && (((i + 1) * chunk_size) > data_slice.len())
            {
                // finish current chunk
                temp_vec.append(&mut data_slice[i * chunk_size..].to_vec());
                // add padding
                let remaining = ((i + 1) * chunk_size) as usize - data_slice.len();
                // eprint!("final slice, padding");
                for _ in 0..remaining {
                    eprint!(".");
                    temp_vec.push(0);
                }
            } else {
                let new_chunk =
                    &data_slice[(i * chunk_size) as usize..((i + 1) * chunk_size) as usize];
                // eprintln!("normal chunk: {:02x?}", new_chunk);
                temp_vec.append(&mut new_chunk.to_vec())
            }
            chunks.push(temp_vec);
        }
        // eprintln!("Finished chunking");

        let num_check_blocks_produced = self.m - self.k;
        let mut check_blocks_produced = vec![vec![0; chunk_size]; num_check_blocks_produced];
        let check_block_ids: Vec<usize> = (self.k..self.m).map(|x| x as usize).collect();
        eprintln!("num: {}", num_check_blocks_produced);
        eprintln!("blocks: {:?}", check_blocks_produced);
        eprintln!("ids: {:?}", check_block_ids);

        ///////// internals

        let mut k = 0;
        while k < chunk_size {
            let stride = if (chunk_size - k) < STRIDE {
                chunk_size - k
            } else {
                STRIDE
            };
            for i in 0..num_check_blocks_produced {
                let fecnum = check_block_ids[i];
                if fecnum < self.k {
                    return Err(Error::Tbd);
                }
                let p = &self.enc_matrix[fecnum as usize * self.k..];
                eprintln!("enc_matrix: {:02x?}", &self.enc_matrix);
                eprintln!("p: {:02x?}", p);
                for j in 0..self.k {
                    eprintln!("Loc 2");
                    self.statics.addmul(
                        &mut check_blocks_produced[i][k..],
                        &chunks[j][k..k + stride],
                        p[j],
                        stride,
                    );
                }
            }

            k += STRIDE;
        }

        ///////// internals

        let mut ret_chunks = vec![];
        ret_chunks.append(&mut chunks);
        ret_chunks.append(&mut check_blocks_produced);
        // eprintln!("ret_chunks: {:02x?}", ret_chunks);
        Ok(ret_chunks)
    }
    fn build_decode_matrix_into_space(&self, index: &[usize], k: usize, matrix: &mut [Gf]) {
        for i in 0..k {
            let p = &mut matrix[i * k..];
            if index[i] < k {
                // we'll assume it's already 0
                // memset(p, 0, k);
                p[i] = 1;
            } else {
                // memcpy(p, &(code->enc_matrix[index[i] * code->k]), k);
                for j in 0..k {
                    p[j] = self.enc_matrix[(index[i] * self.k) + j];
                }
            }
        }
        self.statics._invert_mat(matrix, k);
    }
    // takes the data with it's block index, and how much padding there is after the message
    pub fn decode(&mut self, encoded_data: &[(usize, Vec<u8>)], padding: usize) -> Vec<u8> {
        // eprintln!("\nDecoding");

        let mut share_nums: Vec<usize> = vec![];
        let mut chunks: Vec<Vec<u8>> = vec![vec![]; self.m];

        for (num, chunk) in encoded_data {
            share_nums.push(num.clone());
            chunks[*num] = chunk.to_vec();
            // chunks.insert(*num, chunk.to_vec());
        }
        eprintln!("encoded data: {:02x?}", encoded_data);
        eprintln!("share_nums: {:02x?}", share_nums);
        eprintln!("chunks: {:02x?}", chunks);

        let sz = chunks[share_nums[0] as usize].len();
        let mut ret_chunks = vec![vec![0; sz]; self.k];

        let mut complete = true;
        let mut missing = std::collections::VecDeque::new();
        let mut replaced = vec![];
        // check which of the original chunks are missing
        for i in 0..self.k {
            if !share_nums.contains(&i) {
                complete = false;
                missing.push_back(i);
                eprintln!("Missing {}", i);
            }
        }

        // replace the missing chunks with fec chunks
        for i in self.k..self.m {
            if chunks[i].len() != 0 {
                match missing.pop_front() {
                    Some(index) => {
                        eprintln!("Moving {} to {}", i, index);
                        replaced.push(index);
                        share_nums.insert(index, i);
                        chunks[index] = chunks[i].to_vec();
                        eprintln!("share_nums: {:02x?}", share_nums);
                        eprintln!("chunks: {:02x?}", chunks);
                    }
                    None => {}
                }
            }
        }

        if complete {
            let flat = Self::flatten(&mut chunks[..self.k].to_vec());
            return flat[..flat.len() - padding].to_vec();
        }

        /////////////// internal decode

        let mut m_dec = vec![0; self.k * self.k];
        let mut outix = 0;

        self.build_decode_matrix_into_space(&share_nums, self.k, &mut m_dec[..]);

        for row in 0..self.k {
            assert!((share_nums[row] >= self.k) || (share_nums[row] == row));
            if share_nums[row] >= self.k {
                // if it's not a normal block
                // memset(outpkts[outix], 0, sz);
                for i in 0..sz {
                    ret_chunks[outix][i] = 0;
                }
                for col in 0..self.k {
                    eprintln!("Loc 2");
                    self.statics.addmul(
                        &mut ret_chunks[outix][..],
                        &chunks[col][..],
                        m_dec[row * self.k + col],
                        sz,
                    );
                }
                outix += 1;
            }
        }

        /////////////// end internal decode

        eprintln!("replaced: {:02x?}", replaced);
        eprintln!("ret_chunks: {:02x?}", ret_chunks);
        // fix the replaced chunks
        for i in 0..replaced.len() {
            chunks[replaced[i]] = ret_chunks[i].to_vec();
            eprintln!("chunks: {:02x?}", chunks);
        }
        let ret_vec = Self::flatten(&mut chunks[0..self.k].to_vec());

        // remove padding
        ret_vec[..ret_vec.len() - padding].to_vec()
    }
}
