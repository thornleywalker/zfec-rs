/* Copyright (C) 2022, Walker Thornley
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

//! A pure Rust implementation of the Zfec library.
//!
//! The general concept of Zfec is to break a message into blocks, or "chunks", then generate additional chunks
//! with parity information that can be used to identify any missing chunks.
//!
//! Notice: Zfec only provides Forward Error Correcting functionality, not encryption. Any message coded with
//! Zfec should be encrypted first, if security is necessary.
//!
//! Implemented directly from https://github.com/tahoe-lafs/zfec

#[cfg(test)]
mod tests;

mod statics;

use std::fmt;

use statics::Statics;

/* To make sure that we stay within cache in the inner loops of fec_encode().  (It would
probably help to also do this for fec_decode().*/
const STRIDE: usize = 8192;

//static UNROLL: usize = 16; /* 1, 4, 8, 16 */
// TODO: Implement unrolling
// could be done at build time. Run some basic unrolling tests
// in the build.rs, whichever unrolling is fastest, have a macro
// that does the unrolling

//const FEC_MAGIC: u32 = 0xFECC0DEC;

#[derive(Debug)]
/// Possible errors
pub enum Error {
    ZeroK,
    ZeroM,
    BigN,
    KGtN,
    NotEnoughChunks,
    Tbd,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Zfec error: {}",
            match self {
                Self::ZeroK => "'k' must be greater than 0",
                Self::ZeroM => "'m' must be greater than 0",
                Self::BigN => "'n' must be less than 257",
                Self::KGtN => "'k' must be less than 'n'",
                Self::NotEnoughChunks => "Not enough chunks were provided",
                Self::Tbd => "Unknown error",
            }
        )
    }
}
impl std::error::Error for Error {}

type Gf = u8;
type Result<Fec> = std::result::Result<Fec, Error>;

/// A chunk of encoded data
///
/// A `Chunk` can be deconstructed into a `(Vec<u8>, usize)` tuple
///
/// # Example
///
/// ```
/// use zfec_rs::Chunk;
///
/// let val: Vec<u8> = vec![0, 1, 2, 3, 4];
/// let chunk = Chunk::new(val.clone(), 0);
/// let (chunk_vec, chunk_i): (Vec<u8>, usize) = chunk.into();
/// assert_eq!(val, chunk_vec);
/// ```
#[derive(Debug, Clone)]
pub struct Chunk {
    pub data: Vec<u8>,
    pub index: usize,
}
impl Chunk {
    /// Creates a new chunk
    pub fn new(data: Vec<u8>, index: usize) -> Self {
        Self { data, index }
    }
}
impl From<Chunk> for (Vec<Gf>, usize) {
    fn from(val: Chunk) -> Self {
        (val.data, val.index)
    }
}
impl std::fmt::Display for Chunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {:?}", self.index, self.data)
    }
}

/// Forward Error Correcting encoder/decoder.
///
/// The encoder can be defined with 2 values: `k` and `m`
///
/// * `k` is the number of chunks needed to reconstruct the original message
/// * `m` is the total number of chunks that will be produced
///
/// The first `k` chunks contain the original unaltered data, meaning that if all the original chunks are
/// available on the decoding end, no decoding needs to take place.
///
/// The final `(m-k)` chunks contain the parity coding necessary to reproduce any one of the original chunks.
///
/// Coding is done with respect to the chunk's location within the encoded data. This means that each chunk's
/// sequence number is needed for correct reconstruction.
///
/// # Example
///
/// ```
/// use zfec_rs::Fec;
///
/// let message = b"Message to be sent";
///
/// let fec = Fec::new(5, 8).unwrap();
///
/// let (mut encoded_chunks, padding) = fec.encode(&message[..]).unwrap();
/// encoded_chunks.remove(2);
/// let decoded_message = fec.decode(&encoded_chunks, padding).unwrap();
///
/// assert_eq!(message.to_vec(), decoded_message);
/// ```
pub struct Fec {
    /// Number of chunks needed to reconstruct the original message
    k: usize,
    /// Total number of chunks that will be produced
    m: usize,
    enc_matrix: Vec<Gf>,
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
    /// Generates a new encoder/decoder
    pub fn new(k: usize, m: usize) -> Result<Fec> {
        //// eprintln!("Creating new - k: {}, n: {}", k, n);
        if k < 1 {
            return Err(Error::ZeroK);
        }
        if m < 1 {
            return Err(Error::ZeroM);
        }
        if m > 256 {
            return Err(Error::BigN);
        }
        if k > m {
            return Err(Error::KGtN);
        }
        let mut tmp_m: Vec<Gf> = vec![0; m * k];

        // m rows by k columns
        let mut enc_matrix: Vec<Gf> = vec![0; m * k];

        /*
         * fill the matrix with powers of field elements, starting from 0.
         * The first row is special, cannot be computed with exp. table.
         */
        tmp_m[0] = 1;
        (1..k).for_each(|col| {
            tmp_m[col] = 0;
        });
        for row in 0..(m - 1) {
            //// eprintln!("row: {}", row);
            let p: &mut [u8] = &mut tmp_m[(row + 1) * k..];
            (0..k).for_each(|col| {
                p[col] = statics::STATICS.gf_exp[Statics::modnn((row * col) as i32) as usize];
            });
        }

        /*
         * quick code to build systematic matrix: invert the top
         * k*k vandermonde matrix, multiply right the bottom n-k rows
         * by the inverse, and construct the identity matrix at the top.
         */
        // eprintln!("tmp_m: {:02x?}", tmp_m);
        statics::STATICS.invert_vdm(&mut tmp_m, k); /* much faster than _invert_mat */
        statics::STATICS.matmul(
            &tmp_m[k * k..],
            &tmp_m[..],
            &mut enc_matrix[k * k..],
            m - k,
            k,
            k,
        );
        /*
         * the upper matrix is I so do not bother with a slow multiply
         */
        // the Vec is initialized to 0's when defined
        // memset(retval->enc_matrix, '\0', k * k * sizeof(gf));
        for i in 0..k {
            //// eprintln!("i: {}", i);
            enc_matrix[i * (k + 1)] = 1;
        }

        // unnecessary in Rust, tmp_m gets dropped
        // free(tmp_m);

        Ok(Fec { k, m, enc_matrix })
    }
    /// Performs the encoding, returning the encoded chunks and the amount of padding
    ///
    /// Because all chunks need to be the same size, the data is padded with `0`s at the end as needed
    pub fn encode(&self, data: &[u8]) -> Result<(Vec<Chunk>, usize)> {
        // eprintln!("\nEncoding k: {}, m: {}", self.k, self.m);
        // clean side
        let chunk_size = self.chunk_size(data.len());
        // eprintln!("chunk_size: {}", chunk_size);
        let data_slice = data;

        let mut chunks = vec![];

        // eprintln!("data: {:02x?}", data);
        // eprintln!("data len: {:02x?}", data.len());
        let mut padding = 0;
        for i in 0..self.k {
            let mut temp_vec = vec![];
            if (i * chunk_size) >= data_slice.len() {
                // eprintln!("empty chunk");
                temp_vec.append(&mut vec![0; chunk_size].to_vec());
                padding += chunk_size;
            } else if ((i * chunk_size) < data_slice.len())
                && (((i + 1) * chunk_size) > data_slice.len())
            {
                // finish current chunk
                temp_vec.append(&mut data_slice[i * chunk_size..].to_vec());
                // add padding
                let added = ((i + 1) * chunk_size) - data_slice.len();
                // eprint!("final slice, padding");
                let mut added_padding = vec![0; added];
                temp_vec.append(&mut added_padding);
                padding += added;
            } else {
                let new_chunk = &data_slice[(i * chunk_size)..((i + 1) * chunk_size)];
                // eprintln!("normal chunk: {:02x?}", new_chunk);
                temp_vec.append(&mut new_chunk.to_vec())
            }
            chunks.push(temp_vec);
        }
        // eprintln!("Finished chunking");

        let num_check_blocks_produced = self.m - self.k;
        let mut check_blocks_produced = vec![vec![0; chunk_size]; num_check_blocks_produced];
        let check_block_ids: Vec<usize> = (self.k..self.m).collect();
        // eprintln!("num: {}", num_check_blocks_produced);
        // eprintln!("blocks: {:?}", check_blocks_produced);
        // eprintln!("ids: {:?}", check_block_ids);

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
                let p = &self.enc_matrix[fecnum * self.k..];
                // eprintln!("enc_matrix: {:02x?}", &self.enc_matrix);
                // eprintln!("p: {:02x?}", p);
                for j in 0..self.k {
                    // eprintln!("Loc 2");
                    statics::STATICS.addmul(
                        &mut check_blocks_produced[i][k..],
                        &chunks[j][k..k + stride],
                        p[j],
                        stride,
                    );
                }
            }

            k += STRIDE;
        }

        ///////// end internals

        let mut ret_chunks = vec![];
        ret_chunks.append(&mut chunks);
        ret_chunks.append(&mut check_blocks_produced);
        // eprintln!("ret_chunks: {:02x?}", ret_chunks);
        let mut ret_vec = vec![];
        for (i, chunk) in ret_chunks.iter().enumerate() {
            ret_vec.push(Chunk {
                    index: i,
                data: chunk.to_vec(),
            });
        }
        Ok((ret_vec, padding))
    }
    /// Performs the decoding
    pub fn decode(&self, encoded_data: &Vec<Chunk>, padding: usize) -> Result<Vec<u8>> {
        // eprintln!("\nDecoding");
        if encoded_data.len() < self.k {
            return Err(Error::NotEnoughChunks);
        }

        let mut share_nums: Vec<usize> = vec![];
        let mut chunks: Vec<Vec<u8>> = vec![vec![]; self.m];

        for chunk in encoded_data {
            let num = chunk.index;
            share_nums.push(num);
            chunks[num] = chunk.data.clone();
        }
        // eprintln!("encoded data: {:02x?}", encoded_data);
        // eprintln!("share_nums: {:02x?}", share_nums);
        // eprintln!("chunks: {:02x?}", chunks);

        let sz = chunks[share_nums[0]].len();
        let mut ret_chunks = vec![vec![0; sz]; self.k];

        let mut complete = true;
        let mut missing = std::collections::VecDeque::new();
        let mut replaced = vec![];
        // check which of the original chunks are missing
        for i in 0..self.k {
            if !share_nums.contains(&i) {
                complete = false;
                missing.push_back(i);
                // eprintln!("Missing {}", i);
            }
        }

        // replace the missing chunks with fec chunks
        for i in self.k..self.m {
            if !chunks[i].is_empty() {
                if let Some(index) = missing.pop_front() {
                    // eprintln!("Moving {} to {}", i, index);
                    replaced.push(index);
                    share_nums.insert(index, i);
                    chunks[index] = chunks[i].to_vec();
                    // eprintln!("share_nums: {:02x?}", share_nums);
                    // eprintln!("chunks: {:02x?}", chunks);
                }
            }
        }

        if complete {
            let flat = Self::flatten(&mut chunks[..self.k].to_vec());
            return Ok(flat[..flat.len() - padding].to_vec());
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
                    // eprintln!("Loc 2");
                    statics::STATICS.addmul(
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

        // eprintln!("replaced: {:02x?}", replaced);
        // eprintln!("ret_chunks: {:02x?}", ret_chunks);
        // fix the replaced chunks
        for i in 0..replaced.len() {
            chunks[replaced[i]] = ret_chunks[i].to_vec();
            // eprintln!("chunks: {:02x?}", chunks);
        }
        let ret_vec = Self::flatten(&mut chunks[0..self.k].to_vec());

        // remove padding
        Ok(ret_vec[..ret_vec.len() - padding].to_vec())
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
    fn build_decode_matrix_into_space(&self, index: &[usize], k: usize, matrix: &mut [Gf]) {
        for i in 0..k {
            let p = &mut matrix[i * k..];
            if index[i] < k {
                // we'll assume it's already 0
                // memset(p, 0, k);
                p[i] = 1;
            } else {
                // memcpy(p, &(code->enc_matrix[index[i] * code->k]), k);
                (0..k).for_each(|j| {
                    p[j] = self.enc_matrix[(index[i] * self.k) + j];
                });
            }
        }
        statics::STATICS._invert_mat(matrix, k);
    }
}
