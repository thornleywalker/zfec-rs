// bindgen
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
// end bindgen

#[cfg(test)]
mod tests;

pub struct Fec {
    internal: *mut fec_t,
    k: usize,
    m: usize,
}
impl Fec {
    pub fn new(k: usize, m: usize) -> Fec {
        Fec {
            internal: unsafe { fec_new(k as u16, m as u16) },
            k: k,
            m: m,
        }
    }
    pub fn encode(&mut self, data: &Vec<u8>) -> Vec<u8> {
        // clean side
        let chunk_size = (data.len() as f64 / self.k as f64).ceil() as usize;
        let data_slice = &data[..];

        let mut chunks = vec![];

        for val in data {
            eprint!("{:x}.", val);
        }
        eprintln!("");

        for i in 0..self.k {
            let mut temp_vec = vec![];
            if ((i + 1) * chunk_size) > data_slice.len() {
                eprint!("final slice, padding");
                temp_vec.append(&mut data_slice[i * chunk_size..data_slice.len() - 1].to_vec());
                let remaining = data_slice.len() - (i * chunk_size) as usize;
                for _ in 0..(remaining - 1) {
                    eprint!(".");
                    temp_vec.push(0);
                }
                eprintln!("");
            } else {
                eprintln!("normal chunk");
                temp_vec.append(
                    &mut data_slice[(i * chunk_size) as usize..((i + 1) * chunk_size) as usize]
                        .to_vec(),
                )
            }
            chunks.push(temp_vec);
        }
        eprintln!("Finished chunking");
        for chunk in chunks {
            eprint!("Chunk:");
            for val in chunk {
                eprint!("{:x}.", val);
            }
            eprintln!("");
        }

        let mut check_blocks_produced = vec![0; self.m - self.k];
        let check_block_ids = self.k..self.m;
        let check_blocks_produced = self.m - self.k;

        // dirty side
        unsafe {
            let inpkts = chunks.as_ptr();
            for (i, chunk) in chunks.enumerate() {}
            fec_encode(
                self.internal,
                &inpkts,
                &fecs,
                num_block_nums,
                block_nums,
                sz,
            );
        }
    }
    pub fn decode(&mut self, data: &Vec<u8>) -> Vec<u8> {
        vec![]
    }
}
impl Drop for Fec {
    fn drop(&mut self) {
        unsafe {
            fec_free(self.internal);
        }
    }
}
