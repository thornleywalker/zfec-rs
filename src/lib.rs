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
        eprintln!("chunk_size: {}", chunk_size);
        let data_slice = &data[..];

        let mut chunks = vec![];

        eprintln!("data: {:?}", data);
        eprintln!("data len: {:?}", data.len());

        for i in 0..self.k {
            let mut temp_vec = vec![];
            if (i * chunk_size) >= data_slice.len() {
                eprintln!("empty chunk");
                temp_vec.append(&mut vec![0; chunk_size].to_vec());
            }
            else if ((i * chunk_size) < data_slice.len()) && (((i + 1) * chunk_size) > data_slice.len())
            {
                // finish current chunk
                temp_vec.append(&mut data_slice[i * chunk_size..].to_vec());
                let remaining = ((i + 1) * chunk_size) as usize - data_slice.len();
                eprint!("final slice, padding");
                for _ in 0..remaining {
                    eprint!(".");
                    temp_vec.push(0);
                }
            } else {
                let new_chunk =
                    &data_slice[(i * chunk_size) as usize..((i + 1) * chunk_size) as usize];
                eprintln!("normal chunk: {:?}", new_chunk);
                temp_vec.append(&mut new_chunk.to_vec())
            }
            chunks.push(temp_vec);
        }
        eprintln!("Finished chunking");

        let num_check_blocks_produced = self.m - self.k;
        let mut check_blocks_produced = vec![vec![0; chunk_size]; num_check_blocks_produced];
        let check_block_ids: Vec<u32> = (self.k..self.m).map(|x| x as u32).collect();

        // dirty side
        unsafe {
            let inpkts = chunks
                .iter()
                .map(|chunk| chunk.as_ptr())
                .collect::<Vec<*const u8>>()
                .as_ptr();
            let fecs = check_blocks_produced
                .iter_mut()
                .map(|block| block.as_mut_ptr())
                .collect::<Vec<*mut u8>>()
                .as_mut_ptr();
            let block_nums = check_block_ids.as_ptr();
            let num_block_nums = num_check_blocks_produced as u64;
            let sz = chunk_size as usize;

            fec_encode(
                self.internal,
                inpkts,
                fecs,
                block_nums,
                num_block_nums,
                sz as u64,
            );
            eprintln!("chunks: {:?}", chunks);
            eprintln!("check_blocks_produced: {:?}", check_blocks_produced);
        }
        let mut ret_vec = vec![];
        for chunk in &mut chunks {
            ret_vec.append(chunk);
        }
        for block in &mut check_blocks_produced {
            ret_vec.append(block);
        }
        ret_vec
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
