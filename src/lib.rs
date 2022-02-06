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
    pub fn encode(&mut self, data: &Vec<u8>) -> Vec<Vec<u8>> {
        eprintln!("\nEncoding k: {}, m: {}", self.k, self.m);
        // clean side
        let chunk_size = self.chunk_size(data.len());
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
            } else if ((i * chunk_size) < data_slice.len())
                && (((i + 1) * chunk_size) > data_slice.len())
            {
                // finish current chunk
                temp_vec.append(&mut data_slice[i * chunk_size..].to_vec());
                // add padding
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
        unsafe {
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
        let mut ret_chunks = vec![];
        ret_chunks.append(&mut chunks);
        ret_chunks.append(&mut check_blocks_produced);
        eprintln!("ret_chunks: {:?}", ret_chunks);
        ret_chunks
    }
    // takes the data with it's block index, and how much padding there is after the message
    pub fn decode(&mut self, encoded_data: &Vec<(usize, Vec<u8>)>, padding: usize) -> Vec<u8> {
        eprintln!("\nDecoding");

        let mut share_nums: Vec<u32> = vec![];
        let mut chunks: Vec<Vec<u8>> = vec![vec![]; self.m];

        for (num, chunk) in encoded_data {
            share_nums.push(num.clone() as u32);
            chunks[*num] = chunk.to_vec();
            //chunks.insert(*num, chunk.to_vec());
        }
        eprintln!("encoded data: {:?}", encoded_data);
        eprintln!("share_nums: {:?}", share_nums);
        eprintln!("chunks: {:?}", chunks);

        let sz = chunks[share_nums[0] as usize].len();
        let mut ret_chunks = vec![vec![0; sz]; self.k];

        let mut complete = true;
        let mut missing = vec![];
        let mut replaced = vec![];
        for i in 0..self.m {
            if !share_nums.contains(&(i as u32)) {
                complete = false;
                missing.insert(0, i);
                eprintln!("Missing {}", i);
            }
            if i >= self.k {
                match missing.pop() {
                    Some(index) => {
                        eprintln!("Moving {} to {}", i, index);
                        replaced.push(index);
                        share_nums.insert(index, i as u32);
                        chunks[index] = chunks[i].to_vec();
                        eprintln!("share_nums: {:?}", share_nums);
                        eprintln!("chunks: {:?}", chunks);
                    }
                    None => {}
                }
            }
        }
        if complete {
            let flat = Self::flatten(&mut chunks[..self.k].to_vec());
            return flat[..flat.len() - padding].to_vec();
        }

        let inpkts = chunks
            .iter()
            .map(|chunk| chunk.as_ptr())
            .collect::<Vec<*const u8>>()
            .as_ptr();
        let outpkts = ret_chunks
            .iter_mut()
            .map(|block| block.as_mut_ptr())
            .collect::<Vec<*mut u8>>()
            .as_mut_ptr();
        let index = share_nums.as_ptr();
        eprintln!("Inner call");
        unsafe {
            fec_decode(self.internal, inpkts, outpkts, index, sz as u64);
        }
        eprintln!("replaced: {:?}", replaced);
        eprintln!("ret_chunks: {:?}", ret_chunks);
        // fix the replaced chunks
        for i in 0..replaced.len() {
            chunks[replaced[i]] = ret_chunks[i].to_vec();
            eprintln!("chunks: {:?}", chunks);
        }
        let ret_vec = Self::flatten(&mut chunks[0..self.k].to_vec());

        // remove padding
        ret_vec[..ret_vec.len() - padding].to_vec()
    }
}
impl Drop for Fec {
    fn drop(&mut self) {
        unsafe {
            fec_free(self.internal);
        }
    }
}
