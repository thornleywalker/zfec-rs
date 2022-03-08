use super::*;
use rand::{thread_rng, Rng};
use std::collections::BTreeMap;

const DATA: &[u8] = b"some_ssidthe_password";
const ENCODED: &[u8] =
    b"some_ssidthe_password\x00\x00\x00\x00]\xd8\x94\xea\x91\x1bGU\xff+\x882[\xa6\xd3";

#[test]
fn encoder_5_8_test() {
    encoder_test(5, 8);
}
fn encoder_test(k: usize, m: usize) {
    let mut fec = Fec::new(k, m).unwrap();
    let mut encoded_chunks = fec.encode(&DATA.to_vec()).unwrap();
    let mut encoded = vec![];
    for chunk in &mut encoded_chunks {
        encoded.append(chunk);
    }
    // let _: () = encoded_chunks
    //     .iter_mut()
    //     .map(|chunk| encoded.append(chunk))
    //     .collect();
    eprintln!("encoded: {:02x?}", encoded);
    assert_eq!(encoded, ENCODED);
}

#[test]
// tests if fec can decode for k=5, m=8
fn decoder_5_8_test() {
    decoder_test(5, 8);
}
#[test]
// tests various combinations of k and m
fn decoder_extensive() {
    for m in 2..20 {
        for k in 1..m - 1 {
            decoder_test(k, m);
        }
    }
}
fn map_to_vec(map: &BTreeMap<usize, Vec<u8>>) -> Vec<(usize, Vec<u8>)> {
    let mut ret_vec = vec![];
    for (i, v) in map {
        ret_vec.push((*i, v.clone()));
    }
    ret_vec
}
// assumes encoder works
fn decoder_test(k: usize, m: usize) {
    let mut fec = Fec::new(k, m).unwrap();
    let mut chunks_enc: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    let chunk_size = fec.chunk_size(DATA.len());
    let padding = chunk_size * fec.k - DATA.len();
    for (i, chunk) in fec.encode(&DATA.to_vec()).unwrap().iter().enumerate() {
        chunks_enc.insert(i, chunk.to_vec());
    }
    // test if decoder can decode from complete message
    eprintln!("padding: {}", padding);
    let decoded = fec.decode(&map_to_vec(&chunks_enc), padding);
    assert_eq!(
        decoded,
        DATA.to_vec(),
        "Failed to decode from complete k: {}, m: {} encoded message",
        k,
        m
    );
    eprintln!("Successfully decoded at k: {}, m: {}", fec.k, fec.m);

    // test for missing each part of each group
    for i in 0..m {
        eprintln!("With #{} missing", i);
        let mut broken_enc = chunks_enc.clone();
        broken_enc.remove(&i);
        //eprintln!("brkn_enc: {:02x?}", broken_enc);
        let decoded = fec.decode(&map_to_vec(&broken_enc), padding);
        assert_eq!(
            decoded,
            DATA.to_vec(),
            "Failed to decode while missing {}th block",
            i,
        );
    }

    // test for mising n parts of each group, up to the max allowable
    let max_missing = m - k;
    let mut rng = rand::thread_rng();
    // just to really give it a go
    for _ in 0..20 {
        for n in 1..=max_missing {
            eprintln!("With {} missing chunks", n);
            let mut broken_enc = chunks_enc.clone();
            for i in 0..n {
                let keys = &broken_enc.keys().collect::<Vec<&usize>>()[..];
                eprintln!("keys: {:02x?}", keys);
                let index = rng.gen_range(0..keys.len());
                let to_remove = *keys[index];
                eprintln!("Removing #{}", to_remove);
                let _ = broken_enc.remove(&to_remove);
            }
            eprintln!("broken: {:02x?}", broken_enc);
            let decoded = fec.decode(&map_to_vec(&broken_enc), padding);
            assert_eq!(
                decoded,
                DATA.to_vec(),
                "Failed to decode from k: {}, m: {} encoded message with {} missing chunks",
                k,
                m,
                n
            );
        }
    }
}
