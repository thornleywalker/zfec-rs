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

use super::*;
use rand::{thread_rng, Rng};
use std::{collections::BTreeMap, fs, io::prelude::*, time};

const DATA: &[u8] = b"some_ssidthe_password";
const ENCODED: &[u8] =
    b"some_ssidthe_password\x00\x00\x00\x00]\xd8\x94\xea\x91\x1bGU\xff+\x882[\xa6\xd3";

#[test]
fn encoder_5_8_test() {
    encoder_test(5, 8);
}
fn encoder_test(k: usize, m: usize) {
    let fec = Fec::new(k, m).unwrap();
    let (mut encoded_chunks, _) = fec.encode(&DATA.to_vec()).unwrap();
    let mut encoded = vec![];
    for chunk in &mut encoded_chunks {
        encoded.append(&mut chunk.data);
    }
    assert_eq!(encoded, ENCODED);
}
// tests if fec can decode for k=5, m=8
#[test]
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
// assumes encoder works
fn decoder_test(k: usize, m: usize) {
    let mut fec = Fec::new(k, m).unwrap();
    // let mut chunks_enc: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    let (chunks, padding) = fec.encode(&DATA).unwrap();
    // for (i, chunk) in chunks.iter().enumerate() {
    //     chunks_enc.insert(i, chunk.to_vec());
    // }
    // test if decoder can decode from complete message
    // eprintln!!("padding: {}", padding);
    let decoded = fec.decode(&chunks, padding).unwrap();
    assert_eq!(
        decoded,
        DATA.to_vec(),
        "Failed to decode from complete k: {}, m: {} encoded message",
        k,
        m
    );
    // eprintln!!("Successfully decoded at k: {}, m: {}", fec.k, fec.m);

    // test for missing each part of each group
    for i in 0..m {
        // eprintln!!("With #{} missing", i);
        let mut broken_enc = chunks.clone();
        broken_enc.remove(i);
        //eprintln!("brkn_enc: {:02x?}", broken_enc);
        let decoded = fec.decode(&broken_enc, padding).unwrap();
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
            // eprintln!!("With {} missing chunks", n);
            let mut broken_enc = chunks.clone();
            for _ in 0..n {
                let keys = &broken_enc
                    .iter()
                    .map(|chunk| chunk.index)
                    .collect::<Vec<usize>>()[..];
                // eprintln!!("keys: {:02x?}", keys);
                let index = rng.gen_range(0..keys.len());
                let _ = broken_enc.remove(index);
            }
            // eprintln!!("broken: {:02x?}", broken_enc);
            let decoded = fec.decode(&broken_enc, padding).unwrap();
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
