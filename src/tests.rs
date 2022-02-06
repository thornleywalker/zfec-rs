use super::*;

const DATA: &[u8] = b"some_ssidthe_password";
const ENCODED: &[u8] =
    b"some_ssidthe_password\x00\x00\x00\x00]\xd8\x94\xea\x91\x1bGU\xff+\x882[\xa6\xd3";

#[test]
fn encoder_5_8_test() {
    encoder_test(5, 8);
}
fn encoder_test(k: usize, m: usize) {
    let mut fec = Fec::new(k, m);
    let mut encoded_chunks = fec.encode(&DATA.to_vec());
    let mut encoded = vec![];
    let _: () = encoded_chunks
        .iter_mut()
        .map(|chunk| encoded.append(chunk))
        .collect();
    eprintln!("encoded: {:?}", encoded);
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
    for m in 5..8 {
        for k in 2..m - 1 {
            decoder_test(k, m);
        }
    }
}
// assumes encoder works
fn decoder_test(k: usize, m: usize) {
    let mut fec = Fec::new(k, m);
    let mut chunks_enc = vec![];
    let chunk_size = fec.chunk_size(DATA.len());
    let padding = chunk_size * fec.k - DATA.len();
    for (i, chunk) in fec.encode(&DATA.to_vec()).iter().enumerate() {
        chunks_enc.push((i, chunk.to_vec()));
    }
    // test if decoder can decode from complete message
    eprintln!("padding: {}", padding);
    let decoded = fec.decode(&chunks_enc, padding);
    assert_eq!(
        decoded,
        DATA.to_vec(),
        "Failed to decode from complete k: {}, m: {} encoded message",
        k,
        m
    );
    eprintln!("Successfully decoded at k: {}, m: {}", fec.k, fec.m);

    // test for missing parts of each group
    for i in 0..k - 1 {
        let mut broken_enc = vec![];
        broken_enc.append(&mut chunks_enc[0..i].to_vec());
        broken_enc.append(&mut chunks_enc[(i + 2)..].to_vec());
        let decoded = fec.decode(&broken_enc, padding);
        assert_eq!(
            decoded,
            DATA.to_vec(),
            "Failed to decode while missing {}th block",
            i
        );
    }
}
