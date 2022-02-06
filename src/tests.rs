use super::*;

const DATA: &[u8] = b"some_ssidthe_password";
const ENCODED: &[u8] =
    b"some_ssidthe_password\x00\x00\x00\x00]\xd8\x94\xea\x91\x1bGU\xff+\x882[\xa6\xd3";

#[test]
fn encoder_5_8_test() {
    let mut fec = Fec::new(5, 8);
    let encoded = fec.encode(&DATA.to_vec());
    assert_eq!(encoded, ENCODED.to_vec());
}

#[test]
// tests if fec can decode from complete message
fn decoder_5_8_test() {
    // test if decoder can decode from complete message
    let k = 5;
    let m = 8;
    let mut fec = Fec::new(k, m);
    let decoded = fec.decode(&ENCODED.to_vec());
    assert_eq!(
        decoded,
        DATA.to_vec(),
        "Failed to decode from complete encoded message"
    );

    // test for missing one of each group
    let chunk_size = fec.chunk_size(DATA.len());
    for i in 0..k {
        let mut broken_enc = vec![];
        broken_enc.append(&mut ENCODED[0..i * chunk_size].to_vec());
        broken_enc.append(&mut ENCODED[(i + 1) * chunk_size..].to_vec());
        let decoded = fec.decode(&broken_enc);
        assert_eq!(
            DATA.to_vec(),
            decoded,
            "Failed to decode while missing {}th block",
            i
        );
    }
}
