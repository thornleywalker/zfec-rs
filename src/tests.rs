use super::*;

#[test]
fn it_works() {
    let mut fec = Fec::new(6, 10);
    let data = vec![
        0, 3, 5, 89, 9, 4, 0, 5, 8, 1, 1, 4, 2, 7, 6, 6, 8, 1, 5, 6, 7, 5, 67,
    ];
    let encoded = fec.encode(&data);
    eprintln!("encoded: {:?}", encoded);
    let decoded = fec.decode(&encoded);
    eprintln!("decoded: {:?}", decoded);
    eprintln!("originl: {:?}", &data);
    assert_eq!(decoded, data);
}
