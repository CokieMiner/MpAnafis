#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| mp_anafis_fuzz::unsigned::run(data));
