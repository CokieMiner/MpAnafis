//! Native digit blocks, padding, reversal, and append boundaries.

use alloc::vec;

use proptest::{
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{Convert, RadixParameters};

#[test]
fn radix_chunk_writes_match_scalar_digits_at_native_power_boundaries() {
    let check = |radix: u32, value: usize| {
        let parameters = RadixParameters::for_limb(radix);
        let chunk = value
            .checked_rem(parameters.max_power)
            .expect("positive native power");
        let base = usize::try_from(radix).expect("small radix");
        for padded in [false, true] {
            let mut scalar = chunk;
            let mut reference = vec![];
            loop {
                reference.push(Convert::byte_from_digit(
                    u8::try_from(scalar.checked_rem(base).expect("positive radix"))
                        .expect("one digit"),
                ));
                scalar = scalar.checked_div(base).expect("positive radix");
                if scalar == 0 {
                    break;
                }
            }
            if padded {
                reference.resize(parameters.max_digits, b'0');
            }
            let mut reverse = vec![b'x'];
            Convert::write_radix_chunk::<true>(chunk, base, parameters, padded, &mut reverse);
            assert_eq!(reverse.first(), Some(&b'x'));
            assert_eq!(reverse.get(1..).expect("appended digits"), reference);
            reference.reverse();
            let mut forward = vec![b'x'];
            Convert::write_radix_chunk::<false>(chunk, base, parameters, padded, &mut forward);
            assert_eq!(forward.first(), Some(&b'x'));
            assert_eq!(forward.get(1..).expect("appended digits"), reference);
        }
    };
    for radix in 3_u32..=36 {
        if radix.is_power_of_two() {
            continue;
        }
        let base = usize::try_from(radix).expect("small radix");
        let mut power = 1_usize;
        loop {
            for value in [
                0,
                power.checked_sub(1).expect("positive power"),
                power,
                power.checked_add(1).expect("native power below maximum"),
            ] {
                check(radix, value);
            }
            let Some(next) = power.checked_mul(base) else {
                break;
            };
            power = next;
        }
    }
    let strategy = (3_u32..=36, any::<usize>());
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(radix, value)| {
            if !radix.is_power_of_two() {
                check(radix, value);
            }
            Ok(())
        })
        .expect("native radix block property");
}
