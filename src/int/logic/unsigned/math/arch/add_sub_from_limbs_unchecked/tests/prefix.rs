//! Binary carry propagation through four generate/propagate lanes.

/// Returns the input carry for each lane and the carry leaving lane 3.
///
/// Each input mask contains four bits and `incoming` is binary. Addition
/// generates at `raw < left` and propagates at `raw == MAX`; subtraction
/// generates at `left < right` and propagates at `raw == 0`.
pub const fn lane_carries(generate: u32, propagate: u32, incoming: u32) -> (u32, u32) {
    // The distance-one combine covers two lanes; distance two extends each
    // group to every preceding lane. A group propagates the external carry
    // exactly when every lane in that group propagates it.
    let distance_one_generate = generate | (propagate & (generate << 1));
    let distance_one_propagate = propagate & ((propagate << 1) | 0b0001);
    let prefix_generate =
        distance_one_generate | (distance_one_propagate & (distance_one_generate << 2));
    let prefix_propagate = distance_one_propagate & ((distance_one_propagate << 2) | 0b0011);
    let incoming_mask = 0_u32.wrapping_sub(incoming);
    let outputs = (prefix_generate | (prefix_propagate & incoming_mask)) & 0b1111;
    let inputs = incoming | ((outputs << 1) & 0b1110);
    (inputs, (outputs >> 3) & 1)
}

#[test]
fn four_lane_prefix_matches_binary_carry_recurrence() {
    for generate in 0..16 {
        for propagate in 0..16 {
            for incoming in 0..=1 {
                let (actual_inputs, actual_final) = lane_carries(generate, propagate, incoming);
                let mut carry = incoming;
                let mut inputs = 0;
                for lane in 0..4 {
                    inputs |= carry << lane;
                    carry = ((generate >> lane) & 1) | (((propagate >> lane) & 1) & carry);
                }
                assert_eq!((actual_inputs, actual_final), (inputs, carry));
            }
        }
    }
}
