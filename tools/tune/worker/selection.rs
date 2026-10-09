//! Validated indices in a frozen whole-profile worker catalog.

/// Worker selection parsing. Indices always refer to the complete domain order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellSelection;

impl CellSelection {
    /// Empty input executes the complete catalog. Explicit indices must be
    /// sorted, unique and in bounds; validation precedes operand construction.
    pub fn parse(encoded: &str, count: usize) -> Result<Vec<usize>, String> {
        if count == 0 {
            return Err("worker domain has no cells on this target".to_owned());
        }
        if encoded.is_empty() {
            return Ok((0..count).collect());
        }
        let mut indices = Vec::new();
        for field in encoded.split(',') {
            let (span, step_text) = field.split_once('/').unwrap_or((field, "1"));
            let step = step_text
                .parse::<usize>()
                .map_err(|error| error.to_string())?;
            let (start_text, end_text) = span.split_once('-').unwrap_or((span, span));
            let start = start_text
                .parse::<usize>()
                .map_err(|error| error.to_string())?;
            let end = end_text
                .parse::<usize>()
                .map_err(|error| error.to_string())?;
            if start > end || end >= count || step == 0 {
                return Err("worker cell range exceeds its catalog".to_owned());
            }
            indices.extend((start..=end).step_by(step));
        }
        if indices.iter().any(|&index| index >= count)
            || indices
                .windows(2)
                .any(|pair| matches!(pair, [a, b] if a >= b))
        {
            return Err(
                "worker cells must be sorted, unique and within the domain catalog".to_owned(),
            );
        }
        Ok(indices)
    }

    /// Encode sorted indices as maximal arithmetic progressions. Sparse output
    /// selections stay short even when the complete division catalog is large.
    #[must_use]
    pub fn encode(indices: &[usize]) -> String {
        let mut fields = Vec::new();
        let mut position = 0;
        while let Some(&first) = indices.get(position) {
            let mut end = position;
            let step = indices
                .get(position.checked_add(1).expect("finite catalog"))
                .map_or(1, |next| {
                    next.checked_sub(first).expect("sorted unique indices")
                });
            while let (Some(&current), Some(&next)) = (
                indices.get(end),
                indices.get(end.checked_add(1).expect("finite catalog")),
            ) {
                if next.checked_sub(current) != Some(step) {
                    break;
                }
                end = end.checked_add(1).expect("finite catalog");
            }
            let last = *indices.get(end).expect("selected run end");
            fields.push(if end == position {
                first.to_string()
            } else {
                format!("{first}-{last}/{step}")
            });
            position = end.checked_add(1).expect("finite catalog");
        }
        fields.join(",")
    }
}
