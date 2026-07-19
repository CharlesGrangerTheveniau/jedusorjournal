/// Normalize text for rendering with a font whose coverage is Latin-1 only
/// (verified: EMS Allure has full Latin-1 coverage but no Latin-9 œ/Œ and
/// none of the typographic punctuation LLMs like to emit — em/en dashes,
/// curly quotes, ellipsis are all above U+00FF). Unsupported characters are
/// silently skipped at layout time, so without these substitutions an
/// answer like "but—c'est" renders as the jammed-together "butc'est".
/// All other characters pass through unchanged.
pub fn normalize(input: &str) -> String {
    input
        .chars()
        .flat_map(|c| match c {
            'œ' => vec!['o', 'e'],
            'Œ' => vec!['O', 'E'],
            '—' | '–' | '−' => vec!['-'],
            '\u{2018}' | '\u{2019}' => vec!['\''],
            '\u{201C}' | '\u{201D}' => vec!['"'],
            '…' => vec!['.', '.', '.'],
            other => vec![other],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitutes_lowercase_oe_ligature() {
        assert_eq!(normalize("cœur"), "coeur");
    }

    #[test]
    fn substitutes_uppercase_oe_ligature() {
        assert_eq!(normalize("ŒUF"), "OEUF");
    }

    #[test]
    fn leaves_accented_characters_untouched() {
        assert_eq!(normalize("Où étais-tu ?"), "Où étais-tu ?");
    }

    #[test]
    fn leaves_plain_ascii_untouched() {
        assert_eq!(normalize("Hello, world!"), "Hello, world!");
    }

    #[test]
    fn substitutes_typographic_punctuation() {
        assert_eq!(normalize("but—c'est"), "but-c'est");
        assert_eq!(normalize("l\u{2019}été"), "l'été");
        assert_eq!(normalize("\u{201C}oui\u{201D}"), "\"oui\"");
        assert_eq!(normalize("Eh bien…"), "Eh bien...");
    }
}
