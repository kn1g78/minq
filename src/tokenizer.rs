//! Thin wrapper around the `tokenizers` crate for HF `tokenizer.json` files.

use std::path::Path;

use crate::{MinqError, Result};

/// Text tokenizer loaded from a HuggingFace `tokenizer.json`.
pub struct TextTokenizer {
    inner: tokenizers::Tokenizer,
}

/// Common end-of-sequence marker strings across LLaMA/Qwen-style models.
const EOS_CANDIDATES: &[&str] = &[
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "</s>",
    "<eos>",
];

impl TextTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| MinqError::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    /// Encode text into token ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| MinqError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids back into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| MinqError::Tokenizer(e.to_string()))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Ids of any well-known EOS markers present in this tokenizer's vocab.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        EOS_CANDIDATES
            .iter()
            .filter_map(|t| self.token_to_id(t))
            .collect()
    }
}

/// Byte length of the longest common prefix of `a` and `b`, aligned to char
/// boundaries (the prefix length always ends on a boundary of both strings).
pub fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut len = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        len += ca.len_utf8();
    }
    len
}

/// The safely-committable prefix of a freshly decoded string.
///
/// Decoders use `from_utf8_lossy` semantics, so when the id sequence so far
/// ends mid-character, the incomplete trailing bytes appear as a run of
/// U+FFFD at the very end. Committing that run to the terminal would leave
/// permanent `�` once the completing token arrives, so it is held back
/// until the character completes. Genuine mid-text U+FFFD is untouched.
fn safe_prefix(full: &str) -> &str {
    full.trim_end_matches('\u{FFFD}')
}

/// Incremental detokenizer for streaming output.
///
/// Decoding each token in isolation corrupts multi-byte UTF-8 characters
/// that span token boundaries (terminals show `�`). Instead, every push
/// re-decodes the *whole* generated id sequence and yields only the suffix
/// not printed before, measured as the char-aligned common prefix with the
/// previously emitted text and held back while the sequence might still end
/// mid-character. If generation stops mid-character (e.g. at a max-tokens
/// cut), the uncompletable trailing bytes are simply never printed — the
/// same policy llama.cpp uses. The emitted text therefore equals a one-shot
/// decode of all ids whenever the sequence ends on a complete character,
/// and never contains a mojibake `�`.
pub struct IncrementalDecode {
    ids: Vec<u32>,
    /// Text already emitted (never contains a held-back incomplete tail).
    printed: String,
}

impl Default for IncrementalDecode {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalDecode {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            printed: String::new(),
        }
    }

    /// Record one more token; returns the newly completed text to print, or
    /// `None` if decoding failed (the caller should skip printing, the state
    /// stays consistent for the next push).
    pub fn push(&mut self, tokenizer: &TextTokenizer, id: u32) -> Option<String> {
        self.ids.push(id);
        let full = tokenizer.decode(&self.ids).ok()?;
        let safe = safe_prefix(&full);
        // `printed` is always a prefix of the next safe prefix: the held-back
        // region only ever completes, it never rewrites earlier text.
        let prefix = common_prefix_len(&self.printed, safe);
        let delta = safe[prefix..].to_string();
        self.printed = safe.to_string();
        Some(delta)
    }

    /// Everything emitted so far.
    pub fn text(&self) -> &str {
        &self.printed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_prefix_respects_char_boundaries() {
        assert_eq!(common_prefix_len("hello", "hello world"), 5);
        assert_eq!(common_prefix_len("abc", "abd"), 2);
        assert_eq!(common_prefix_len("", "abc"), 0);
        assert_eq!(common_prefix_len("北京", "北京"), "北京".len());
        // Multi-byte: common prefix must stop before the differing char.
        assert_eq!(common_prefix_len("北京a", "北京b"), "北京".len());
        assert_eq!(common_prefix_len("北x", "北y"), "北".len());
        // A prefix cut mid-char is impossible by construction.
        assert_eq!(common_prefix_len("北京", "北"), "北".len());
    }

    #[test]
    fn safe_prefix_holds_back_only_trailing_replacement_chars() {
        // Trailing U+FFFD = incomplete tail bytes -> held back.
        assert_eq!(safe_prefix("abc\u{FFFD}"), "abc");
        assert_eq!(safe_prefix("abc\u{FFFD}\u{FFFD}"), "abc");
        assert_eq!(safe_prefix("\u{FFFD}"), "");
        // Complete text and genuine mid-text U+FFFD are untouched.
        assert_eq!(safe_prefix("abc"), "abc");
        assert_eq!(safe_prefix("北京"), "北京");
        assert_eq!(safe_prefix("a\u{FFFD}b"), "a\u{FFFD}b");
        assert_eq!(safe_prefix("B. 西部"), "B. 西部");
    }
}
