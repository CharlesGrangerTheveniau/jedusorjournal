//! Detect which reMarkable document was most recently opened, so the diary
//! effect can pick a persona prompt for a designated diary notebook while
//! still answering normally everywhere else.
//!
//! On-device investigation (see docs/superpowers/plans/2026-07-09-tom-riddle-diary.md,
//! Task 8) showed xochitl does NOT keep an open file descriptor on the
//! current document while displaying it, ruling out an fd-scan approach.
//! Instead, each document's `<uuid>.metadata` file (a JSON object) has a
//! `lastOpened` field: a string containing epoch milliseconds, maintained
//! by the app itself. We rank all metadata files by that field and return
//! the most recent.

use serde::Deserialize;
use std::path::Path;

const XOCHITL_DATA_DIR: &str = "/home/root/.local/share/remarkable/xochitl";

#[derive(Deserialize)]
struct Metadata {
    #[serde(rename = "lastOpened")]
    last_opened: Option<String>,
}

/// Detect the UUID of the document most recently opened, by scanning every
/// `<uuid>.metadata` file in the xochitl data directory and ranking by the
/// `lastOpened` field. Returns `None` if the directory doesn't exist, is
/// empty, or no metadata file has a parseable `lastOpened` value — callers
/// must treat `None` as "use the neutral prompt," never as an error.
pub fn detect_open_document() -> Option<String> {
    detect_open_document_in(Path::new(XOCHITL_DATA_DIR))
}

fn detect_open_document_in(data_dir: &Path) -> Option<String> {
    let entries = std::fs::read_dir(data_dir).ok()?;

    let mut best: Option<(u64, String)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("metadata") {
            continue;
        }
        let Some(uuid) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(contents) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(metadata) = serde_json::from_str::<Metadata>(&contents) else {
            continue;
        };
        let Some(last_opened_str) = metadata.last_opened else {
            continue; // folders and some documents have no lastOpened field
        };
        let Ok(last_opened) = last_opened_str.parse::<u64>() else {
            continue;
        };

        match &best {
            Some((best_ts, _)) if *best_ts >= last_opened => {}
            _ => best = Some((last_opened, uuid.to_string())),
        }
    }

    best.map(|(_, uuid)| uuid)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fake xochitl data directory under a unique temp path:
    /// `<root>/<uuid>.metadata` files with real-shaped JSON content.
    struct Fixture {
        root: std::path::PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!("gw_notebook_test_{}_{}", name, std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            Fixture { root }
        }

        fn write_metadata(&self, uuid: &str, last_opened_ms: Option<u64>) {
            let body = match last_opened_ms {
                Some(ts) => format!(r#"{{"lastOpened": "{}", "visibleName": "test"}}"#, ts),
                None => r#"{"visibleName": "folder, no lastOpened"}"#.to_string(),
            };
            std::fs::write(self.root.join(format!("{}.metadata", uuid)), body).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn detects_the_most_recently_opened_document() {
        let fixture = Fixture::new("most_recent");
        fixture.write_metadata("older-doc", Some(1_000_000));
        fixture.write_metadata("newest-doc", Some(2_000_000));
        fixture.write_metadata("oldest-doc", Some(500_000));

        let result = detect_open_document_in(&fixture.root);
        assert_eq!(result, Some("newest-doc".to_string()));
    }

    #[test]
    fn ignores_entries_without_a_last_opened_field() {
        let fixture = Fixture::new("no_last_opened");
        fixture.write_metadata("a-folder", None); // e.g. a folder, per real device data
        fixture.write_metadata("a-document", Some(1_000_000));

        let result = detect_open_document_in(&fixture.root);
        assert_eq!(result, Some("a-document".to_string()));
    }

    #[test]
    fn returns_none_when_directory_is_empty() {
        let fixture = Fixture::new("empty_dir");
        let result = detect_open_document_in(&fixture.root);
        assert_eq!(result, None);
    }

    #[test]
    fn returns_none_when_directory_does_not_exist() {
        let missing = std::env::temp_dir().join("gw_notebook_test_does_not_exist_12345");
        let result = detect_open_document_in(&missing);
        assert_eq!(result, None);
    }

    #[test]
    fn ignores_non_metadata_files() {
        let fixture = Fixture::new("non_metadata");
        fixture.write_metadata("a-document", Some(1_000_000));
        std::fs::write(fixture.root.join("a-document.content"), "{}").unwrap();
        std::fs::write(fixture.root.join("random.txt"), "not metadata").unwrap();

        let result = detect_open_document_in(&fixture.root);
        assert_eq!(result, Some("a-document".to_string()));
    }
}
