from __future__ import annotations

from am_mvt.ingestion.literature_manifest import save_literature_manifest


def main() -> None:
    output_path, manifest = save_literature_manifest()

    print("Step 02c complete: literature manifest exported.")
    print(f"Manifest: {output_path}")
    print(f"PDF articles listed: {len(manifest)}")

    if manifest.empty:
        print("No formal PDFs found in data/raw/pdfs/.")
        print("Run scripts/02b_prepare_pdfs.py --apply first.")
        return

    print(f"Ready for parsing: {int(manifest['ready_for_parsing'].sum())}")
    print(f"Title verified: {int(manifest['title_verified'].sum())}")
    print(
        "Needs human check: "
        f"{int(manifest['needs_human_check'].fillna(True).astype(bool).sum())}"
    )
    print("Only metadata is exported; PDF files are not copied.")


if __name__ == "__main__":
    main()
