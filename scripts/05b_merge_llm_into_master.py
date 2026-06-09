from __future__ import annotations

from am_mvt.extraction.merge_llm_records import append_llm_records_to_master
from am_mvt.modelling.build_views import save_modelling_views


def main() -> None:
    print("Merging audited LLM extracted records into master dataset...")
    output_path, summary = append_llm_records_to_master(make_backup=False)

    print("\nStep 05b complete: LLM records merged into master dataset.")
    print(f"Updated master dataset: {output_path}")
    print("\nMerge summary:")
    print(summary)

    view_paths, view_summary = save_modelling_views(master_path=output_path)
    print("\nModelling views rebuilt from the audited merged master dataset:")
    for name, path in view_paths.items():
        print(f"{name}: {path}")
    print(view_summary)


if __name__ == "__main__":
    main()
