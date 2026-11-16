"""Write ``corpus_studies.csv``: one row per study of the OpenDV corpus with its venue group.

Venues come from the source records of each set (``analyses/development_sources/*.yaml``,
``analyses/heldout*/sources.yaml``). The 11 development studies fetched from their repositories
have no such record; their venues are below, from the paper cited in the study's mapping header
or, for two of them, from the DOI's Crossref record (checked 2026-11-15).

Venue groups:

- ``sigchi``: conferences and journals sponsored or co-sponsored by ACM SIGCHI: CHI, CSCW (PACM HCI),
  AutomotiveUI, MobileHCI, UbiComp/IMWUT and ACM/IEEE HRI;
- ``acm_other``: other ACM venues (DOI prefix 10.1145): Mensch und Computer, AVI, MUM, IMX, CHASE
  and ACM THRI; several are held in cooperation with SIGCHI;
- ``ieee``: IEEE conferences and journals;
- ``journal_other``: other journals and conferences (human factors, transport, psychology, ...);
- ``no_paper``: a data repository without a paper DOI.

    python -m analyses.opendv_corpus.studies
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RECORDS = (
    REPO / "analyses" / "development_sources" / "sources.yaml",
    REPO / "analyses" / "development_sources" / "sources_search.yaml",
    REPO / "analyses" / "heldout" / "sources.yaml",
    REPO / "analyses" / "heldout_2" / "sources.yaml",
)

# Development studies without a source record: (venue, year, DOI, domain, where the venue is from).
REMOTE_DEVELOPMENT = {
    "roads_chi25": ("CHI 2025", 2025, "10.1145/3706598.3713476", "automotive", "mapping header"),
    "ehmi_optimization_chi25": ("CHI 2025", 2025, "10.1145/3706598.3714187", "automotive", "manifest DOI (CHI '25 proceedings 3706598)"),
    "osf_cwd6h": ("CHI 2025", 2025, "10.1145/3706598.3713762", "automotive", "mapping header (CHI '25 proceedings 3706598)"),
    "road_bumps_touch": ("AutomotiveUI 2018", 2018, "10.1145/3239060.3239071", "automotive", "mapping header"),
    "3d_display_comp_vehicle": ("MUM 2023", 2023, "10.1145/3626705.3627786", "automotive", "Crossref"),
    "longitudinal_usa_germany_ehmi": (None, None, None, "automotive", "repository without a paper DOI"),
    "ehmi_for_all_chi26": ("CHI 2026", 2026, "10.1145/3772318.3790585", "automotive", "mapping header"),
    "fact_av": (None, None, None, "automotive", "repository without a paper DOI"),
    "running_into_traffic_chi26": ("CHI 2026", 2026, "10.1145/3772318.3790401", "automotive", "mapping header"),
    "psych_conflict_handling": ("IMWUT 2024", 2024, "10.1145/3678511", "automotive", "mapping header, Crossref"),
    "scrolling_in_the_deep_chi25": ("CHI 2025", 2025, "10.1145/3706598.3713187", "attention and media", "mapping header"),
}

_SIGCHI = re.compile(
    r"\bCHI\b|CSCW|AutomotiveUI|Automotive User Interfaces|MobileHCI|IMWUT|Interactive, Mobile, Wearable"
    r"|Human-Robot Interaction \(HRI\)|ACM/IEEE International Conference on Human-Robot Interaction",
    re.IGNORECASE,
)


def venue_group(venue: str | None, doi: str | None) -> str:
    if not venue and not doi:
        return "no_paper"
    venue = venue or ""
    if _SIGCHI.search(venue) and "Transactions on Human-Robot" not in venue:
        return "sigchi"
    if (doi or "").lower().startswith("10.1145/"):
        return "acm_other"
    if (doi or "").lower().startswith("10.1109/"):
        return "ieee"
    return "journal_other"


def _year(venue: str | None) -> int | None:
    years = re.findall(r"(?:19|20)\d\d", venue or "")
    return int(years[-1]) if years else None


def build() -> pd.DataFrame:
    records = {}
    for path in RECORDS:
        for src in yaml.safe_load(path.read_text(encoding="utf-8")).get("sources", []):
            records[src["source_id"]] = src
    manifest = yaml.safe_load((HERE / "sources_manifest.yaml").read_text(encoding="utf-8"))["sources"]
    rows = []
    for entry in manifest:
        sid = entry["source_id"]
        if sid in REMOTE_DEVELOPMENT:
            venue, year, doi, domain, venue_from = REMOTE_DEVELOPMENT[sid]
        else:
            rec = records[sid]
            venue = (rec.get("venue") or "").replace("&amp;", "&").replace("�", "–")
            year, doi, domain, venue_from = _year(venue), rec.get("doi"), rec.get("domain"), "source record"
        rows.append({
            "source_id": sid, "study_set": entry["study_set"], "venue": venue, "year": year,
            "doi": doi, "venue_group": venue_group(venue, doi), "domain": domain,
            "venue_from": venue_from, "mapping_path": entry.get("mapping_path"),
            "mapping_corrected_from": entry.get("mapping_corrected_from"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    out = HERE / "corpus_studies.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} studies)")
    print(pd.crosstab(df["venue_group"], df["study_set"], margins=True))


if __name__ == "__main__":
    main()
