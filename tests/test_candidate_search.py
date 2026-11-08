"""Pure helpers of the open-data search (no network)."""

from analyses.candidate_search import search_open_data as s


def test_licence_classification() -> None:
    assert s.licence_of({"rightsList": [{"rightsIdentifier": "cc-by-4.0"}]}) == ("cc-by-4.0", "open")
    assert s.licence_of({"rightsList": [{"rightsIdentifier": "cc0-1.0"}]})[1] == "open"
    by_uri = {"rightsList": [{"rightsUri": "https://creativecommons.org/licenses/by/4.0/legalcode"}]}
    assert s.licence_of(by_uri)[1] == "open"
    assert s.licence_of({"rightsList": [{"rightsIdentifier": "cc-by-nc-4.0"}]})[1] == "restricted"
    assert s.licence_of({"rightsList": [{"rightsIdentifier": "cc-by-nc-nd-4.0"}]})[1] == "restricted"
    assert s.licence_of({"rightsList": []}) == ("", "unknown")
    assert s.licence_of({"rightsList": [{"rights": "All rights reserved"}]})[1] == "unknown"


def test_concept_key_merges_versions() -> None:
    zenodo = {"relatedIdentifiers": [{"relationType": "IsVersionOf", "relatedIdentifierType": "DOI",
                                      "relatedIdentifier": "10.5281/zenodo.100"}]}
    assert s.concept_key("10.5281/zenodo.101", zenodo) == "10.5281/zenodo.100"
    assert s.concept_key("10.4121/14465715.v2", {}) == "10.4121/14465715"
    assert s.concept_key("10.6084/m9.figshare.123.v1", {}) == "10.6084/m9.figshare.123"
    assert s.concept_key("10.17632/abc12def.3", {}) == "10.17632/abc12def"
    # 4TU sometimes declares the base DOI a version of ".v1"
    reverse = {"relatedIdentifiers": [{"relationType": "IsVersionOf", "relatedIdentifierType": "DOI",
                                       "relatedIdentifier": "10.4121/abc.v1"}]}
    assert s.concept_key("10.4121/abc", reverse) == s.concept_key("10.4121/abc.v1", {})


def test_linked_papers_skip_repository_dois() -> None:
    attrs = {"relatedIdentifiers": [
        {"relationType": "IsSupplementTo", "relatedIdentifierType": "DOI",
         "relatedIdentifier": "10.1145/3613904.3642298"},
        {"relationType": "IsSupplementTo", "relatedIdentifierType": "DOI", "relatedIdentifier": "10.5281/zenodo.5"},
        {"relationType": "IsSupplementTo", "relatedIdentifierType": "URL",
         "relatedIdentifier": "https://doi.org/10.1016/J.TRF.2021.10.017"},
        {"relationType": "HasVersion", "relatedIdentifierType": "DOI", "relatedIdentifier": "10.1145/1"},
    ]}
    assert s.linked_paper_dois(attrs) == ["10.1145/3613904.3642298", "10.1016/j.trf.2021.10.017"]


def test_clean_title_and_cues() -> None:
    title = 'Data from: "Interaction between pedestrians and AVs"'
    assert s._clean_title(title) == "Interaction between pedestrians and AVs"
    assert s._clean_title("Dataset for Try This for Size") == "Try This for Size"
    cues = s.cues("Raw NASA-TLX and SUS of a within-subjects study")
    assert {"nasa_tlx", "sus", "repeated_measures"} <= set(cues.split(","))


def test_file_profile_flags_unsafe_and_archives() -> None:
    profile = s.file_profile([("data.csv", 10), ("bundle.zip", 100), ("model.pkl", 5), ("paper.pdf", 7)])
    assert profile["n_tabular"] == 1 and profile["has_archive"] and profile["has_unsafe"]
    assert profile["total_bytes"] == 122


def test_file_rules_keep_tables_and_codebooks_and_leave_out_logs_and_code():
    from analyses.candidate_search.build_sources import classify

    assert classify("data/questionnaire.xlsx", 10_000) == ("table", "")
    assert classify("Readme_data.xlsx", 10_000)[0] == "documentation"
    assert classify("Questionnaires/Post-trial questionnaire.pdf", 10_000)[0] == "documentation"
    assert classify("paper.pdf", 10_000)[0] == "skip"
    assert classify("signals/raw_emg.csv", 10_000)[0] == "skip"
    assert classify("big.csv", 60_000_000) == ("skip", "table over 50 MB")
    assert classify("model.pkl", 10)[0] == "skip"
    assert classify("__MACOSX/._q.csv", 10)[0] == "skip"
    assert classify("Supplementary material.zip", 10) == ("archive", "")
