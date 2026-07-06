# AI Disclosure Statement

## AI tool used

Claude (Anthropic), accessed via the Claude.ai chat interface.

## Tasks for which it was used

AI assistance was extensive and is disclosed in full below, rather than described
narrowly as "coding support." Specifically, Claude was used to:

1. **Read and translate the original Stata replication code.** The original
   author's replication package includes a ~2,600-line Stata do-file
   (`generate tables.do`). Claude read this file section by section and
   translated its data-construction logic (propensity-score estimation, the
   "cleaned line" purification of bookmaker spreads, the weekly win/expected-win
   panel, lag/lead construction, IPEDS reporting-date resolution, inverse-
   probability-weight construction) into Python/pandas, since no Stata license
   was available.
2. **Design and implement both estimators.** Claude implemented the Sequential
   Treatment Effects (STE) model (a weighted-least-squares regression, given in
   closed form in the paper's equation 8) and the propensity-score IV/matching
   model (paper equations 3-5, a custom per-week/per-bin stratified estimator)
   in `statsmodels`.
3. **Debug against the published results.** Claude iteratively compared
   replicated coefficients against the values published in the paper's Tables
   1 and 3-6, and used discrepancies to find and fix two implementation issues:
   (a) a label-matching bug that caused several Table 1 rows to display as
   missing, and (b) a missing year-fixed-effects residualization step in the
   IV model's dependent variable, which is present in the original Stata code
   but not documented in the paper's prose, and whose omission had produced IV
   estimates that correlated with the published values at only r ~ 0.87 (with
   one sign flip) before the fix, versus r = 0.998 after.
4. **Produce the comparison, figures, and report text.** Claude computed the
   correlation/sign-agreement/sample-size comparison statistics, generated the
   two figures, and drafted the written replication report (including the
   description of methodology, results, and limitations below).
5. **Organize the code and repository** into the structure required by this
   assignment.

## How the output was checked, corrected, or modified

- Every stage of the data pipeline was validated against an external
  benchmark before proceeding to the next: the rebuilt panel's summary
  statistics were checked against the paper's published Table 1 (all values
  matched to the reported precision) before any estimator-specific code was
  written; each estimator's output was then checked against the paper's
  Tables 3-6.
- The full pipeline was run end-to-end from a clean state (deleting all
  intermediate files and rerunning `01_data_cleaning.py` -> `02_analysis.py`
  -> `03_tables_figures.py` from scratch) to confirm it is reproducible and
  free of hidden state or manual steps.
- The two implementation issues described above (Table 1 label mismatch; IV
  model residualization) were caught specifically *because* the numerical
  comparison to the published results was done carefully rather than
  accepted at face value -- this comparison is itself part of the submitted
  analysis (see the report's "Replication Accuracy" section) rather than a
  one-off debugging step.

## Student confirmation

*[To be completed and personally verified by each student before submission.]*

We have reviewed the code in `code/01_data_cleaning.py`, `code/02_analysis.py`,
and `code/03_tables_figures.py`, and the accompanying replication report. We
understand the data-construction steps, the two estimators (the STE model's
closed-form regression and the IV model's per-week propensity-score-bin
stratification), and the reasons for the differences between the replicated
and published results discussed in the report. Each of us can explain any
part of this submission if asked.

Signed: _______________________ (Xuan Wang)       Date: _______________________
Signed: _______________________ (Mai Tran)         Date: _______________________
