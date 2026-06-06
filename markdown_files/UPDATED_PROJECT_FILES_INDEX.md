# Updated Project Files Index

_Last updated: 2026-06-05_

This folder has been updated for the official Intuit SMB Underwriting Challenge repo.

Sources incorporated:

```text
/Users/stevenyang/Downloads/hackathon-brief.pdf
https://github.com/intuit/intuit-techweek-nyc-hackathon-2026
repo files: README.md, dataset/README.md, data_dictionary.csv, validate_submission.py,
intervention_queries.csv, cohort_week_definitions.csv, submission_B_template.csv,
submission_D_writeup_template.md
```

## Gap assessment from the official brief and repo

These gaps were found or were under-emphasized in the project notes, then patched into the files below:

```text
1. Judging criteria were not stated as an explicit scorecard.
   Added mapping for SP&L, Straj, Scal, SC, and Swrite.

2. Hackathon context and operational deadlines were too easy to miss.
   Added Intuit NY Tech Week AI/ML Hackathon context for June 5-6, 2026.
   Added team registration by 8PM Friday, no team changes after registration,
   Saturday 14:00 submission deadline, and validator-before-upload guidance.

3. Deliverable D needed stricter treatment.
   Added the exact five-section order, 4-page body limit, 11pt font,
   0.75in margins, and the fact that causal reasoning carries the most weight.

4. Deliverable B needed stronger emphasis that it is over the team's own approved loans.
   Updated strategy/metrics language around approved-set cohort trajectories and monotonic CDR.

5. Counterfactual deliverable needed clearer causal framing.
   Added explicit do(feature=value) language and support checks for the 900 repo queries.

6. Submission validation needed to be treated as a hard disqualification gate.
   Added exact filenames, flat-folder rule, row counts, interval-order checks,
   ID coverage, and non-decreasing B trajectories.
```

## Updated files

```text
intuit_repo_analysis_action_plan.md
  Comprehensive repo-specific analysis and step-by-step action plan.

intuit_ml_hackathon_game_plan.md
  Rewritten from generic causal-ML guidance into the actual A/B/C/D strategy.

setup.md
  Rewritten with exact repo setup, unzip, project structure, scripts, and validation steps.

hackathon_relevant_technical_concepts.md
  Rewritten around selective labels, PD, profit, survival, counterfactuals, calibration, and regulatory framing.

hackathon_metrics_libraries_techniques.md
  Rewritten with deliverable-specific metrics, calibration, interval methods, model stack, and submission checks.

codex_prompts_for_winning_hackathon_strategy.md
  Rewritten as concrete Codex implementation prompts for this exact repo and dataset contract.
```

## Recommended read order

```text
1. intuit_repo_analysis_action_plan.md
2. setup.md
3. intuit_ml_hackathon_game_plan.md
4. hackathon_relevant_technical_concepts.md
5. hackathon_metrics_libraries_techniques.md
6. codex_prompts_for_winning_hackathon_strategy.md
```

## Immediate next command locally

```bash
git clone https://github.com/intuit/intuit-techweek-nyc-hackathon-2026.git intuit-smb-underwriting
cd intuit-smb-underwriting
unzip dataset/dataset-compressed.zip -d dataset
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn scipy matplotlib jupyter notebook ipykernel lightgbm xgboost catboost joblib pyyaml tqdm optuna shap
```

Then follow `setup.md` and use the Codex prompts to implement the pipeline.
