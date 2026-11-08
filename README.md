# Does More Football Mean More Enjoyment?
Pilot study (n = 18 respondents, 14 with rated matches) testing diminishing marginal utility in football viewing.

## Headline results
- H1 (enjoyment falls with match order): mean within-person slope -0.67/match, 95% CI [-1.64, +0.30]. Not significant (t-test, Wilcoxon and permutation test all p ≈ 0.15-0.16).
- H2 (engaged fans decline less): difference +0.93, bootstrap CI [-0.57, +2.53]. Inconclusive.
- Favourite team playing: +1.20 enjoyment within person, CI [0.57, 1.83].
Small convenience sample; association only, not causation.

## Methods
Per-person OLS slopes, one-sample t / Wilcoxon / sign-flip permutation tests, Welch t, Mann-Whitney and bootstrap for group difference, person fixed-effects regression with cluster-robust standard errors.

## Reproduce
```
pip install -r requirements.txt
python analysis/analyze.py data/responses.csv
```
Then open `index.html` (it reads `results.js` and `charts/`). Timestamps were removed from the data before publication.

## Files
`analysis/analyze.py` analysis script · `data/responses.csv` survey export (timestamps blanked) · `responses_anonymised.csv` cleaned dataset · `exclusion_log.txt` cleaning log · `survey_questions.md` questionnaire
