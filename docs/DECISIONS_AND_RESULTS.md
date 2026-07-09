# Decision Log and Results Summary

## Part 1 — Decision log (defend these in the viva)

| # | Decision | Alternatives considered | Rationale |
|---|----------|------------------------|-----------|
| 1 | 2 Hz player-state sampling | full 64-tick; 1 Hz | positional behaviour needs ~seconds resolution; keeps per-demo tick tables ~50k rows; features insensitive at this scale |
| 2 | Manifest with `event__stage__match` naming from day one | plain filenames | HLTV filenames collide across events; encoding the event makes files self-describing and auto-fills manifest metadata |
| 3 | Behaviour-only clustering features (19); kills/damage as `ctx_*` only | include performance stats | roles must describe *how*, not *how well* (Drachen principle); performance features would conflate skill with style |
| 4 | Exclude `planted_bomb`, `defused_bomb` from clustering | keep them | round *events*, not playstyles; defuses occur only in CT-won rounds → would couple the taxonomy (hence consistency) to outcomes non-behaviourally. Development runs produced a degenerate "defuser role" (~2% of CT rounds) with them included |
| 5 | Exclude `enemies_flashed` | keep it | it is the *success* of utility (throwing is already captured); its heavy-tailed count formed an outlier pocket hijacking a SOM cluster at any k (~1.5% "flash-outlier role" in development runs) |
| 6 | Separate SOMs per side (T / CT) | one joint taxonomy | the sides have structurally different jobs; a joint map mostly rediscovers "T vs CT" |
| 7 | SOM + k-means on the codebook; plain k-means kept as agreement baseline | k-means only; GMM; HDBSCAN | direct methodological bridge to Drachen et al. 2009; two-stage clustering per Vesanto & Alhoniemi 2000 |
| 8 | SOM_ITERATIONS = 100,000 (raised from 20,000) | leave at 20k | convergence for a 22×22 grid; measured seed-to-seed ARI gain ≈ +0.05, plateau beyond 100k |
| 9 | k = 4 (T) | 5–7 | clear silhouette peak (0.106 vs ~0.09), balanced shares |
| 10 | k = 5 (CT) | k = 4 (marginally better silhouette 0.108 vs 0.090) | k = 5 splits the dominant 48% rifler cluster into orthogonal, balanced Site Anchor vs Rotating Rifler profiles — better construct validity and consistency resolution where the data mass lies; k = 4 covered by the k ± 1 robustness run |
| 11 | Manual role naming from profiles/radars, matched by *signature*, never by index | trust heuristic suggestions | the human inspection step is the source paper's method; cluster numbering is arbitrary between fits |
| 12 | Consistency scoped to player-halves | whole-match | T and CT taxonomies are separate; mixing sides is meaningless; contiguous-block logic handles OT swaps |
| 13 | Rolling consistency = priors only, ≥ 3 prior rounds | include current round; whole-half value | eliminates outcome→consistency leakage by construction; costs rounds 1–3 of each half (69.8% usable) |
| 14 | Modal-role share as headline; entropy and switch rate as alternates; team mean and team min aggregations | single metric | pre-specifying the family avoids post-hoc metric shopping; all alternates reported |
| 15 | Round-level logit, T-perspective, SEs clustered by match | round-level OLS; match-level only | binary outcome; rounds within a match are dependent; effective N ≈ 76 matches, honestly reported via CIs |
| 16 | Ecos kept for role assignment; removed in a robustness run | drop ecos everywhere | economy context is behaviour too (`buy_value_rel_team`, `buy_type`); the no-eco run then tests dependence — and proved decisive (see below) |
| 17 | Uniform pre-specified robustness battery applied to *every* candidate effect | robustness for the headline only | the battery caught the weakest-link effect as economy-mediated; asymmetric scrutiny would have shipped a false positive |
| 18 | Correlation framing throughout | causal language | observational esports data; the design distinguishes prospective from contemporaneous association, which is the strongest honest claim available |
| 19 | Variant computations use `save=False` | overwrite-and-restore | permutation replicates once overwrote canonical parquets; builders now protect Drive artifacts by construction |

## Part 2 — Results summary (canonical numbers)

**Corpus.** 76 maps · 10 events (2024–25) · 13 teams · 1,652 rounds ·
16,520 player-rounds (8,260/side). QC: 75/76 clean; one flag = the 42-round
triple-OT IEM Cologne 2024 final (official 22–20), verified and retained.

**Taxonomy (shares of player-rounds).**
T: Aggressor 42.6% · Support Rifler 40.1% · Lurker 11.0% · AWPer 6.3%.
CT: Rotating Rifler 25.8% · Site Anchor 25.1% · Eco/Save 20.8% ·
Aggressor 17.5% · AWPer 10.7%. A near-identical AWPer emerged independently
on both sides; CT > T AWPer share matches the weapon's defensive role.

**Validity.** Buy-type crosstabs isolate economy (CT Eco/Save is
predominantly non-full-buy; T Aggressor absorbs eco rushes — 39% eco).
Star players land correctly without supervision (ZywOo/m0NESY/broky →
CT:AWPer 0.46–0.53; donk → T:Aggressor 0.54, AWPer rounds ≈ 1.5%). Most
consistent player-halves (≥ 8 rounds): sh1ro, mezii at 11/12; IGL apEX
appears at both extremes. Stability: global ARI T 0.471 / CT 0.561
(k-means agreement 0.390 / 0.600); per-cluster Jaccard — specialists
reproduce strongly (AWPer 0.90 T / 0.93 CT; Eco 0.74; Aggressor 0.68–0.70;
Support/Rotator 0.64–0.73), positional-isolation roles are soft
(Lurker 0.47, Anchor 0.53): role structure is real but graded.

**Consistency.** Per player-half (n = 1,670): modal share mean 0.548
(SD 0.160, range 0.25–1.00); normalised entropy 0.626; switch rate 0.589 —
roles are round-level tactical assignments, and consistency is a modal
property of a half, not round-to-round repetition.

**Outcome (three levels).**
1. *Prospective round-level logit* (n = 1,153; SEs clustered on 76 matches):
   cons_diff β = 0.319, 95% CI [−0.881, 1.520], p = 0.60 — null, with large
   average-consistency effects ruled out. Positive control: equip_diff
   β = 0.328/$1k, z = 9.8. Map FEs recover known balance (nuke −0.576,
   p = 0.009). Quartile win rates flat: 48.4 / 46.5 / 46.0 / 46.7%.
2. *Alternatives:* entropy_diff β = 0.001 (p = 0.998); weakest-link
   cons_min_diff β = 1.102 (p = 0.039) → sent to the battery.
3. *Contemporaneous half-level OLS* (n = 152): team modal share β = 0.542,
   p = 0.002 — consistency and success strongly associated within halves.

**Robustness.** cons_diff null holds everywhere: no-eco β = 0.254
(p = 0.71, n = 680); k−1 −0.298 / k+1 0.927 (both ns); per-map betas scatter
(−0.72…+3.04, small n); permutation p = 0.47 (permuted −0.154 ± 0.387).
cons_min_diff is **fragile**: sign flips under no-eco (−0.516, p = 0.427),
collapses at k−1 (0.295); permutation p = 0.010 (minimum attainable at 100
replicates) rejects only "no player-linked structure", which the
eco-exclusion check adjudicates toward an economy artefact. Reported as
exploratory.

**Interpretation.** Winning enables consistency more than consistency
produces wins; the pipeline's contribution is a measurement-plus-design
framework that can distinguish these channels.
