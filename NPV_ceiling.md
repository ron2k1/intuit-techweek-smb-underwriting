# NPV Ceiling

## Core interpretation

If the headline NPV appears to cap around roughly **$15 million**, that is probably not just a modeling bug. It is likely the model finding the **economic frontier** of profitable underwriting.

This hackathon is not simply a classification task where the goal is to approve as many SMBs as possible. Part A is an **underwriting decision problem**: decide who to fund, estimate probability of default, and produce calibrated risk estimates. That means the right objective is not raw approval volume or even classification accuracy. The right objective is expected value under credit economics.

In other words:

> Approving more borrowers only helps until the marginal borrower has negative expected NPV.

Once the obvious low-risk, high-revenue borrowers are funded, additional approvals increasingly pull in applicants whose default risk consumes the loan economics.

---

## Why the ceiling happens

The loan economics are thin relative to default loss.

Approximate gross upside on a fully performing 60-day loan:

```text
3% origination fee + 35% APR × 60 / 365 ≈ 8.75%
```

So if loss given default is high, the break-even default probability is nowhere near 50%. It is in the single digits to low teens.

| LGD assumption | Approximate break-even PD |
|---:|---:|
| 100% LGD | ~8.0% |
| 70% LGD | ~11.1% |
| 50% LGD | ~14.9% |

That means a borrower with a 20%, 30%, or 40% default probability may still look acceptable under a naive classification lens, but is likely terrible under an underwriting-profit lens.

---

## The likely reason NPV flattens near $15M

The cumulative NPV curve probably looks something like this:

1. **Early approvals:** high-quality SMBs, positive expected NPV.
2. **Middle approvals:** still profitable, but lower margin and higher uncertainty.
3. **Later approvals:** marginal risk rises faster than incremental revenue.
4. **Beyond the peak:** each new approval adds funded volume but reduces cumulative expected NPV.

So the approximate $15M level may be the point where the available positive-EV applicant pool has been mostly exhausted.

This is normal in credit. A lender does not maximize profit by funding everyone. It maximizes profit by funding the set of borrowers whose risk-adjusted expected return is positive.

---

## Key causes of the ceiling

### 1. Positive expected-NPV borrowers are finite

There may only be a limited number of applicants whose expected interest and fees exceed expected default losses.

After those borrowers are approved, additional volume becomes progressively worse.

### 2. PD ranking may already capture the obvious good borrowers

If the model is reasonably good, the first tranche of approvals contains the cleanest risk-adjusted opportunities.

Past that point, the remaining applicants are more ambiguous: weaker cash flow, noisier signals, sparse history, worse cohort effects, missing bank-feed data, or higher prior-underwriter concern.

### 3. Selection bias / reject inference matters

Observed repayment outcomes are usually available only for historically approved and matured loans. That means the model may be estimating:

```text
P(default | prior lender approved)
```

But the policy is trying to estimate:

```text
P(default | we approve this applicant now)
```

Those are not the same thing.

The farther the new approval policy expands beyond the historical approval frontier, the less reliable the default estimate becomes.

### 4. Risk concentration can hurt without true black swans

This does not require a rare black-swan event. In credit, correlated ordinary losses are enough.

Examples:

- Same sector
- Same geography
- Same acquisition channel
- Same cohort week
- Same weak cash-flow pattern
- Same missing-data pattern
- Same prior-decision boundary

A few concentrated bad pockets can destroy the thin ~8.75% gross yield.

### 5. The scoring likely rewards discipline, not reckless NPV chasing

The challenge is not just headline NPV. It also includes probability calibration, timing/default-trajectory accuracy, uncertainty intervals, causal reasoning, and writeup quality.

Over-approving may increase funded dollars but hurt:

- PD calibration
- Default timing forecasts
- Interval coverage
- Cohort default curves
- Credibility of the underwriting policy
- Human-reviewed technical defense

---

## Better objective: repo expected NPV, not PD alone

Use the repo's implemented NPV formula from `src/economics.py`, not PD alone and not only the simplified yield/LGD approximation.

The brief frames the task as a small-business lender choosing whom to fund to maximize portfolio profit (`hackathon-brief.pdf`, p. 8), and Deliverable A is the applicant-level accept/reject decision (`hackathon-brief.pdf`, p. 9). The implemented code translates that into the following cash-flow equation:

```text
E[NPV_i | approve] =
  (1 - PD_i) * NPV_repaid_i
  + PD_i * NPV_default_i

NPV_repaid_i =
  F_i + R_i * r * T / 365

NPV_default_i =
  F_i + D_i * (E[t*_i] - 1) + E[recovery_i] - R_i

D_i =
  R_i * (1 + r * T / 365) / T

F_i = 0.03 * R_i
r   = 0.35
T   = 60
```

Where:

- `R_i` = requested/funded principal
- `PD_i` = predicted probability of default
- `F_i` = origination fee
- `r` = annual interest rate
- `T` = loan term in days
- `D_i` = expected daily repayment/draw factor in the default cash-flow formula
- `E[t*_i]` = expected default day conditional on default
- `E[recovery_i]` = expected recovered dollars if default occurs

The simplified intuition is still:

```text
Expected value = performing-loan economics - expected default loss
```

The underwriting rule should approve applicants by **marginal expected NPV**, not by PD threshold alone.

A borrower with higher PD might still be attractive if the exposure, repayment behavior, or recovery assumptions are favorable. Conversely, a borrower with moderate PD may be bad if the requested amount is large and LGD is severe.

---

## Does Deliverable B feed the NPV equation?

No. The submitted 13 x 13 Deliverable B grid should not be an input to the applicant-level NPV equation.

The brief defines Deliverable B as the cumulative default fraction by origination cohort `w` and loan age `a` weeks for the team's own approved set `A_w` (`hackathon-brief.pdf`, p. 9). The final scoring slide treats A's realized portfolio value and B's trajectory accuracy as separate score components (`hackathon-brief.pdf`, p. 14). That makes B a downstream portfolio trajectory forecast conditional on the A approval policy, not an applicant-level pricing input.

The repo uses the same underlying timing model for both A and B:

1. The hazard model estimates applicant-level cumulative default curves over 13 weekly buckets.
2. For A, those applicant-level curves are collapsed into `E[t*_i]` with `src.timing.expected_default_day(...)`, then passed into `src.economics.expected_npv(...)`.
3. For B, those applicant-level curves are filtered to the approved applicants and aggregated into the 13 x 13 cohort-week by loan-age grid.

So the direction is:

```text
Applicant hazard curves -> E[t*_i] -> A expected NPV decisions
Applicant hazard curves + A decisions -> B 13 x 13 trajectory
```

It is not:

```text
B 13 x 13 trajectory -> A expected NPV decisions
```

Using the submitted B grid inside A's NPV would be too coarse because the grid no longer contains applicant amount, individual PD, recovery estimate, or individual timing shape. It is useful for consistency checks and scoring Deliverable B, but the NPV equation should stay applicant-level.

---

## Diagnostic plot to run

Sort all applicants by expected NPV descending.

Then plot:

```text
x-axis: cumulative approved principal
 y-axis: cumulative expected NPV
```

If cumulative expected NPV peaks around **$15M**, then the ceiling is real.

That point is the efficient frontier of the current model and assumptions.

Do not push past the peak unless there is evidence that the model is underestimating a profitable segment.

---

## Recommended approval rule

A strong approval policy should be uncertainty-aware:

```text
Approve if:
  expected_npv > 0
  and PD upper 90% bound is below the economic break-even threshold
  and applicant is not far outside the observed/prior-approved training distribution
  and portfolio concentration limits are not violated
```

This is better than approving purely on point-estimate PD.

Near the approval cutoff, uncertainty matters. A borrower with `PD_point = 7%` but `PD_upper_90 = 18%` may be too risky if the economic break-even PD is around 8–12%.

---

## Practical next experiments

### 1. Plot the marginal NPV curve

Check whether marginal expected NPV turns negative after roughly $15M.

If yes, the cap is economically justified.

### 2. Segment the borrowers after the $15M cutoff

Look at who gets added when trying to push beyond the ceiling.

Useful cuts:

- Sector
- State / geography
- Cohort week
- Requested amount bands
- Bank-feed availability
- Prior-underwriter score
- Prior approval/decline decision
- Thin-file vs rich-file applicants
- Cash-flow volatility
- Revenue trend
- Missingness indicators

The goal is to identify whether the extra approvals are genuinely bad or whether the model is missing a profitable subsegment.

### 3. Use an uncertainty-adjusted cutoff

Instead of:

```text
Approve if expected_npv > 0
```

Try:

```text
Approve if expected_npv_lower_bound > 0
```

or:

```text
Approve if PD_upper_90 < break_even_PD
```

This may reduce headline volume but improve realized NPV, calibration, and defensibility.

### 4. Add concentration controls

Even if each loan is individually positive EV, the portfolio can become fragile if the same risk factor is repeated too many times.

Add caps by:

- Sector
- Cohort week
- State
- Channel
- Prior-risk band
- Requested amount band

### 5. Stress test LGD and PD calibration

Run approval policies under different assumptions:

| Scenario | PD adjustment | LGD assumption |
|---|---:|---:|
| Base | 1.00× | 70% |
| Conservative | 1.15× | 80% |
| Severe | 1.30× | 90% |
| Optimistic | 0.90× | 60% |

If the $15M policy survives conservative assumptions, it is much more defensible.

---

## Bottom line

The $15M NPV ceiling likely means the model has found the point where profitable underwriting stops.

Pushing beyond it without better risk separation likely converts the strategy from:

```text
Disciplined risk-adjusted lender
```

into:

```text
Volume-chasing lender
```

For this hackathon, the winning move is probably not to approve more borrowers. It is to approve the best positive-EV borrowers, calibrate the risk estimates, defend the uncertainty, and show that the policy avoids unprofitable marginal risk.
