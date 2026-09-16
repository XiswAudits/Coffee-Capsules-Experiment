# Model 1 — Two-Stage Logistic Regression: Step-by-Step Methodology

## 1. Start with the original data
- 11 weeks of observations.
- 3 households.
- Therefore: 11 × 3 = 33 household-week observations.
- For each observation: Regular price, Premium price, Regular quantity, and Premium quantity.

## 2. Convert quantities into choices
For every household-week:
- Regular quantity > 0 → **Regular**
- Otherwise, Premium quantity > 0 → **Premium**
- Otherwise → **No Purchase**

Example: if Regular = 5 and Premium = 0, the observed choice is Regular. If both quantities are 0, the observed choice is No Purchase.

## 3. Split the decision into two stages
Instead of modelling the three choices directly, the model asks two questions:

**Stage 1 — Do I buy?**
- Buy = 1 if Regular or Premium
- Buy = 0 if No Purchase

**Stage 2 — What do I buy, given that I buy?**
- Premium = 1 if Premium
- Premium = 0 if Regular

So the logic is: **Do I buy? → What do I buy?**

## 4. Standardize the prices
Prices are converted into z-scores:

`Z_R = (P_R − mean(P_R)) / SD(P_R)`

`Z_P = (P_P − mean(P_P)) / SD(P_P)`

Using the 33 observations:
- Mean Regular price ≈ €46.09; SD ≈ €8.48
- Mean Premium price ≈ €81.73; SD ≈ €9.81

Example for Week 1:
- Regular = €40 → Z_R ≈ −0.718
- Premium = €70 → Z_P ≈ −1.196

## 5. Add household variables
Household 1 is the reference household. Household 2 and Household 3 are represented by dummy variables. The regression therefore uses:
- Standardized Regular price
- Standardized Premium price
- Household 2 dummy
- Household 3 dummy

## 6. Estimate Stage 1 — probability of buying
A logistic regression is fitted using all 33 observations:

`P(Buy) = 1 / (1 + e^(−Z))`

where:

`Z = β₀ + β_R Z_R + β_P Z_P + β_HH2 HH2 + β_HH3 HH3`

The coefficients are estimated from the observed Buy/No Purchase outcomes.

Approximate fitted coefficients:
- β₀ = 3.003
- β_R = 0.177
- β_P = −1.197
- β_HH2 = 0.555
- β_HH3 = −1.448

For Household 1, the household dummies are zero, so:

`Z = 3.003 + 0.177 Z_R − 1.197 Z_P`

## 7. Calculate an actual purchase probability
For Household 1, Week 1:
- Z_R ≈ −0.718
- Z_P ≈ −1.196

Therefore:

`Z ≈ 3.003 + 0.177(−0.718) − 1.197(−1.196) ≈ 4.31`

Then:

`P(Buy) = 1 / (1 + e^(−4.31)) ≈ 98.7%`

So the model estimates about a **98.7% probability of purchasing something**.

## 8. Keep only observations where a purchase occurred
Stage 2 is conditional on buying.
- Household 1: 11 purchases
- Household 2: 11 purchases
- Household 3: 7 purchases

Therefore, 29 of the 33 observations are used in Stage 2. The four No Purchase observations are excluded.

## 9. Estimate Stage 2 — Regular vs Premium
A second logistic regression is fitted to the 29 buyer observations.

The dependent variable is:
- Premium = 1
- Regular = 0

Approximate fitted coefficients:
- γ₀ = −1.096
- γ_R = 0.520
- γ_P = −1.514
- γ_HH2 = 0.125
- γ_HH3 = 1.542

For Household 1:

`Z = −1.096 + 0.520 Z_R − 1.514 Z_P`

## 10. Calculate an actual conditional choice probability
For Household 1, Week 4:
- Regular = €65 → Z_R ≈ 2.23
- Premium = €75 → Z_P ≈ −0.69
- Actual choice = Premium

Therefore:

`Z ≈ −1.096 + 0.520(2.23) − 1.514(−0.69) ≈ 1.11`

So:

`P(Premium | Buy) ≈ 75%`

The model therefore gives approximately a 75% probability of Premium conditional on buying.

## 11. Combine the two stages
The final probabilities are:

`P(No Purchase) = 1 − P(Buy)`

`P(Premium) = P(Buy) × P(Premium | Buy)`

`P(Regular) = P(Buy) × [1 − P(Premium | Buy)]`

Example: if P(Buy) = 80% and P(Premium | Buy) = 60%:
- No Purchase = 20%
- Premium = 48%
- Regular = 32%

## 12. Derive the Purchase boundary
The Purchase boundary is where the logistic model predicts a 50% probability of buying.

For logistic regression, P(Buy) = 50% means Z = 0.

For Household 1:

`3.003 + 0.177 Z_R − 1.197 Z_P = 0`

After rearranging and converting standardized prices back to euros, the boundary is approximately:

`P_P = 0.17 × P_R + 98.4`

The 0.17 and 98.4 are derived from the estimated coefficients and the price means/standard deviations; they are not separately estimated coefficients.

## 13. Derive the Regular/Premium boundary
For Stage 2, set:

`P(Premium | Buy) = 50%`

which again means Z = 0.

For Household 1:

`−1.096 + 0.520 Z_R − 1.514 Z_P = 0`

After converting back to euros:

`P_P = 0.40 × P_R + 56.3`

## 14. Create the decision map
For many Regular/Premium price combinations, the model calculates:
- P(Regular)
- P(Premium)
- P(No Purchase)

The choice with the highest probability becomes the predicted choice. These predictions create the regions and boundaries shown in the interactive graph.

## Key idea
**Raw data → observed choices → two logistic regressions → estimated coefficients → probabilities → combined final probabilities → 50% boundaries → decision map.**
