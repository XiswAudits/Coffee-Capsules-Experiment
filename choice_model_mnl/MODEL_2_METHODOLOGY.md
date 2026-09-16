# Model 2 — Multinomial Logit (MNL): Step-by-Step Methodology

## 1. Start with the original data
- 11 weeks of observations.
- 3 households.
- Each household has Regular quantity and Premium quantity for every week.
- Prices are shared across households for each week.

## 2. Convert quantities into one observed choice
For each household-week:
- Regular quantity > 0 → **Regular**
- Otherwise, Premium quantity > 0 → **Premium**
- Otherwise → **No Purchase**

The MNL therefore has three alternatives:

**Regular | Premium | No Purchase**

## 3. Estimate one MNL separately for each household
Rather than pooling all households into one MNL, the model fits an independent model for each household using its 11 weekly observations.

This allows each household to have its own:
- Regular alternative-specific constant (ASC)
- Premium alternative-specific constant (ASC)
- Regular price coefficient
- Premium price coefficient

## 4. Standardize the prices
The same pooled price standardization is used across the dataset:

`Z_R = (P_R − mean(P_R)) / SD(P_R)`

`Z_P = (P_P − mean(P_P)) / SD(P_P)`

Using the 33 observations:
- Mean Regular price ≈ €46.09; SD ≈ €8.48
- Mean Premium price ≈ €81.73; SD ≈ €9.81

Example for Week 1:
- Regular = €40 → Z_R ≈ −0.718
- Premium = €70 → Z_P ≈ −1.196

## 5. Define utility for each alternative
The MNL starts by assigning each alternative a utility score.

For a given household:

`V_R = ASC_R + β_R × Z_R`

`V_P = ASC_P + β_P × Z_P`

`V_N = 0`

No Purchase is the reference alternative, so its utility is normalized to zero.

The ASCs capture baseline preference for Regular and Premium relative to No Purchase, while the price coefficients capture how standardized prices enter utility.

## 6. Convert utilities into probabilities using softmax
The MNL converts the three utility values into probabilities:

`P_i = exp(V_i) / [exp(V_R) + exp(V_P) + exp(V_N)]`

The three probabilities always sum to 1.

Example using initial parameters `[0, 0, −1, −1]` and Week 1 prices:
- V_R = 0 + (−1)(−0.718) = 0.718
- V_P = 0 + (−1)(−1.196) = 1.196
- V_N = 0

This gives approximately:
- P(Regular) = 32.2%
- P(Premium) = 52.0%
- P(No Purchase) = 15.7%

These are the probabilities for the **initial parameter guess**, before estimation.

## 7. Compare predicted probabilities with the actual choices
For every week, the model knows the observed choice and calculates the probability assigned to that choice.

For example, if the observed choice is Premium, the likelihood contribution is:

`P(Premium)`

If the observed choice is Regular, it contributes:

`P(Regular)`

If the observed choice is No Purchase, it contributes:

`P(No Purchase)`

The model wants to choose parameters that assign high probabilities to the choices that were actually observed.

## 8. Build the negative log-likelihood
The likelihood across all observations is maximized. In the code, this is expressed as a negative log-likelihood (NLL) that is minimized:

`NLL = −Σ log(P_observed)`

A lower NLL means the model assigns higher probability to the observed choices.

## 9. Estimate the parameters numerically
There is no closed-form solution for the MNL coefficients, so the model uses numerical optimization.

The implementation uses:
- **L-BFGS-B** optimization
- Parameter bounds of **−20 to +20**
- Three different starting points

The best optimization result (lowest NLL) is retained.

The bounds are numerical safeguards. If a parameter reaches ±20, this is an identification warning rather than evidence that the true parameter equals 20 or −20.

## 10. Example of the final estimated utility functions
For Household 2, the estimated parameters are approximately:

- ASC_Regular = 2.003
- ASC_Premium = −1.103
- β_Regular = −0.514
- β_Premium = −3.807

Therefore:

`V_R = 2.003 − 0.514 × Z_R`

`V_P = −1.103 − 3.807 × Z_P`

`V_N = 0`

These values are estimated from Household 2's 11 weekly choices.

## 11. Calculate the final probabilities for a scenario
For Household 2 in Week 1:
- Z_R ≈ −0.718
- Z_P ≈ −1.196

Utilities become approximately:

`V_R = 2.003 − 0.514(−0.718) ≈ 2.372`

`V_P = −1.103 − 3.807(−1.196) ≈ 3.449`

`V_N = 0`

Applying softmax gives approximately:
- P(Regular) ≈ 25.3%
- P(Premium) ≈ 74.0%
- P(No Purchase) ≈ 0.9%

The model therefore predicts Premium as the most likely choice for this scenario.

## 12. Derive the boundaries
The MNL boundaries are based on **equal utility**, rather than a 50% probability from a binary logistic regression.

### Regular = No Purchase
Set:

`V_R = V_N = 0`

So:

`ASC_R + β_R Z_R = 0`

This gives the Regular/No Purchase boundary in price space.

### Premium = No Purchase
Set:

`V_P = V_N = 0`

So:

`ASC_P + β_P Z_P = 0`

This gives the Premium/No Purchase boundary.

### Regular = Premium
Set:

`V_R = V_P`

So:

`ASC_R + β_R Z_R = ASC_P + β_P Z_P`

After converting the standardized prices back to euros, this produces the diagonal Regular = Premium boundary.

## 13. Create the decision map
The model evaluates many combinations of Regular and Premium prices.

For every price pair, it calculates:
1. Regular utility
2. Premium utility
3. No Purchase utility
4. The three softmax probabilities
5. The alternative with the highest probability

The highest-probability alternative determines the background region in the interactive map.

## 14. Add the observed data
The observed weekly choices are plotted on top of the predicted regions. This makes it possible to compare:
- What the household actually chose
- What the MNL predicts across the price space

## 15. Add the quantity layer
The model also reports observed quantity statistics by household.

Expected quantity is interpreted as:

`E[Q] = P(Regular) × E[Q | Regular] + P(Premium) × E[Q | Premium] + P(No Purchase) × 0`

This connects the choice probabilities to the observed quantity data, but it should be interpreted as an illustrative quantity layer rather than a structural quantity-demand model.

## 16. Identification warning
There are only 11 observations per household. The observed choice counts are:

| Household | Regular | Premium | No Purchase |
|---|---:|---:|---:|
| Household 1 | 7 | 4 | 0 |
| Household 2 | 7 | 4 | 0 |
| Household 3 | 0 | 7 | 4 |

Households 1 and 2 have **no observed No Purchase choices**. This creates weak identification/separation for parameters involving the No Purchase alternative. Therefore, if a parameter hits the ±20 bound, it should be interpreted cautiously.

## 17. Important model limitations
- Only 11 observations are available per household.
- Each household is estimated independently, so the sample per model is very small.
- MNL relies on the **IIA (Independence of Irrelevant Alternatives)** assumption.
- The model is exploratory and has not been validated as a causal demand model.
- Estimated coefficients should not automatically be interpreted as validated causal price elasticities or willingness-to-pay estimates.
- A larger dataset and/or a pooled, hierarchical, or mixed-logit specification would provide a stronger basis for inference.

## Key idea
**Raw data → observed choices → standardize prices → define utilities → softmax probabilities → compare with observed choices → negative log-likelihood → numerical optimization → household-specific parameters → final probabilities → equal-utility boundaries → decision map.**
