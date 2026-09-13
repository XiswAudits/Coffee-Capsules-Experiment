# Coffee Capsule Choice Model — Independent Household MNL

This app is the discrete-choice modelling version of the Coffee Capsules Experiment. It uses the same 11-week dataset as the main experiment, but replaces the previous sequential binary-classification approach with **three independent multinomial logit (MNL) models: one model per household**.

## What the model is trying to do

Each household-week is treated as one choice occasion. At the observed pair of prices, the household chooses exactly one alternative:

- **Regular**
- **Premium**
- **No Purchase**

The objective is to estimate how each individual household's choice probabilities change when the Regular and Premium prices change.

## Why zero-purchase weeks stay in the data

A zero-purchase week is a real demand observation, not missing data. It means the household faced the experimental prices and chose not to buy. Therefore, zero weeks remain in the model as the **No Purchase** alternative.

For descriptive demand, the zero-inclusive average is calculated across all 11 weeks. This answers: **how many capsules did this household demand per week over the entire experiment, including weeks in which it bought nothing?**

## Why an MNL model?

The previous application used two sequential binary logistic classifiers:

1. Buy vs No Purchase
2. Regular vs Premium conditional on buying

The MNL instead treats the three outcomes as **competing alternatives in one Random Utility Model**. This is closer to standard choice modelling because the model estimates the relative attractiveness of all available alternatives simultaneously.

For household `h` and alternative `i`:

`U_ih = V_ih + epsilon_ih`

The household is represented as choosing the alternative with the highest realised utility.

## Utility specification

The first implementation is deliberately parsimonious because there are only 11 observations per household:

`V_R = ASC_R + beta_R * P_R`

`V_P = ASC_P + beta_P * P_P`

`V_N = 0`

No Purchase is the reference alternative, so its systematic utility is normalised to zero for identification.

Prices are standardised internally before estimation for numerical stability; displayed boundary equations are converted back to the original price units.

### Variables

**Choice variables**

- `Choice`: observed alternative — Regular, Premium, or No Purchase.
- `Quantity`: observed number of capsules purchased that week.

**Explanatory variables**

- `P_Regular`: Regular price.
- `P_Premium`: Premium price.
- `Household`: used to select which independent model is estimated; households are not pooled into one coefficient system.
- `T`: experiment week, retained as the choice occasion identifier. It is not currently used as a behavioural coefficient because 11 weeks are too few to justify a rich time specification.

Quantity is intentionally not used as an explanatory variable for choice: it is an outcome of the purchase decision. Instead, the app uses observed conditional quantities to create a simple scenario demand layer after estimating choice probabilities.

## MNL probabilities

The model converts systematic utilities into joint choice probabilities:

`P_i = exp(V_i) / [exp(V_R) + exp(V_P) + exp(V_N)]`

Therefore:

`P(Regular) + P(Premium) + P(No Purchase) = 1`

Changing either price changes relative utilities and therefore all three probabilities.

## Estimation

Parameters are estimated by maximum likelihood. If `y_t` is the observed alternative in week `t`, the likelihood is:

`L(theta) = product_t P(y_t | P_R,t, P_P,t; theta)`

and the log-likelihood is:

`log L(theta) = sum_t log P(y_t | P_R,t, P_P,t; theta)`

The implementation uses numerical optimisation with finite parameter bounds.

### Why are bounds necessary?

With only 11 observations per household, some alternatives are never observed:

| Household | Regular | Premium | No Purchase |
|---|---:|---:|---:|
| Household 1 | 7 | 4 | 0 |
| Household 2 | 7 | 4 | 0 |
| Household 3 | 0 | 7 | 4 |

An unconstrained MNL can therefore experience complete or quasi separation and drive coefficients toward extremely large magnitudes. The app keeps the optimisation finite and **flags parameters that reach the bound**. A bound-hitting estimate should not be interpreted as a precise economic coefficient.

This is a deliberate modelling choice to keep the independent household simulator usable while making the identification limitation visible to the viewer.

## Choice map and boundaries

The decision map evaluates the three MNL probabilities over a grid of Regular/Premium prices. The background shows the alternative with the highest probability.

Pairwise boundaries are utility-equality conditions:

### Regular vs Premium

`V_R = V_P`

With the linear price specification this becomes a straight line:

`P_P = slope * P_R + intercept`

### Regular vs No Purchase

`V_R = V_N`

This produces a Regular-price threshold under the current specification.

### Premium vs No Purchase

`V_P = V_N`

This produces a Premium-price threshold under the current specification.

These are **choice-model boundaries**, not validated causal willingness-to-pay thresholds.

## Quantity and zero-inclusive demand

Choice and quantity are kept conceptually separate.

The observed zero-inclusive average is:

`mean(Q_1, Q_2, ..., Q_11)`

where a No Purchase week contributes `Q = 0`.

For an illustrative scenario, expected weekly quantity is calculated as:

`E[Q] = P(Regular) * E[Q | Regular] + P(Premium) * E[Q | Premium] + P(No Purchase) * 0`

This is a simple demand layer using the experiment's observed conditional purchase quantities. It is **not** a separately estimated structural consumption model.

## What the app shows

- Household selector.
- Regular and Premium price simulator.
- Three joint MNL choice probabilities.
- Most likely choice.
- Zero-inclusive observed average demand.
- Scenario expected weekly demand.
- P_R × P_P decision map.
- Observed household-week choices.
- Pairwise utility boundaries.
- Utility and probability equations.
- Maximum-likelihood diagnostics.
- In-sample accuracy.
- Identification warnings when parameters hit finite bounds.
- Expandable explanations of the model, assumptions, demand layer, and limitations.

## Interpretation rules

The model is an **exploratory research model**, not a validated causal demand model. In particular:

- There are only 11 observations per household.
- Household 1 and Household 2 have no observed No Purchase choices.
- Household 3 has no observed Regular choices.
- In-sample accuracy is not out-of-sample validation.
- Coefficients should not be presented as validated causal price elasticities or willingness-to-pay estimates.
- The independent household design prioritises household-specific behavioural interpretation over statistical pooling.

## Why this is still useful

Despite the small sample, the model provides a coherent choice-modelling framework for the experiment. It turns the price pair into three competing probabilities, allows No Purchase to be treated as a genuine outside option, and provides a natural bridge from choice probabilities to expected demand.

The next data expansion should be used to test whether the household-specific coefficients remain stable and whether a pooled, hierarchical, random-parameter, or mixed-logit model becomes statistically defensible.

## Deployment

The Streamlit entry point is:

`choice_model_mnl/app.py`

The app uses the repository's shared `coffee_capsules_data.csv` file and the root `requirements.txt`. On Streamlit Community Cloud, select `choice_model_mnl/app.py` as the application file.

## Reference framework

The implementation follows the standard multinomial-logit discrete-choice framework used in discrete-choice econometrics, including alternative-specific constants, a reference alternative, systematic utility, logit choice probabilities, and maximum-likelihood estimation. Biogeme provides a useful reference implementation and examples of this framework: https://biogeme.epfl.ch/.
