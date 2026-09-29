# Coffee Capsule Demand & Choice Models

Interactive Streamlit applications built from the 11-week coffee-capsule household experiment.

## Applications

### 1. Original demand classifier

The original app is the root `app.py`. It uses a two-stage pooled logistic classification approach to visualise Buy/No Purchase and Regular/Premium boundaries.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Independent household choice model — MNL

The new discrete-choice application is `choice_model_mnl/app.py`.

It estimates **one independent multinomial logit model for each household**, with three competing alternatives:

- Regular
- Premium
- No Purchase

It keeps zero-purchase weeks as genuine demand observations, estimates joint choice probabilities, shows pairwise utility boundaries, and adds a zero-inclusive expected-demand layer.

For the full methodology, equations, assumptions, identification warnings, and deployment instructions, see [`choice_model_mnl/README.md`](choice_model_mnl/README.md).

## Revenue-maximising prices

The question is which Premium price `P` and Regular price `R` maximise profit. Capsule costs are not in the data, and quantities are not choice variables, so the programme maximises **revenue** and the only decision variables are the two prices. That is the same as maximising profit only when marginal cost is zero or already sunk.

The calculation lives in `price_optimisation.py` and is shown in the root `app.py` under **Revenue-maximising prices**. Run it directly with:

```bash
python price_optimisation.py
```

### Demand

For each household separately, two OLS fits on that household's 11 weeks:

```text
Q_R = a_R + b_RR R + b_RP P
Q_P = a_P + b_PR R + b_PP P
```

Coefficients are estimated from `coffee_capsules_data.csv`. They are not hard-coded. Revenue is the quadratic `R Q_R + P Q_P`.

### Linear programme

`scipy.optimize.linprog` minimises, so the objective vector is the **negated gradient** of revenue. Because the true objective is quadratic, each solve is a successive linear programme: at a feasible reference price, revenue is replaced by its first-order Taylor expansion, the LP is solved inside the feasible polygon cut by a trust box around that reference, and the step is kept only when the true quadratic revenue rises. One LP over the whole polygon would sit at a corner; the trust box is what lets an interior revenue hill (Household 1) be reached. The number reported as maximum revenue is the quadratic revenue at the LP prices, not the linear surrogate.

### Constraints

`A_ub x <= b_ub` and the solver bounds are rebuilt for every household from that household's OLS output:

- predicted Regular quantity `>= 0` (otherwise the household stops buying Regular)
- predicted Premium quantity `>= 0`
- `R >= 0`, `P >= 0`
- an upper guardrail at the observed maximum price plus one sample standard deviation, so a price cannot run to infinity when the quadratic is not concave

On this sample the default guardrail is about **R ≤ €73.89** and **P ≤ €110.29**.

### Separate household programmes

Households do not share coefficients, so they do not share the demand boundaries. The primary result is one `(R, P, revenue)` per household. A single shared menu is also solved, as a comparison, by stacking every household's demand constraints and maximising the sum of revenues. The sum of the separate optima is only available if each household can be charged its own prices.

With the default guardrail the successive LP matches the exact quadratic maximum on the same polygon:

| Household | Optimal R | Optimal P | Max revenue | Predicted Q regular | Predicted Q premium |
|---|---:|---:|---:|---:|---:|
| Household 1 | €72.13 | €92.26 | €288.21 | 0.10 | 3.05 |
| Household 2 | €73.89 | €97.23 | €295.57 | 4.00 | 0.00 |
| Household 3 | €73.89 | €48.72 | €467.33 | 0.00 | 9.59 |

The sum of those separate optima is **€1,051.11**. One shared menu is about **R €59.05**, **P €68.79**, total revenue **€854.68**.

Household 1 is an interior maximum of a concave revenue function, just above the highest observed Regular price (€65). Households 2 and 3 are not concave in Regular price: optimal Regular sits on the upper guardrail, and widening that guardrail moves their solution. Household 3 never bought Regular, so those OLS slopes are zero and the optimum extrapolates Premium below the experimental prices. Treat the prices as an exploratory calculation on 11 weeks, not as a validated tariff.

The root app plots each household's feasible region (shaded polygon, demand-constraint lines, labelled optimum), a scenario evaluation at the price sliders, the guardrail sensitivity, and regret. Regret is the household's own maximum revenue minus revenue at another menu (another household's optimum, the shared menu, the experiment mean, or the sliders). An infeasible menu books no revenue, so regret equals the whole optimum. Charging a household its own LP prices has regret zero.

## Dataset

The shared `coffee_capsules_data.csv` contains 11 weekly observations for each of three households, including Regular/Premium prices and household quantities. Zero quantities are retained because they represent observed No Purchase behaviour.

## Streamlit deployment

For the MNL application, create a Streamlit Community Cloud app using:

```text
choice_model_mnl/app.py
```

The application reads the shared dataset from the repository and uses the root `requirements.txt`, which includes SciPy for maximum-likelihood numerical optimisation.

## Important limitation

The independent MNL design is intentionally household-specific, but each household has only 11 observations. Household 1 and Household 2 never observed No Purchase, while Household 3 never observed Regular. An unconstrained MNL can therefore experience separation and produce unstable or divergent estimates. The implementation uses finite optimisation bounds and explicitly flags parameters that reach a bound.

The resulting model should be treated as an exploratory choice-modelling tool, not as validated causal elasticity, willingness-to-pay, or out-of-sample forecasting evidence.
