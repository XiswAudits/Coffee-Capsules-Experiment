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

The question is which Premium price `P` and Regular price `R` maximise profit. Capsule costs are not in the data, so the programme maximises **revenue**. The only decision variables are the two prices. Each household buys a fixed quantity of the product it chooses, so revenue is linear in those prices.

The calculation lives in `price_optimisation.py` and is shown in the root `app.py` under **Revenue-maximising prices** (tabs **Shared prices** and **Per household**). Run it directly with:

```bash
python price_optimisation.py
```

The earlier successive linear programme of quadratic OLS demand has been retired. The app now has this one model.

### Fixed quantities

`d_i` is the average number of capsules bought on weeks when that household chose the product. Estimated from `coffee_capsules_data.csv`, not hard-coded:

| Household | d when buying Regular | d when buying Premium |
|---|---:|---:|
| Household 1 | 4.3333 (9 weeks) | 4.0000 (2 weeks) |
| Household 2 | 4.0000 (7 weeks) | 3.0000 (4 weeks) |
| Household 3 | never buys Regular | 4.2857 (7 weeks) |

### Lines in the R–P plane

A linear probability model (OLS of a 0/1 indicator on an intercept, `R`, and `P`) supplies the 0.5 contour. Premium is the side of a switching line where the fitted score is at least 0.5. A household that sometimes buys nothing also has a stop-buying line.

| Household | Fitted line | Reading |
|---|---|---|
| Household 1 | `P = 1.6030 R − 9.2621` | Always buys. Premium at or below the line, Regular above it. In-sample accuracy 11/11. |
| Household 2 | `P = 0.2305 R + 67.6909` | Always buys. Premium at or below the line, Regular above it. One week sits just on the wrong side (accuracy 10/11). |
| Household 3 | `P = 0.1205 R + 79.2906` | Never buys Regular. Still buys Premium at or below the line, and stops above it. Accuracy 11/11. |

These are the same pattern as a classroom sketch (Household 1 cares about the gap, Household 2 mostly about `P`, Household 3 has a reservation price) but the slopes come from this CSV.

### Shared linear programme

Every assignment of products is solved as its own LP with `scipy.optimize.linprog` (HiGHS). The solver minimises, so the objective vector is the **negated** quantity vector. The best feasible assignment on this sample is Household 1 Premium, Household 2 Regular, Household 3 Premium:

```text
max  4.0000 R + 8.2857 P
subject to
  Household 1 buys Premium:  P <= 1.6030 R − 9.2621
  Household 2 buys Regular:  P >= 0.2305 R + 67.6909
  Household 3 still buys:    P <= 0.1205 R + 79.2906
  0 <= R <= 65
  0 <= P <= 100
```

The upper bounds are the highest Regular and Premium prices in the experiment, so a price cannot run to infinity. Optimum: **R = €65.00**, **P = €87.12**, **revenue = €981.86**. It is a vertex: `R <= 65` and Household 3's buying line are both binding, and the constraints hold.

Other feasible assignments earn less (about €933, €868, €830, €640, and €542). Two assignments are infeasible inside the price box.

### Per-household programmes

Each household is also solved alone, on its own line and its own quantity:

| Household | Buys | Optimal R | Optimal P | Revenue |
|---|---|---:|---:|---:|
| Household 1 | Premium | €65.00 | €94.94 | €379.74 |
| Household 2 | Regular | €65.00 | €100.00 | €260.00 |
| Household 3 | Premium | €65.00 | €87.12 | €373.38 |

Household 2's revenue does not depend on `P` once they are kept on the Regular side, so every feasible Premium price on that edge earns €260. HiGHS returns the vertex `P = 100`.

`python price_optimisation.py` checks that each reported optimum is a vertex, satisfies `A_ub x <= b_ub`, and that revenue equals `c · x`.

The **Shared prices** tab draws all three lines on one graph (solid, dashed, long-dashed), shades the winning feasible region, draws the objective level, and labels `Optimal: P=…, R=…, Revenue=…`. The formulation sits under the graph. The **Per household** tab repeats that for each household and prints the equation and `d_i`. Sensitivity re-solves the shared LP after a ±10% shock to each line coefficient, moves each household's own prices by ±10%, and widens the price box. The regret matrix includes the shared optimum, each household optimum, and the slider scenario.

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
