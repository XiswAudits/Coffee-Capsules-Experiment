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
