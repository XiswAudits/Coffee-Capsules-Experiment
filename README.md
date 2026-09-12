# Coffee Capsule Demand Classifier

Interactive Streamlit website for the supplied coffee-capsule household dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Deploy

The app is deployment-ready for Streamlit Community Cloud. Put `app.py`, `requirements.txt`,
and optionally `coffee_capsules_data.csv` in a GitHub repository, then create a Streamlit app
pointing to `app.py`.

## Important modeling note

The supplied dataset has only 11 observations per household:

- Household 1: Regular + Premium, but no No Purchase observations.
- Household 2: Regular + Premium, but no No Purchase observations.
- Household 3: Premium + No Purchase, but no Regular observations.

Therefore, a separate three-class linear classifier cannot be honestly fitted for every household.
The app handles this explicitly rather than inventing an unsupported third zone.

For Household 1 and Household 2 it fits a binary Regular-vs-Premium logistic classifier and displays
the implied linear boundary `P_P = m * P_R + b`.

For Household 3 it shows the observed Premium/No-Purchase pattern and explains why a
Regular-vs-Premium boundary is not identifiable from the data.

The quantity estimate is the historical average quantity for the predicted choice.
