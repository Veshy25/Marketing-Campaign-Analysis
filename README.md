# Marketing Campaign Analysis

Personal project: a Python data analysis of a grocery retailer's marketing campaign dataset.

I explore customer behaviour, product spending and campaign response to find insights for targeted marketing.

## Questions

1. Which product categories are most popular?
2. Which customer segments are more likely to respond to marketing campaigns?
3. Which purchase channel is most associated with campaign response?

## Data

`marketing_campaign.csv` (semicolon-separated) has 2,240 customers, who enrolled between 30/07/2012 and
29/06/2014. It appears to be the public Kaggle dataset
[Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).

Cleaning in `maincode.py`:
- Drop the 24 rows with a missing income (2,240 → 2,216).
- Remove income and birth-year outliers with the 1.5 × IQR rule (2,216 → 2,205). All results below use these
  2,205 customers.
- Group the marital status entries "YOLO", "Absurd" and "Alone" under "Single".

The dataset doesn't state a currency, so spend figures below are given as plain amounts.

## How to run

Tested with Python 3.9; the pinned versions support Python 3.9–3.12. Run from the repository folder, since the
script reads the CSV by relative path.

```bash
pip install -r requirements.txt
python maincode.py
```

## Outline of `maincode.py`

1. Importing libraries and the data
2. Basic data exploration, visualisation and basic cleaning
3. Data cleaning and transformation
4. Preliminary analysis and visualisation
5. Question 1: which product categories are most popular?
6. Question 2: which customer segments are most likely to respond to campaigns?
7. Question 3: logistic regression on which purchase channel predicts campaign response

## Findings

### Question 1: product popularity

I scored each category on two measures:
- **Reach:** the share of customers who bought it.
- **Depth:** the average spend per customer.

Each measure is scaled so the top category scores 1. The popularity score is the average of the two.

| Category | Customers who bought | Average spend | Popularity score |
|---|---|---|---|
| Wines | 99.4% | 306.17 | 0.997 |
| Meat | 99.95% (2,204 of 2,205) | 165.31 | 0.770 |
| Gold | 97.2% | 44.06 | 0.558 |
| Fish | 82.8% | 37.76 | 0.476 |
| Fruits | 82.2% | 26.40 | 0.454 |
| Sweets | 81.4% | 27.13 | 0.451 |

Wines is the flagship category: nearly every customer buys it, with the highest spend per customer. Meat
reaches virtually every customer but at about half the spend, so it's a stable, essential category rather than
the main revenue driver.

### Question 2: campaign response by segment

A customer counts as having **responded** if they accepted at least one of the six campaigns
(`AcceptedCmp1`–`5` and `Response`). Overall, **27.4%** responded (604 of 2,205).

Each segmentation below covers the whole customer base independently. A chi-square test checks whether the
response rate really differs across each segmentation's groups.

**Age (as at 2014, the latest enrolment year): no meaningful difference.** Rates range from 25.9% (Adult,
31–45) to 30.8% (Senior, 61+), with Young (up to 30) at 28.4% and Middle-aged (46–60) at 27.8%. The test gives
p = 0.42, so these gaps are consistent with chance, and age isn't a useful targeting variable here.

**Education: PhD customers respond most.** 33.0% of PhD customers responded (157 of 476), against 13.0% for
Basic (7 of 54). Response rates differ across education levels (p = 0.004), and PhD customers stand out against
everyone else (p = 0.002).

**Marital status: customers without a partner respond somewhat more.** Single (31.4%), divorced (30.0%) and
widowed (34.2%) customers respond more than married (25.3%) or cohabiting (25.2%) ones. Taken together, 31.3%
of customers without a partner responded (245 of 783), against 25.2% of those with one (359 of 1,422),
p = 0.003. Widows have the highest rate but are a small group (76 customers), and their rate isn't
distinguishable from the other customers' rate (p = 0.22).

### Question 3: purchase channels and campaign response

I ran a logistic regression of response on the number of web, catalogue and in-store purchases. The features
are standardised, so each coefficient is the change in the log-odds of responding for a one standard deviation
increase in that channel, holding the other two channels constant. The model was
trained on 70% of customers and tested on the other 30%.

| Channel | Coefficient | p-value |
|---|---|---|
| Catalogue purchases | 0.71 | < 0.001 |
| Web purchases | 0.26 | < 0.001 |
| Store purchases | −0.19 | 0.013 |

Customers who buy more through the catalogue are substantially more likely to accept campaigns. Web purchases
have a smaller positive association. Holding web and catalogue purchases constant, more in-store purchases are
associated with a slightly lower chance of responding. On its own, though, in-store buying goes with *higher*
response: responders average 6.60 store purchases against 5.53 for non-responders. So the negative coefficient is about extra
store purchases among customers with the same web and catalogue activity, not about in-store buyers in general.

These are associations, not causes: frequent catalogue buyers may simply be more engaged customers. The model
ranks customers reasonably well (ROC-AUC 0.754 on the test set). At the default 0.5 threshold, though, it
identifies only 21.5% of actual responders, so it works better for ranking customers than as a yes/no
targeting rule.
