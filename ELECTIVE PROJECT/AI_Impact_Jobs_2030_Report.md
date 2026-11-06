# AI Impact on Jobs 2030 - Written Report
### Machine Learning Activity: Logistic Regression

---

## Group Members and Roles

| Name | Role |
|------|------|
| Kenneth Gasmen | Research Lead: background research, dataset documentation, written conclusion |
| Mark Tarayao | Data Lead: data cleaning, exploration, and preparation |
| Jomer Ignacio | Modeling Lead: model building, evaluation, and results interpretation |

**AI Usage Disclosure:** Claude was used to help structure the notebook and format the code. All written explanations, analysis, and conclusions in this report are our own work and a we use help for expounding our explanations with ai.

**Dataset:** [AI Impact on Jobs 2030](https://www.kaggle.com/datasets/khushikyad001/ai-impact-on-jobs-2030) by Khushi Yadav, Kaggle. MIT License.


## Part 1: Research the Dataset and Its Topic

### Q1. What is the name of your dataset, where did you get it, and who collected it?

The dataset is called **"AI Impact on Jobs 2030"**. We downloaded it from Kaggle, where it was published by user **Khushi Yadav** (username: khushikyad001) under an MIT license. It is a synthetically generated dataset. This means its values were created by an algorithm to look realistic, not collected from actual employers or workers. The dataset has **3,000 rows and 18 columns**. Each row represents a fictional job profile with a salary, education requirement, skill scores, an AI exposure index, and an automation risk category.

---

### Q2. What real-world problem does this dataset help answer, and why does it matter?

This dataset is built around one question: **which types of jobs are most likely to be automated by AI by the year 2030, and what makes a job more or less at risk?**

This is an important question right now. Automation has already removed whole categories of jobs that once employed millions of people, such as assembly line operators, toll collectors, and bank tellers. AI is now pushing that same pressure into office work that used to seem safe. The big difference this time is speed. It took decades for factory machines to change manufacturing. AI is changing legal research, financial analysis, and customer service within just a few years.

Workers who do not see displacement coming have no time to prepare. Governments that ignore the pattern cannot build support systems fast enough. The World Economic Forum estimates that over 85 million jobs could be displaced globally by 2025, while 97 million new roles appear that need very different skills. Tools that identify which jobs are most at risk give workers, employers, and policymakers time to respond before serious harm is done.


### Q3. What do outside sources say are the main factors affecting automation risk?

**Source 1: World Economic Forum - Future of Jobs Report 2023**
https://www.weforum.org/reports/the-future-of-jobs-report-2023

The WEF says the highest-risk jobs are those built on **routine and repetitive cognitive tasks** such as data entry, scheduling, document processing, and standard customer interactions. The more predictable and structured a job's tasks are, the easier they are to program into a machine. The report also says that **AI adoption speed by industry** is a major factor. Industries that are actively using AI tools like Finance, Manufacturing, and Retail tend to automate roles faster than industries where human judgment is harder to replace, such as healthcare or primary education. The WEF projects that 44% of workers' core skills will be disrupted within five years.

**Source 2: McKinsey Global Institute - Jobs Lost, Jobs Gained (2017, updated)**
https://www.mckinsey.com/featured-insights/future-of-work/jobs-lost-jobs-gained

McKinsey identifies three main factors. First, **skill type**: jobs that require physical work in unpredictable settings, like plumbing or surgery, or jobs that depend heavily on social judgment, are harder to automate than jobs that process structured information. Second, **education level**: lower-education roles often cluster around repetitive tasks, though this is not always the case since even highly educated financial analysts face exposure. Third, **industry and sector**: manufacturing and food service face far higher exposure than management consulting or clinical medicine. McKinsey estimates that roughly 60% of occupations have at least 30% of their tasks that could technically be automated with current technology. Whether they actually get automated depends on cost and social factors, not just what is technically possible.

---

### Q4. Hypothesis: Most Important Features Before Running the Model

Based on what the research above tells us, we predict these features will matter most before we run any model:

1. **Automation_Probability_2030** - This is a numeric score from 0.0 to 1.0 representing the estimated chance of the job being automated. It should have the strongest relationship with the Risk_Category target because both are measuring automation likelihood, just in different ways.
2. **AI_Exposure_Index** - Jobs in areas with more AI exposure face more active replacement pressure. This should be a strong positive predictor of High Risk.
3. **Average_Salary** - Higher-paid jobs usually need harder-to-automate skills. We expect salary to be a negative predictor, meaning higher salary = lower risk.
4. **Education_Level** - Jobs that only need a high school diploma tend to involve more routine tasks, making them easier targets for automation.
5. **Years_Experience** - More experienced workers tend to hold more complex roles. Experience may work as a small negative predictor of risk.

We will check these predictions against the real model results in Part 4, Question 9.

---

### Q5. Ethical Risks and Concerns

**Synthetic data does not show real inequality.** This dataset was generated by an algorithm. It cannot show that in the real world, workers in lower-income countries face automation with far fewer resources to adapt. They have no retraining programs, no strong unemployment safety nets, and little access to the new jobs being created. A model trained on this data gives false confidence about how evenly the impact will be spread.

**Risk of making bad decisions with this model.** If a company uses a model like this to decide who gets retraining and who gets let go, they could displace workers based on a probability score that was never meant to drive individual decisions. A job labeled High Risk might still be done better by a specific experienced person than any AI tool available. The label ignores that.

**Salary and education carry historical bias.** In the real world, who earns a high salary and who has a PhD are not random outcomes. They reflect decades of unequal access to education and opportunity. Using these as features can underestimate risk for already-privileged groups and overestimate it for already-disadvantaged workers.

**Compressing three categories into two loses real information.** Our binary target treats Medium Risk the same as Low Risk. A job at Medium Risk is in a genuinely different position than one at Low Risk. Workers at Medium Risk who fall into our class 0 may get no warning from a system using this model, which is a problem on its own.

---

## Part 2: Understand and Prepare the Data

### Q1. How many rows and columns? Data dictionary.

The dataset has **3,000 rows and 18 columns**. There are no missing values in any column and no duplicate rows.

| Column | Type | Description |
|--------|------|-------------|
| `Job_Title` | Categorical | Name of the profession, for example "Software Engineer", "Security Guard", or "Financial Analyst" |
| `Average_Salary` | Numeric | Estimated average annual salary in USD. Range: $30,030 to $149,798. Mean: $89,372. |
| `Years_Experience` | Numeric | Years of experience needed for the role. Range: 0 to 29. Mean: 14.7 years. |
| `Education_Level` | Categorical (ordinal) | Minimum education required: High School, Bachelor's, Master's, or PhD |
| `AI_Exposure_Index` | Numeric | A score from 0.0 to 1.0 measuring how much the job is exposed to AI tools and workflows. Mean: 0.50. |
| `Tech_Growth_Factor` | Numeric | A multiplier for the technology growth rate in this job's field. Mean: 1.01. |
| `Automation_Probability_2030` | Numeric | Estimated probability from 0.0 to 1.0 that this job will be automated by 2030. Mean: about 0.50. Strongly linked to the target. |
| `Risk_Category` | Categorical | The automation risk label: Low, Medium, or High. This becomes our binary target. |
| `Skill_1 through Skill_10` | Numeric | Ten skill scores from 0.0 to 1.0. They represent unnamed skill dimensions related to automation resistance. |

---

### Q2. What is the target column, which model did you choose, and why?

**Target:** The `Risk_Category` column has three values: Low, Medium, and High. We converted it into a binary column called `High_Risk`:
- `High_Risk = 1` if `Risk_Category` is "High" (740 rows, 24.7%)
- `High_Risk = 0` if `Risk_Category` is "Low" or "Medium" (2,260 rows, 75.3%)

We chose this split because the activity asks for a yes/no prediction. "High Risk" represents a meaningfully different situation from the other two categories. It is the group of workers who face urgent, near-term displacement. Someone at Medium Risk has more time to adapt. Someone at High Risk needs action now. Putting Low and Medium together into a single "not urgent" class reflects that practical difference.

**Model chosen: Logistic Regression.**

Logistic Regression is the right model here because the target is binary. It calculates the probability that a job profile belongs to the High Risk class using the sigmoid function. This keeps all outputs between 0 and 1, making them directly readable as probabilities. Linear Regression would be wrong for this because it predicts any number on a continuous scale with no limits, so it cannot produce valid probability estimates for a yes/no outcome.

---

### Q3. Missing values, duplicates, and outliers.

**Missing values:** None. All 18 columns across all 3,000 rows are fully filled in. No filling or replacement was needed.

**Duplicates:** None. The dataset has no repeated rows.

**Outliers:** The numeric columns show no extreme outliers based on the descriptive statistics. Salary ranges from $30,030 to $149,798 with a standard deviation of $34,608, which is a reasonable spread for a synthetic global dataset. The skill scores (Skill_1 through Skill_10) are all between 0.0 and 1.0 by design. AI_Exposure_Index and Automation_Probability_2030 are also bounded between 0 and 1. Because Logistic Regression is not very sensitive to feature scale, no outlier removal was done. StandardScaler was applied before modeling to bring every feature to a zero mean and unit variance, which handles any scale differences between columns.

---

### Q4. Which features are categorical and how were they encoded?

| Column | Encoding Method | Reason |
|--------|----------------|--------|
| `Education_Level` | Ordinal: High School=0, Bachelor's=2, Master's=3, PhD=4 | There is a clear meaningful order. A PhD represents more education than a Bachelor's. Assigning ordered integers preserves that relationship. |
| `Job_Title` | Dropped | There are too many unique job titles to encode usefully, and they work like row identifiers. Keeping them would not help the model generalize to new data. |
| `Risk_Category` | Dropped (source of target) | The binary target `High_Risk` was created directly from this column. Keeping it as a feature would mean the model is just reading the answer. That is data leakage. |
| All other columns | No encoding needed | Already numeric. They were passed directly to StandardScaler before modeling. |

After encoding, the feature matrix has **16 columns** and **3,000 rows**.

---

### Q5. Visualizations: what does each one show?

**Visualization 1: Risk_Category distribution and binary target balance**
*(viz1_target_distribution.png)*

The dataset has 1,521 Medium Risk jobs (50.7%), 740 High Risk (24.7%), and 739 Low Risk (24.7%). After converting to binary, we have 740 High Risk (24.7%) and 2,260 Not-High-Risk (75.3%). The classes are unbalanced at roughly a 3 to 1 ratio. This matters for evaluation because a model that never predicts High Risk would still score 75.3% accuracy without learning anything useful. Accuracy alone cannot tell us whether the model is doing its job.

**Visualization 2: Salary distribution by risk category**
*(viz2_salary_by_risk.png)*

The salary histograms for Low, Medium, and High Risk jobs overlap heavily. This shows that salary alone cannot separate the classes. There are High Risk jobs paying over $100,000 and Low Risk jobs paying under $50,000. However, the High Risk distribution does lean slightly lower on average, which is consistent with our hypothesis that salary is a weak negative predictor of risk.

**Visualization 3: Proportion of High Risk jobs by education level and growth projection**
*(viz3_industry_risk.png)*

Because this dataset does not include an industry column, this visualization shows the proportion of High Risk jobs grouped by Education Level categories. It confirms that certain job profiles cluster more heavily in the High Risk class and that the grouping is not perfectly clean, which is expected from a synthetic dataset.

**Visualization 4: Correlation heatmap of numeric features**
*(viz4_heatmap.png)*

`Automation_Probability_2030` has a very strong positive correlation with `High_Risk` at r = 0.757. This is by far the strongest relationship in the dataset. `Skill_1` shows a small positive correlation at r = 0.035, and `Average_Salary` shows a small negative correlation at r = -0.033. All other features have correlations close to zero, meaning they have almost no linear relationship with the target on their own. No pair of features showed severe multicollinearity. All inter-feature correlations stayed well below 0.8.

**Visualization 5: Boxplots by risk class**
*(viz5_boxplots.png)*

The boxplot of `Automation_Probability_2030` by class shows a clean separation. High Risk jobs cluster at high probability values between 0.7 and 1.0. Low/Medium Risk jobs cluster at lower values between 0.0 and 0.5. For `Average_Salary` and `Years_Experience`, the two class boxes overlap heavily. This confirms that these features contribute much less to the model's decisions on their own.

---

### Q6. Which features are strongly related to the target? Is there multicollinearity?

The feature most strongly related to `High_Risk` is **`Automation_Probability_2030`** with a correlation of r = 0.757. This makes sense because the dataset was synthetically built in a way where `Risk_Category` was likely created partly from this probability score.

`Skill_1` shows the next highest correlation at r = 0.035, followed by `Average_Salary` at r = -0.033 and `AI_Exposure_Index` at r = 0.017. All other feature correlations with the target are essentially zero.

For multicollinearity, we checked all pairs of features in the heatmap. No two features had a correlation above 0.8, so multicollinearity is not a problem here. The 10 skill scores are independent of each other by design, and the other numeric features show no strong linear relationships with each other.

One important point: we dropped `Risk_Category` completely before modeling because the target came from it. Keeping `Automation_Probability_2030` also raises a concern. It is so closely tied to the target that it could be seen as a form of near-leakage. However, we kept it because it is a distinct numeric measurement, not the label itself. Its presence explains why the model performs so well on synthetic data.

---

### Q7. Describe the target distribution. How balanced are the two classes?

- **Low/Medium Risk (0):** 2,260 rows, 75.3%
- **High Risk (1):** 740 rows, 24.7%

The classes are **unbalanced** at roughly a 3 to 1 ratio. A classifier that always predicts class 0 and never identifies any job as High Risk would score 75.3% accuracy without learning anything. That is the naive baseline our model needs to beat.

Our model achieves 99.83% accuracy, which is 24.5 percentage points above that baseline. More importantly, it achieves 99.32% Recall on the High Risk class, meaning it almost never misses a truly high-risk job. A naive always-predict-majority model would score 0% on Recall.

Because the classes are unbalanced, we look at Recall and Precision alongside accuracy. A high accuracy score alone does not tell us whether the model is actually learning to identify High Risk jobs.

---

## Part 4: Logistic Regression

### Q1. Problem statement

> We use a job profile's numeric features, including its automation probability score, AI exposure index, average salary, years of experience required, education level, and ten skill scores, to predict **whether a profession will be classified as High automation risk by 2030**.

---

### Q2. Why can't we use Linear Regression? What does the sigmoid function do?

Linear Regression predicts a number on an unlimited scale. Our target `High_Risk` is not a number. It is a category with only two values: 0 or 1. If you use Linear Regression on a binary target, you get predictions like 1.4 or -0.2, which have no meaning as probabilities and cannot be converted to a class label in any reliable way.

The **sigmoid function** is what makes Logistic Regression work for this problem. It takes any real number produced by the linear part of the model and squashes it into the range between 0 and 1:

```
P(High_Risk = 1) = 1 / (1 + e^(-(b0 + b1*x1 + b2*x2 + ... + b16*x16)))
```

The result is a probability. An output of 0.91 means the model thinks there is a 91% chance the job is High Risk. An output of 0.12 means a 12% chance, which would be classified as Not High Risk. We then apply a threshold, by default 0.50, to turn that probability into a final class label. This is why Logistic Regression is the right tool for binary classification.

---

### Q3. What threshold did you use, and what happens when you change it?

The default threshold is **0.50**. If the predicted probability is 0.50 or above, the job is classified as High Risk.

Results from the threshold comparison in Cell 17:

| Threshold | Predicted High Risk | Accuracy | Recall | Precision |
|-----------|--------------------:|----------|--------|-----------|
| 0.3 | more | lower | higher | lower |
| 0.4 | more | 98.50% | 99.32% | 94.84% |
| **0.5 (used)** | **148** | **99.83%** | **99.32%** | **100.00%** |
| 0.6 | fewer | lower | lower | higher |
| 0.7 | fewer | lower | lower | higher |

At threshold 0.5, the model achieves perfect Precision at 1.0000. Every job it calls High Risk actually is High Risk. Recall is 99.32%, meaning only one truly High Risk job was missed across all 600 test cases.

Lowering the threshold to 0.4 did not improve Recall at all in this case. Recall stayed exactly the same at 99.32% while Precision dropped and Accuracy fell. This happened because the model's probability scores are very decisive. Most predictions are either near 0.0 or near 1.0, with very few borderline cases. Changing the threshold did not recover the one missed job. It only added false positives. For this dataset, the default threshold of 0.5 is the best choice.

---

### Q4. Coefficients: what do they mean?

**Model Intercept:** -8.4144. This is the base log-odds of being High Risk when every feature is at its average value after scaling. A large negative intercept makes sense here because 74.7% of jobs are Not High Risk, so the starting probability should be low.

**Top Feature Coefficients:**

| Feature | Coefficient |
|---------|------------|
| Automation_Probability_2030 | +10.1096 |
| Average_Salary | -0.1838 |
| Skill_9 | -0.1251 |
| Skill_2 | +0.1177 |
| AI_Exposure_Index | +0.0879 |

**Feature 1: Automation_Probability_2030 (Coefficient: +10.1096)**

This is the most influential feature by a large margin. A coefficient of +10.11 after scaling means that for every one standard deviation increase in a job's automation probability score, the log-odds of being classified as High Risk increase by 10.11. In plain terms: if a job has a high automation probability score, the model will almost certainly call it High Risk. This single feature drives nearly all of the model's decisions. The odds ratio is exp(10.11), which equals approximately 24,600. That is an enormous number. It tells us this feature and the target are measuring nearly the same thing from two different angles.

**Feature 2: Average_Salary (Coefficient: -0.1838)**

Salary is a small negative predictor. Higher-paid jobs are slightly less likely to be classified as High Risk. The effect is very small compared to the automation probability score. Salary provides some information at the edges, but it is not what determines the outcome. This is consistent with McKinsey's finding that higher pay correlates with more complex tasks, but salary is not a reliable protection against automation by itself.

---

### Q5. Confusion Matrix

From Cell 19, on the 600-row test set:

|  | Predicted Low/Med Risk | Predicted High Risk |
|--|------------------------|---------------------|
| **Actually Low/Med Risk** | TN = 452 | FP = 0 |
| **Actually High Risk** | FN = 1 | TP = 147 |

**True Positives (TP = 147):** Jobs that are genuinely High Risk and the model correctly identified them. These are the 147 workers who would get a timely warning under any policy that uses this model.

**True Negatives (TN = 452):** Jobs correctly identified as Not High Risk. No unnecessary alarm for these 452 workers.

**False Positives (FP = 0):** The model labeled zero Low/Medium Risk jobs as High Risk. It produced no false alarms at all in the test set.

**False Negatives (FN = 1):** Exactly one truly High Risk job was predicted as Not High Risk. This is the most serious error. In a real deployment, this worker would receive no warning and could lose their job without any time to prepare.

---

### Q6. Accuracy, Precision, Recall, F1, and which matters most

| Metric | Value | What it means for this problem |
|--------|-------|-------------------------------|
| **Accuracy** | 99.83% | 599 out of 600 test predictions were correct |
| **Precision** | 1.0000 (100%) | Every job the model labeled High Risk actually was High Risk. Zero false alarms. |
| **Recall** | 0.9932 (99.32%) | Of the 148 truly High Risk jobs in the test set, the model caught 147 of them |
| **F1-Score** | 0.9966 | Near-perfect balance between Precision and Recall |

**Recall matters most for this problem.** A False Negative means telling a high-risk worker they are safe. That worker then loses their job with no preparation time. That is a direct, serious harm. A False Positive means warning a safe worker unnecessarily. They prepare for something that does not happen, which is inconvenient but not a lasting harm. Because missing someone is worse than a false alarm, we prioritize Recall. Our model achieves 99.32% Recall, missing only 1 out of 148 High Risk jobs in the test set.

---

### Q7. Class imbalance and the baseline accuracy trap

Our dataset has 75.3% Low/Medium Risk and 24.7% High Risk. A model that always predicts class 0 and never identifies any High Risk job would score **75.3% accuracy** without learning anything. That is the naive baseline.

Our model achieves **99.83% accuracy**, which is 24.5 percentage points above that baseline. More importantly, it achieves 99.32% Recall on the minority class. A naive majority-class model would score exactly 0% on Recall.

This is why accuracy alone is not enough when classes are unbalanced. The naive baseline passes a surface-level accuracy check at 75.3% while being completely useless at the actual task. Our model passes both the accuracy test and the Recall test, which together show that it is genuinely learning to identify High Risk jobs.

---

### Q8. Training vs testing: is there overfitting?

| Set | Accuracy |
|-----|---------|
| Training | 99.71% |
| Testing | 99.83% |
| Gap | 0.12% |

The testing accuracy is actually slightly higher than the training accuracy. This is the opposite of overfitting. A gap of 0.12 percentage points is essentially nothing.

The 5-fold cross-validation scores from Cell 16 confirm this. The fold scores were 1.0000, 0.9896, 0.9938, 0.9896, and 0.9854. The mean was 99.17% and the standard deviation was 0.49%. A low standard deviation means performance is consistent across all five test folds. No single fold was unexpectedly better or worse than the others. This is strong evidence that the model generalizes well to data it has never seen.

Logistic Regression is naturally resistant to overfitting because it is a simple linear model with built-in regularization. The regularization parameter C=1.0 penalizes large coefficients, which prevents the model from fitting too tightly to the specific training data.

---

### Q9. Most important features: does this match the hypothesis?

From Cell 18, the five features with the largest absolute coefficients are:

| Rank | Feature | Coefficient | Effect |
|------|---------|-------------|--------|
| 1 | Automation_Probability_2030 | +10.1096 | increases risk |
| 2 | Average_Salary | -0.1838 | decreases risk |
| 3 | Skill_9 | -0.1251 | decreases risk |
| 4 | Skill_2 | +0.1177 | increases risk |
| 5 | AI_Exposure_Index | +0.0879 | increases risk |

**Checking against our Part 1 hypothesis:**

Automation_Probability_2030 was confirmed as the dominant predictor by an enormous margin. We expected it to be important, but we did not expect it to be 55 times stronger than the next feature.

Average_Salary was confirmed as a small negative predictor. Higher-paid jobs are slightly less likely to be High Risk, which matches the McKinsey research on salary and task complexity.

AI_Exposure_Index was confirmed as a positive predictor. Jobs with higher AI exposure in their field face greater risk, which is consistent with the WEF findings.

Education_Level did not appear in the top five. It did not add much predictive power beyond what Automation_Probability_2030 already captures.

Skill_9 and Skill_2 appeared in the top five, which we did not predict. Without knowing what these unlabeled skill dimensions actually measure, we can only say one of them increases risk and the other decreases it. We cannot give a meaningful real-world explanation for them.

The biggest takeaway is how completely Automation_Probability_2030 controls the model. Every other feature operates at coefficients below 0.2, which is small enough to be nearly irrelevant. The model is essentially a one-feature classifier with some minor adjustments from the other 15 columns.

---

### Q10. Improvement: threshold tuning before and after

We tested whether lowering the threshold from 0.5 to 0.4 would improve Recall, since catching every High Risk job is our main priority.

| Metric | Threshold 0.5 | Threshold 0.4 | Change |
|--------|---------------|---------------|--------|
| Accuracy | 99.83% | 98.50% | down 1.33% |
| Recall | 99.32% | 99.32% | no change |
| Precision | 100.00% | 94.84% | down 5.16% |
| F1-Score | 99.66% | 97.03% | down 2.63% |

The threshold change did not help. Recall stayed exactly the same while Precision and Accuracy both dropped. This happened because most of the model's predictions are already very confident. Probabilities cluster near 0.0 or near 1.0, with very few cases in the middle range between 0.4 and 0.5. Lowering the threshold did not recover the one missed High Risk job. It only created new false positives from jobs the model was already assigning low-confidence probabilities.

The conclusion here is that the default threshold of 0.5 is already optimal for this dataset. The model is precise enough that threshold tuning does not improve anything. The strategy was reasonable, but the data did not have enough borderline cases to benefit from it.

---

## Part 5: Reflect and Conclude

### Q1. Summary of findings

Our Logistic Regression model predicts whether a job profile will be classified as High automation risk by 2030 with **99.83% accuracy** on the held-out test set. This is 24.5 percentage points above the naive baseline of always predicting the majority class. The model achieved a Recall of 99.32%, meaning it correctly flagged 147 out of 148 genuinely High Risk jobs in the test set, missing only one. Precision was perfect at 100%, meaning every job it called High Risk actually was High Risk. Training and testing scores differ by only 0.12%, and 5-fold cross-validation produced a mean of 99.17% with a standard deviation of 0.49%. The model is not overfitting.

The dominant predictor was `Automation_Probability_2030` with a coefficient of +10.11. This single feature drives nearly the entire prediction. All other features contribute only at the margins.

---

### Q2. Why Logistic Regression and not Linear Regression?

We chose Logistic Regression because the target is binary. A job is either High Risk or it is not. Linear Regression is designed for continuous numeric outputs and would produce values like 1.4 or -0.2 that cannot be interpreted as class memberships.

If we had used Linear Regression on this dataset, we would have predicted `Automation_Probability_2030` as a continuous value from 0.0 to 1.0. Our evaluation metrics would have been MAE, RMSE, and R², measuring how close our predicted scores are to the real probability values. The model would answer "how probable is automation for this job?" rather than "is this job High Risk or not?" Those are related but different questions. We would also lose the confusion matrix, the Recall and Precision framework, and the ability to adjust a decision threshold.

---

### Q3. Key difference between Linear and Logistic Regression

**Linear Regression** predicts a continuous number. It fits a line through the data that minimizes the total squared error between predicted and actual values. It is evaluated by how far off its number predictions are from the real numbers.

Example from our dataset where Linear Regression would apply: Predicting `Average_Salary` from `Years_Experience`, `Education_Level`, and `AI_Exposure_Index`. The output is a dollar amount on a continuous scale.

**Logistic Regression** predicts the probability of belonging to one of two classes. It applies the sigmoid function to the linear combination of features to keep all outputs between 0 and 1. It then assigns a class using a threshold. It is evaluated by how correctly it assigns classes, not by how close a number is to a target value.

Example from our dataset: Predicting whether a job is High Risk (1) or Not High Risk (0). The output is a probability like 0.94, which we convert to class 1 using a 0.5 threshold.

Their evaluation metrics are different because they answer different types of questions. R² measures how much variance in a continuous outcome the model explains. That is meaningless for a classifier. Recall measures how many true positive cases the model found. That is meaningless for a regression predicting salary.

---

### Q4. Limitations of the model

**1. The data is synthetic, and so are the patterns.**
Every row was created by an algorithm, not observed from real workplaces. The dominant predictor, `Automation_Probability_2030`, was almost certainly built to directly determine `Risk_Category` during data generation. In real-world data, the relationship between job characteristics and automation outcomes would be much messier and far less clean. A model that scores 99.83% accuracy on this synthetic dataset would likely perform much worse on actual employment data.

**2. The model depends almost entirely on one feature.**
`Automation_Probability_2030` has a coefficient 55 times larger than the next most important feature. This is effectively a single threshold rule dressed up as a multi-feature model. It does not tell us which real-world job characteristics actually matter for automation risk. It just reads a pre-assigned score.

**3. The binary target drops useful information.**
We merged Medium Risk and Low Risk into one class. A worker at Medium Risk is in a different situation than one at Low Risk, but both get the label 0. A worker at Medium Risk who the model confidently classifies as 0 may have a false sense of safety.

**4. No industry, location, or time information.**
This dataset has no industry column, no country data, and no time-series structure. Real automation risk varies enormously across these factors. A manufacturing worker in Germany faces a very different regulatory and economic situation than a customer service agent in the Philippines. This model captures none of that.

**5. The skill columns have no labels.**
Skill_1 through Skill_10 have no description. We cannot say what they measure. We know Skill_9 decreases risk slightly and Skill_2 increases it slightly, but we cannot explain what those skills are or give any actionable advice based on them.

---

### Q5. Do we trust this model to make real decisions?

No.

The model performs well on the synthetic test set, but that is because the synthetic data was built with clean, simple rules that a logistic model can read easily. The 99.83% accuracy is not proof that the model has learned anything useful about the real job market. It learned to classify according to the rules baked into the data generator.

If a government program used this model to decide which workers receive retraining support, they would be acting on a system that has never seen a real displacement event, real labor market data, or real workers. The one False Negative in our test set represents one worker who would receive no warning. At larger scale, even a 0.7% false negative rate means thousands of vulnerable workers get nothing.

The model is useful for learning how Logistic Regression works, for exploring which synthetic features correlate with risk, and as a starting point for thinking about what a real model would need. It is not ready to drive decisions that affect people's livelihoods.

---

### Q6. What would we do next with more time and data?

**Get real data.** The Bureau of Labor Statistics O*NET database and the WEF Future of Jobs survey both contain real occupation-level information about task composition, employment trends, and automation exposure. Testing against real data would show whether the patterns here hold in practice.

**Use a richer feature set.** The biggest missing pieces in this dataset are industry type, country, task composition details, and union and regulatory context. In the real world, these factors change the automation picture dramatically.

**Handle class imbalance more carefully.** With 75.3% in class 0 and 24.7% in class 1, we could use SMOTE (Synthetic Minority Oversampling Technique) or class weighting in the model to ensure equal learning from both classes.

**Try a model that handles non-linear patterns.** Logistic Regression works here because the synthetic data is linear and clean. Real-world data is not. A Gradient Boosted classifier or a Random Forest would capture interaction effects that a linear model misses.

**Make individual predictions explainable.** For a tool meant to inform workforce policy, being able to explain each prediction is not optional. SHAP values would let us show exactly which features pushed a specific job toward High Risk or away from it, making the model's reasoning visible and checkable.

---

### Q7. Individual reflections

**Kenneth Gasmen - Research Lead:**
I did not expect AI to be replacing jobs this fast. When I read the WEF report saying 44% of skills will be disrupted in five years, that kind of hit different. I always thought automation was a far future thing but it is already happening now. The hard part for me was finding research that actually matched our dataset since ours had no industry or country info, so I had to work with what we had and be honest about the gaps.

**Mark Tarayao - Data Lead:**
Honestly the dataset was way cleaner than I expected. No missing values at all which made my job easier but also made me think this is clearly not real data. I chose to put Medium Risk with Low Risk because when I think about it practically, Medium Risk workers are not in immediate danger while High Risk workers are, so the split made sense to me. Looking at the charts was more helpful than staring at numbers. The boxplot for Automation_Probability_2030 already showed the two classes separating clearly before we even trained the model.

**Jomer Ignacio - Modeling Lead:**
What got me was how one feature basically ran the whole model. The coefficient for Automation_Probability_2030 was +10 and everything else was below 0.2. So it felt less like a real machine learning model and more like the dataset was just designed that way. The cross validation part was useful though because seeing the scores stay consistent across five different splits told me the model was not just getting lucky on one specific test set. That part I actually understood and found useful.

**Hardest challenge as a group:**
The notebook kept breaking because we assumed the column names before actually checking them. The Automation_Risk column turned out to have text values like High, Low, and Medium instead of numbers, and calling .median() on text crashed everything. Once we figured that out and handled the text properly, the rest of the code ran fine.

---

## Sources

1. Khushi Yadav. *AI Impact on Jobs 2030*. Kaggle, 2024.
   https://www.kaggle.com/datasets/khushikyad001/ai-impact-on-jobs-2030

2. World Economic Forum. *Future of Jobs Report 2023*.
   https://www.weforum.org/reports/the-future-of-jobs-report-2023

3. McKinsey Global Institute. *Jobs Lost, Jobs Gained: Workforce Transitions in a Time of Automation*, 2017.
   https://www.mckinsey.com/featured-insights/future-of-work/jobs-lost-jobs-gained-what-the-future-of-work-will-mean-for-jobs-skills-and-wages

4. Scikit-learn. *Logistic Regression Documentation*.
   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
