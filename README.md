# pre-interview-notes
Notes and Resources on Machine Learning and Statistics


# Designing ML Systems
1. Handling missing values: MNAR, MAR (linked with another feature making it not-at-random), MCAR   
3. Feature hashing helps limit the # of features. Impact of collisions is random so better than "unknown" category
4. NNs don't work well with unit variance.
5. Position embeddings: either can be treated the same way as word or can have fixed pos. embedding (a case of fourier features) where we have a single column with a function like sin/cosine as a function of position p. When positions are continuous (e.g. in 3D coordinates)
6. Data leakage: sometimes the way data is collected or underlying process implicitly leaks output into features.
7. For time-series data, split by time to prevent leakage.
8. Use statistics from train split for scaling, imputaiton etc. (even EDA)
9. Groups (like records of same patient taken milliseconds apart) should be in same split.
10. Features (or sometimes gorups of features) having high correlation can indicate data leakage. Perform ablation studies.
11. Test performane should only be used for reporting, not for decision making.
12. feature stores != feature definion management
13. feature generalization to unseen data. e.g. commmend_id may be bad but user_id might be impportant.
14. Shouldn't just remove a feature based on coverage. Should look at relationship with output.
15. Feature scaling impacts gradient boosted trees
16. With smaller dataset, keep track of learning curve to see if it hints that more data will improve accuracy of itself or of a competing algorithm.
17. Basic assumptions (prediction assumption, iid, smoothness (values close together), tractable to compute p(z|x) in gen. models, boundaries like linear boundary for linear models, conditional idependences (features indepdenent of each other given class),
18. Ensembles increase the probability of correct prediction. bagging, boosting (weak learners with weighted data to focus on mistakes), stacking (meta-learner from base-learners)
19. ML models can fail silently.
20. Failures can be: theoretical constraincts (e.g. violation of assumptions), poor implementation, bad hyperparaemters, bad data, bad feature engineering,
21. Good practices to avoid problems. Start with simplest architecture (like one-layered RNN before expanding), overfit to training and evaluate on same (to make sure minimal loss achieved), set random seeds.
22. Distributed training:
- Data parallelism: split data and train models separaptely. reconcile gradients on central node. synchronous (have to wait) or asyncrhonous (update as they arrive). Async requires more time but when # of params is high, weight updates are sparse so staleness is less of an issue. Use smaller batch size on main worker node to even out compute requirements.
- Model parallelism: handle different parts of the model on diff machines. If assigning diff layers of NN to diff machines, not really parallelism.
- Pipeline parallelism: Reduce idle time when part of computation (micro-batches) is complete and passes results to other machine so it can start working.
- Phases: pre-ML/heuristics, simple models, optimize simple, complex models

# Resources

## Statistics

1. https://towardsdatascience.com/50-statistics-interview-questions-and-answers-for-data-scientists-for-2021-24f886221271


# Statistics:

## Tests:
- If effect size to be detected is very small, then to each a given power level, a greater sample size is required.
- In other words, if an effect size = X is observed, then if the sample size was huge, the probabilit of making a Type 2 Error (of not rejecting a false hypothesis) is less as compared to if the effect size X was observed using a small sample.
- Increasing sample size maintains probability of Type 1 error and decreases probability of Type 2 error. The latter makes sense based on the point discussed above.
- Type 1 Error is when you reject a true Null Hypothesis because the difference of the statistics lead to very small chances, that were too small for you to believe it was true (the p-value was too small based on the alpha you set, the chances that didn't matter for you). Regardless of the size of the sample that demonstrated that difference in the statistics, the alpha remains the same, although p-value gets smaller (increasing power).
- alpha is what you define. It's the probability that you'll make a type 1 error. This probability does not change even if you get smaller p-values based on a number of trials / samples. Alpha is the cut-off for p-value.
- Power of a test (1 - beta) depends on effect size, alpha and sample sizes of the two groups. https://medium.com/swlh/why-sample-size-and-effect-size-increase-the-power-of-a-statistical-test-1fc12754c322
- I have yet to come across an example where effect size where we are using a single group / sample to infer for the population. Maybe in the case when we are studying one sample, the other group refers to the poplulation. Maybe.
- Effect sizes like Cohen's D are essentially differences. It's just that they are converted to standardized form using a same standardization (assuming it is same for both groups) or a pooled standard deviation (from both groups). This standardization is actually questionable sometimes. https://rpsychologist.com/cohend/
- Chi-square independence of test: row percentage multiplied by column total (gives you expected frequency)

# Evaluation Metrics



1. Classification

TPR = Recall = Sensitivity = TP / ( TP + FN )

Out of all the actual positives, how many got detected. 
Maximizing TPR -> Minimizing FNR

Precision = TP  / ( TP + FP )

Out of the selected, how many are correct.

TNR = Specificity = TN / ( TN + FP )
Maximizing TNR -> Minimizing FPR

Out of all the negative selected elements, how many are actually negative

1 - Specificity = FPR = FP / ( FP + TN )

ROC Curve: TPR vs FPR

PR Curve: Precision vs Recall (TPR)

- AUC can be negative: if a classifier performs even worse than a random classifier
- Prefer PR-curves over ROC in cases where TNs are not of great interest and are huge thus skewing the results, for example, in Information Retrieval.

2. ML

http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/
https://towardsdatascience.com/top-30-data-science-interview-questions-7dd9a96d3f5c
https://towardsdatascience.com/over-100-data-scientist-interview-questions-and-answers-c5a66186769a#ba56
