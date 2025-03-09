# Notes on Data Science, Machine Learning and Statistics


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
21. Failures can be: theoretical constraincts (e.g. violation of assumptions), poor implementation, bad hyperparaemters, bad data, bad feature engineering,
22. Good practices to avoid problems. Start with simplest architecture (like one-layered RNN before expanding), overfit to training and evaluate on same (to make sure minimal loss achieved), set random seeds.
23. Models with high bias may not improve from more training data but a high variance model might.
24. Distributed training:
  - Data parallelism: split data and train models separaptely. reconcile gradients on central node. synchronous (have to wait) or asyncrhonous (update as they arrive). Async requires more time but when # of params is high, weight updates are sparse so staleness is less of an issue. Use smaller batch size on main worker node to even out compute requirements.
  - Model parallelism: handle different parts of the model on diff machines. If assigning diff layers of NN to diff machines, not really parallelism.
  - Pipeline parallelism: Reduce idle time when part of computation (micro-batches) is complete and passes results to other machine so it can start working.
25. Phases: pre-ML/heuristics, simple models, optimize simple, complex models
26. When there's class imbalance, F1-score does not indicate how well model is doing as compared to random.
27. Baselines: random baseline (uniform, label distribution or naive), heuristic, human or BAU,
28. Evaluation Methods:
29. Perturbation tests, invariance tests (changing sensitive info should not change output like gender, directional expectation tests for sanity checks, model calibration for actual probabilities, slice-based evaluations to detect bias, or to make sure critical groups are prioritized, or to detect upstream (e.g. latency from mobile users) issues. Find slices by heuristics, error analysis, slice finder (clustering-based techniques).

30. Software degrades over time (software rot)
31. Batch prediction (async) for high trhoughput, for low latency: online (sync) with batch features, or online with batch and streaming (streaming prediction). hybrid when popular/expensive queries pre-processed in batch and others on-demand.
32. Latency and hardware can help decide between online vs batch and cloud vs edge.
33. In streaming, incoming features need to be stored in data warehouse in addition to streaming features to prediction service via real-time transport.
34. Batched doesn't necessarily mean less efficient. In fact, online means less resources wasted on datapoints not used (users never logged in).
35. Problem with batch is recency (if users' preferences can change significantly), also need to know datapoints in advance.
36. Can use feature store to ensure batch features used during training are equal to streaming features used in inference.
37. Decrease inference latency by: compression, inference optimization, improving hardware
38. Compression:
    - LoRa. high-dim tensors -> decompose to compact tensors. reduces number of parameters. e.g. compact convolution filters but not widely adopted yet.
    - Knowledge distillation -> teacher student (less data, faster, smaller) model. can also train both at the same time. both can have different architectures (tree vs NN).
    - Pruning: like removing sub-trees or zeroing params in NN (reduces # of non-zero params, reducing storage, improving computation). can introduce bias.
    - Quantization: trainiing (quantization-aware) or post-training in lower precision. If int (8 bits), it's called fixed precision. Quantization reduces memory footprint, storage requirements, faster computation. BUT means can only represent a smaller range of values, rounding/scaling up can result in errors, risk of overflowing/underflowing. Mixed-precision popular in traiing. Training in fixed not popular but standard in inference (like dege devices).
    - Input shape can be optimized for improving throughput and latency as well.

39. Cloud easier to start but more costs, requires nettwork connectivity, have higher latency, more privacy concerns but on-edge might drain device power
40. Compiling for hardware: Intermediate representatin is the middleman between framework and hardware. High-level, tuned, low-level IRs -> MACHINE CODE. IRs are computation graphs.
41. Model optimization:
    - Compiled lower IR could be slow because different frameworks (pandas, pytorch etc) optimized differently. For optimizing, use compiles that support optimization. Local (set of ops) and global (entire comp. graph) optimization. Vectorization (instead of loop), parallelization (process input chunks), loop tiling (hadware dependent, change data access order to match memory layout), operator fusion (veritcally for merging sequential ops or horizontally for parallel ops that share same inputs to fuse ops to avoid redudant memory accesses). for exmaple, two consective 3 x 3 convs CBR (Convolution, bias, RELU) have a receptive field of 5 x 5.
    - Hand-designed fusion can be manual, depends on hardware and expertise. ML-based helps estimate cost by generating ground truth for cost estimation model and then predicting time for each sub-graph and then an algo determine the most optimal path to take.
    - In browsers, can use js but it's limited and very slow. WASM is an open-source standard which we can use to compile models. WASM is faster than js but still slow.
42. Failures: Software system (deployment, downtime, harware, depency). ML-specific (train-serving skew, data/trends change over time,
43. Degenerate feedback loops (where model outputs influence future system inputs). Especially common in cases of natural labels. That's popular movies keep getting more popular. Also known as  “exposure bias,” “popularity bias,” “filter bubbles,” and sometimes “echo chambers.”. Can magnify bias and lead to systems performing sub-optimally.
  - detect by measuring diversity of items. or bucketing items and measuring model performance. once online, if outputs become more homogenous, most liekly feedback loop.
  -  correct by:
    -  randomization to reduce homogeneity, like tiktok seeds traffic for a new video randomly to decide whether to promote/demote. improvement in diversity comes at a cost of accuracy. "Contextual bandits as an exploration strategy" can make recommendations more "fair" for content creators.
    -  if position matters (like pos. of song on spotify), it can affect feedback. can train the model with position encoded and then mark as 1 during inference. another approach is to train 2 models, 1 model predicts probability that an item will be seen and considered given position, other predicts if an item will be clicked given they considered.
44. Data Distribution Shifts
  - Covariate, P(X) changes but P(Y|X). During development, because of selection bias, up/down-sampling, algo (like active learning). In prod, environment changes.
  - Label (prior) shift, P(Y) changes but P(X|Y). Sometimes covariate can result in label shfit.
  - Concept Shift (posterior), P(Y|X) changes but P(X). Same input, diff output. Usualyl cyclic/seasonal. Can use diff models to deal with seasonal drifts.
  - Feature change (schema or range of values)
  - Detection: natural/labels will help determine if model perfformance degrading. P(X), P(y), P(Y|X), P(X|Y). Y here is ground truht but if not available, at least look at predictions.
  - Can use summary stats but not a guaruantee. Use 2 sample test. Statistical != practical signf. Might only be worth worryinga about if diff detected with small sample size. E.g KS test but only works on 1D data.
  - Time-series shifts also possible, time window selection will make a different. should track data post production. Shorter intervals  can lead to "alert fatigue" but can help detect issues faster. Cummulitive can hide trends which sliding window stats can uncover.
  - Retrain using massive dataset in the hope issue is prevented and/or retrain at a certain cadence. Retraining can be stateless or fine-tuning. Feature selection can impact need for frequency of retraining. Can also divide model into sub-models where some are trained more frequent.
45. Observability is part of monitoring. Log important metrics related to network (latency, throughput), hardware (cpu utilization, uptime), ml-related.
  - ML-related: raw inputs, features, predictions, accuracy. from harder to easier to monitor. from "less likely to be caused by human errors" to be more closer to business.
  - accuracy: explicit feedback or inferred/natural labels. if not ground truth, at least secondary metrics can help detect degradation.
  - predictions: ground truth may have lag. monitor output to see if something is weird (like more 0s). If model is same, diff output can be due to diff input.
  - features: summary stats, business rules etc.
  - 3 pillars of monitoring: logs, metrics, traces.
  - distributed tracing - each process has ID s.
  - log analysis (ml-based anomaly detection, prob. of other services being affected.
  - dashboards are helpful for monitoring for both engineers and non. but too many metrics/dashboards can lead to "dashboard rot".
  - Alerts - alert policy (trigger, duration, suppression), channels, description
  - Monitoring is about tracking outputs. Monitoring makes no gurantee it will help you find out what went wrong. Monitoring assumes its possible to run tests and let data pass through system to narrow down the problem. Observability makes a stronger assumption that internal states can be inferred using outputs. Allows more fine-grained metrics. Observability and Interpretability go hand-in-hand.
  - Monitoring/obs. is essentially passive.
46. Continual learning:
  - about setting up infrastructure.
  - learning with every sample can lead to catastrophic forgetting. less efficient as hardware designed for batch processing, unable to exploid data parallelism. instead update with micro-batch (task dependent).
  - replica from current champion model. evaluate challenger vs champion.
  - can do stateless or stateful (fine-tuning, incremental learning). stateful requires less data, allows faster converage, less compute. with stateful, can train from scratch every now and then to realibrate the model. can also combine stateful and staeless models using techniques like **parameter server.**
  - Data iteration vs model iteration. Model iteration mostly needs stateless but can explore knowledge transfer and model surgery.
  - `

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
