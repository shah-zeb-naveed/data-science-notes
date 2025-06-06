# Practical Insights on Machine Learning, Data Science and System Design

This repository is a curated collection of real-world insights on machine learning system design, MLOps best practices, scalable inference, and model deployment. Whether you're building a recommendation engine, optimizing real-time prediction latency, or exploring online learning and A/B testing strategies, these notes cover practical ML challenges rarely addressed in textbooks. Ideal for data scientists, ML engineers, and AI practitioners looking to bridge the gap between research and production.

📈 Keywords: machine learning deployment, scalable inference, MLOps notes, ML engineering system design, production ML, FAISS real-time search, Redis caching ML, feature store tips, inference latency optimization.

**Favorite Resources:**
- MLOps by Andrew NG
- ML System Design Book by Chip Hyuen 
- Grokking the ML Interview - Educative.io
- https://developers.google.com/machine-learning/guides/rules-of-ml
- https://developers.google.com/machine-learning/recommendation


# ML System Design Framework

- Clarify/frame/define scope in terms of usage, integration, current state/boostrapping
  
- Actors (entities, input, context)
  
- Requirements:
   - Functional requirements:
      - how will it be used, real-time, latency/QPS, resources/timelines, SLAs etc. Incorporating new users/items often a consideration.
   - Non functional:
      - which components will overload, scalable (ask or assume # of datapoints in each stage), available, latency, observability (monitoring, tracing, logging, MLOps)
      - perf. and capacity, both during training time (data size/capacity/), inference (SLA)
      - complexities of ML models (training, inference, sample (how hungry it is))
      - distribute workload (e.g. N documents to be ranked) to multiple model shards to meet SLA but resources are limited and comm will have its own latency

- Metrics: ML objective aligned with business
   - Offline, component-wise e.g. ndcg
   - Online  and end-to-end/downstream system metrics e.g. user's engagement metrics.
   - identify/track negative actions/metrics.
   - Downstream business metrics (DAU, session time, etc.)
   - Consider normalization of metric by e.g. daily active users. 

- Modelling approaches
   - Objectives can be broken into multiple models.
   - Break down complex problems (e.g. multi-stage recommender), identify inputs/outputs of each stage
      - can create multiple propensity models (like, comment, share) and weighted avaerage using business-defined importances.
      - funnel approach: increasing model complexity down the road
         - helps achieve SLA
         - helps in error analysis.
         - index (inverted)/bool/collab filtering for fast-retrieval
  - common issues (cold start, etc.)

- Dataset Generation
   - sources: crowdsourcing, existing systems, SMEs, or build a product in a way that collects training data
   - time span: decide based on problem making assumptions to make sure enoguh data/patterns is captured
      - can be picked using experimentation.
   - schema: triplet generation (user, pos, neg) e.g recommendation engine.
   - Balance / Negative Sampling
      - CTR metric is an online/business metric. Often 1% so expect negative samples.
     
- Label generation:
   - heuristics (programming/LFs), hand labelling, negative sampling (specific definition of a negative interaction), etc.
   - often "middle" examples that we are uncertain about or have weak signals can be ignored.
   - semi-supervision/active learning
     
- Feature engineering
   - Types:
      - aggregated
      - delayed
      - raw (real-time/on-demand)
      - online features (precomputed)
      - precomputed embeddings
      - histograms
      - time decay/windows
      - normalization: by metric for fair comparison, (timestamp into UTC), PII (can group critical info) etc.
   - embedding generation:
      - upstream model, neural network (dynamically generated (activations) or using weights (static))
         -  part of main network or trained separately with positive/negative pair
         -  word2vec -> doc2vec approach: embedding by averaging and then cosine
      - PCA over one-hot
      - node2vec for graphs
      - hashing trick
      - embedding from metadata if column too high cardinal and low samples.
      - hybrid e.g. hashing + embedding from metadata, etc.

- Feature Store
   - Maintain entity-wise/time_window-wise feature tables e.g user_features_table, product_feautres_table
   - Some features (especially on-demand) may only come from raw data (might not make sense to make them a part of feature table e.g. trip_distance, weather).
   - On-demand features can be logged in UC.
   - Create training data using primary keys and timestamp to ensure point-in-time correctness
   - Can publish to an online store like Azure Cosmos DB (or dbx managed) for real-time inference
     

- Splitting logic:
   - Often predicting future so time-based split avoids data leakage and unbiased estimate. backtesting.
      - Simulate a production system. Can give an idea of how frequentyl to re-train

- Inference:
   - aggregator service/load balancer.
   - can route to different model based on query e.g. ml model for ios vs android.
   - multiple application servers behind load balancer, in case of multi-phase models, manage flow.
  
- deployment strategies
   - green/blue, canary, A/B, B/A, shadow test, bandits, interleaving

- Retraining logic/frequency (online, batch, trigger-based)
   - log model output and get inferred (implicit or explicit) label. Feed back to training data for subseuquent runs.
   - online joiner for streaming features
   - ensure feedback loop

- ci/cd

- Software Engineering / Large-Scale ML Systems
   - caching: cache features instead of predictions if entities are too many. embeddings often cached in key-value store.
   - REST: stateless, supports caching
   - Scale app server and model services separately
   - each model service as a pod on k8s for autoscaling. kube-proxy enables inter-service communication

   - latency target: 200ms (total). # of components might mean 50 ms per service
   - Databases
      - leverage kafka / rdbms / blob storage  
      - indexing
      - sharding: geo-based sharding (also reduces latency), range-based, hash-based. so one server is not overloaded. cross-shard queries expensive.
      - read replicas handle read requests while main handles writes, helps in consistency through replication
      - terminology: "server" is sharded. "data" is partitioned. (commonly but people use interchangably)
      - parittioning is about logical structures. could be located on same shard or spread across.
   - Archive older data in cold storage
   - thiink about a parent service which might call the ML service. e.g. a general search service calling ranking service.

---

# Notes

1. Requirements of ML systems: availability, reliability, scalability, maintainability, adaptability
2. Types of ML problems:
   - multilabel classicication: 1 model for all classes with binary vector as output [1, 0, 1] or 1 model for each class where output vector only has one 1 [1], [0]
   - Multiple ways to solve a problem: if using User + Environment features for outputting a score for each app where output vector size based on current N apps -> can't scale for new apps but instead, use User + Env + App features (for all available apps) and output a single value.
   - Multiple objectives: one approach is to join different losses in one objective and set their co-efficients but this requires retraining on each choice of co-efficient combo. Instead can train different models for different losses and than create scores based on heuristics.
3. Data Engineering
   - binary files are compact
   - row vs columnar tables, good for inserts or reading rows (csvs), parquet good for columnar analytics
      - column oriented for training pipeline keeps costs low and high throughput.
   - NOSQL: document-type have rare relationhips. graph-based suit better for such cases.
   - OLTP: transactional, fast writes so users dont wait
     - ACID: definitons vary.
       - atomicity, entire transaction is sucessful or fails completely
       - consistency, data follows pre-defined rules. 
       - isolation, transactions that happen at the same time don't change data at the same time
       - durability, once stored, db failure doesn't matter
     - if not ACID, usually BASE (basically available, soft state and eventual consistency)
   - OLAP: for analytical needs.
   - both OLAP/OLTP become outdated. New is Lakehouse (best of data warehouse and data lake)
   - ELT instead of ETL so transformations done by apps that pull from warehouses and ELT process doesn't care about source schema changes
   - Data passing b/w services (aka request driven). Having different services for different components of the company's ML needs.
   - data broadcasted to a broker (called events in event-driven hub)
4. Sampling
     - Non-probaiblity sampling (convenience, snowball (start small and then increase), juedgement (experts decide what to use), quota sampling (custom defined groups). Signifciant selectio bias.
     - Simple random (minoirty class might be sampled out compltely)
     - Stratified (maintain proportions, may not be opossible in multi-label)
     - Weighted sampling (each data point has assigned probability isntead of uniform. e.g. recent data might be more important)
     - Reservoir sampling (good for streaming data, leetcode question also.
     - Importance samploing (when main distribution not accessible/exensive. use proposal distribution. e.g. using old policy to calcualte rewards in reinforcemnet learning)
     - for balancing, keep test/val set intact
        - balancing can result in uncalibrating probabilities so use with caution
5. Training data:
   - drop useless data (e.g. bot impressions)
6. Features:
   - onehot encoder: high compute/dims for high cardinality
   - time decay and multiple windows to capture short term, long term, recency patterns
   - transfer learning
     - NNs learn latent features hierarchically. Can retrain weights of a model pre-trained on one domain for another. OR can use featurization to extract features for a new model.
  - Handling missing values: MNAR, MAR (linked with another feature making it not-at-random), MCAR   -
  - Feature hashing helps limit the # of features. Impact of collisions is random so better than "unknown" category. Or can group into "Other".
  - Feature crossing: like lat x long to define city blocks. hashing is more ciritcal now.
  - can apply clipping before scaling/min-max.
  - NNs work well with unit variance????
  - Embeddings:
      - project item/user in same space.
      - Position embeddings: either can be treated the same way as word or can have fixed pos. embedding (a case of fourier features [sins of diff frequencies]) where we have a single column with a function like sin/cosine as a function of position p. When positions are indefinite/continuous (e.g. in 3D coordinates)
      - d=(D)^1/4 is one heuristic for embedding
      - Contextual: ELMo (bi-RNN), BERT (bi-Trans)
      - task-based embedding specialized but requires more samples/compute
      - take inspiration from word2vec -> doc2vec. items belonging to a user can be averaged to get user embeddingf
  6.1 Features (or sometimes gorups of features) having high correlation can indicate data leakage. Perform ablation studies.
  6.2 introduce non-linearities in linear models using hand-engineering feature interactions 
  6.3 Test performane should only be used for reporting, not for decision making.
  6.4 feature stores != feature definion management
  6.5 feature generalization to unseen data. e.g. commment_id may be bad but user_id might be impportant.
  6.6 Shouldn't just remove a feature based on coverage. Should look at relationship with output.
  6.7 Feature scaling impacts gradient boosted trees (according to author. not according to chatgpt)
7 Transfer learning:
    - extract layers and use as features, train a few layers (if less data and more commonolaties), train more layers or train entire netowrk by warm starting with pre-trained model
8. Labels: label multiplicity when diff labels by annotaters. natural labels from data. natural label based on user behavior aka behavioral labels. can start with hand labels, natural labels or and explicit feedback.
  - programmaetic labels (weak supervision, labelling  functions based on heuristics). could be keyword, regular exp, database lookups etc. can use multiple LFs. small hand labels used for guidance/evaluation. can be used in different projects. why ml? because LFs might not cover all samples so can use ML for such samples. Can also use this as a starting point for producitonizing ML while ground truth labels are collected.
    - leverage programmatic data labelling platforms like Snorkel (allows uses labelling functions which they call as "weak supervision", collaboration b/w domain xperts.
  - semi-supervision: various methods. can use ml to predict on unlabelled and then retraining on confident samples. repeat the process. OR, use KNN/clustering approach. OR purturb samples to create artificial samples. leave signifcant eval set and and then continue training the champion on all data.
  - acitve learner (a model) sends unlaballed samples to human oracles (less confiedent)
9. feedback loop length: sometimes controllable. shorter means faster but might be noisy.
10. Class Imbalance:
  - model might be stuck in non-optimal solution, won't find signal for rare class and/or the cost of misprediction for rare class might be more important.
  - Sampling:
    - under/over/both sampling or smote
    - recommended: can use twophase learning (where we use original data for fine-tuninga fter training model with sampled/balanced data).
      - some argue we shouldn't fix class imbalance as deep NNs can learn patterns
  - Algorithmic methods. Modify cost (manually define cost matrix, or use class-balanced loss or focal loss (penalize model where it's more wrong). Ensemlbes can also prove robust
  - right evaluation metrics are important:
    - accuracy/f1 might be high but misleading. AUC curve?  Accuracy might still be a good metric for a particular class.
    - F1-related metrics focus on positive class. It is asseymetric as depends on what our "positive" class is.  use PR curve instead: does not get skweed because of high TNs)
11. Data augmentation
  - just like CV, NLP can benefit from augmentation (like templaing, or replacing words with synonyms, perturbation (which can also help make models robust to noise, adverserial attacks
  - analyze error rate to decide what kinds of inputs need augmentation, e.g. different types of noise in the background
  - GANs can be used to synthesize data e.g. converting sunny to rainy
  - Adding data might hurt if overrepresent noisy/synethetic data and model is simple (high bias)
13. Data leakage: sometimes the way data is collected or underlying process implicitly leaks output into features e.g. a feature indirectly links to output.
14. Split:
    - For time-series data (not necessarily "forecasting"), split by time to prevent leakage.
    - Use statistics from train split for scaling, imputaiton etc. (even EDA)
    - Groups (like records of same patient taken milliseconds apart) should be in same split.
    - make sure patterns like seasonality are captures within each set
    - stratified sampling
    - remember to retrain on all data before deployment/experimentation
12. Modelling:
  - Regression (linear/logistic) can be implemented by an MLP
  - remove bias (e.g. popularity) by exploration/exploitation strategy, choosing the right metrics, etc.  
  - With smaller dataset, keep track of learning curve to see if it hints that more data will improve accuracy of itself or of a competing algorithm.
  - Basic assumptions (prediction assumption, iid, smoothness (values close together), tractable to compute p(z|x) in gen. models, boundaries like linear boundary for linear models, conditional idependences (features indepdenent of each other given class),
  - Ensembles increase the probability of correct prediction. bagging, boosting (weak learners with weighted data to focus on mistakes), stacking (meta-learner from base-learners)
  - Loss Functions:
     - normalized log loss (baseline avg. prediction) for rare events (makes less sensitive to background CTR/base CTR/naive CTR)
     - quantile loss instead of predicting average (expected value) for over/under-estimating
       

29. Distributed training:
  - Data parallelism: split data and train models separaptely. reconcile gradients on central node. synchronous (have to wait) or asyncrhonous (update as they arrive). Async requires more time but when # of params is high, weight updates are sparse so staleness is less of an issue.
    - Use smaller batch size on main worker node to even out compute requirements.
  - Model parallelism: handle different parts of the model on diff machines. If assigning diff layers of NN to diff machines, not really "parallelism"
  - Pipeline parallelism: Reduce idle time when part of computation (micro-batches) is complete and passes results to other machine so it can start working.
25. Phases of ML Project: pre-ML/heuristics, simple models, optimize simple, complex models
26. Evaluation
  - When there's class imbalance, F1-score does not indicate how well model is doing as compared to random.
  - Baselines: random baseline (uniform, label distribution or naive), heuristic, human or BAU, SOTA/OS
  - Evaluation Methods:
    - Perturbation tests, invariance tests (changing sensitive info should not change output like gender, directional expectation tests for sanity checks, model calibration for actual probabilities, slice-based evaluations to detect bias, or to make sure critical groups are prioritized, or to detect upstream (e.g. latency from mobile users) issues. Find slices by heuristics, error analysis, slice finder (clustering-based techniques).

30. Software degrades over time (software rot)
31. Inference
  - Batch prediction (async) for high trhoughput, for low latency: online (sync) with batch features, or online with batch and streaming (streaming prediction). hybrid when popular/expensive queries pre-processed in batch and others on-demand.
  - An endpoint can serve batch (precomputed predictions). batching requests for efficiency is not same as batch predictions.
  - Latency and hardware can help decide between online vs batch and cloud vs edge.
  - ![image](https://github.com/user-attachments/assets/8ed114ae-e81c-49ed-8c37-a1e585a1b742)
  - In streaming, incoming features need to be stored in data warehouse in addition to streaming features to prediction service via real-time transport.
  - Batched doesn't necessarily mean more efficient. In fact, online means less resources wasted on datapoints not used (users never logged in).
  - Problem with batch is recency (if users' preferences can change significantly), also need to know datapoints in advance.
  - Can use feature store to ensure batch features used during training are equal to streaming features used in inference.
  - Decrease inference latency by: compression, inference optimization, improving hardware
  - Cloud easier to start but more costs, requires nettwork connectivity, have higher latency, more privacy concerns but on-edge might drain device power

40. Hyperparameter Tuning:
    - Hyperopt, skopt: model-based sequential tuning
    - random grid search will find optimal faster because of sampling
41. Compression:
    - LoRa. high-dim tensors -> decompose to compact tensors. reduces number of parameters. e.g. compact convolution filters but not widely adopted yet.
    - Knowledge distillation -> teacher student (less data, faster, smaller) model. can also train both at the same time. both can have different architectures (tree vs NN).
    - Pruning: like removing sub-trees or zeroing params in NN (reduces # of non-zero params, reducing storage, improving computation). can introduce bias.
    - Quantization: trainiing (quantization-aware) or post-training in lower precision. If int (8 bits), it's called fixed precision. Quantization reduces memory footprint, storage requirements, faster computation. BUT means can only represent a smaller range of values, rounding/scaling up can result in errors, risk of overflowing/underflowing. Mixed-precision popular in traiing. Training in fixed not popular but standard in inference (like dege devices).
    - Input shape can be optimized for improving throughput and latency as well e.g image resolution

44. Model optimization:
  - Compiling for hardware: Intermediate representatin is the middleman between framework and hardware. High-level, tuned, low-level IRs -> MACHINE CODE. IRs are computation graphs.
  - Compiled lower IR could be slow because different frameworks (pandas, pytorch etc) optimized differently. For optimizing, use compiles that support optimization. Local (set of ops) and global (entire comp. graph) optimization. Vectorization (instead of loop), parallelization (process input chunks), loop tiling (hadware dependent, change data access order to match memory layout), operator fusion (veritcally for merging sequential ops or horizontally for parallel ops that share same inputs to fuse ops to avoid redudant memory accesses). for exmaple, two consective 3 x 3 convs CBR (Convolution, bias, RELU) have a receptive field of 5 x 5.
  - Hand-designed fusion can be manual, depends on hardware and expertise. ML-based helps estimate cost by generating ground truth for cost estimation model and then predicting time for each sub-graph and then an algo determine the most optimal path to take.
  - In browsers, can use js but it's limited and very slow. WASM is an open-source standard which we can use to compile models. WASM is faster than js but still slow.
46. Failures:
  - Software system (deployment, downtime, hardware, dependency) OR
  - ML-specific (train-serving skew, data/trends change over time)
    - Data Distribution Shifts
      - Covariate, P(X) changes but P(Y|X).
        - During development, because of selection bias, up/down-sampling, algo **(like active learning???).**
        - In prod, environment changes.
      - Label (prior) shift
        - P(Y) changes but P(X|Y). _Sometimes_ covariate can result in label shfit.
      - Concept Shift (posterior), P(Y|X) changes but P(X)
        - Same input, diff output. Usualyl cyclic/seasonal. Can use diff models to deal with seasonal drifts.
      - Feature change (schema or range of values)
      - Detection:
        - natural/labels will help determine if model perfformance degrading. P(X), P(Y), P(Y|X), P(X|Y). Y here is ground truht but if not available, at least look at predictions.
        - Can use summary stats but not a guaruantee. Use 2 sample test. Statistical != practical signf. Might only be worth worryinga about if diff detected with small sample size. E.g KS test but only works on 1D data.
        - Time-series shifts also possible, time window selection will make a different. should track data post production. Shorter intervals  can lead to "alert fatigue" but can help detect issues faster. Cummulitive can hide trends which sliding window stats can uncover.
      - Retrain using massive dataset in the hope issue is prevented and/or retrain at a certain cadence.
      - Retraining can be stateless or fine-tuning.
      - Feature selection can impact need for frequency of retraining.
      - Can also divide model into sub-models where some are trained more frequent.
      - update eval set (or continous monitoring) so recent data changes can be captured.
      - change can be gradual/sudden. user data, generally has slower drift. businesses may change more rapidly.  
  - Degenerate feedback loops (where model outputs influence future system inputs). Especially common in cases of natural labels. That's popular movies keep getting more popular. Also known as  “exposure bias,” “popularity bias,” “filter bubbles,” and sometimes “echo chambers.”. Can magnify bias and lead to systems performing sub-optimally.
    - detect by measuring diversity of items. or bucketing items and measuring model performance. once online, if outputs become more homogenous, most liekly feedback loop.
    -  correct by:
      -  randomization to reduce homogeneity, like tiktok seeds traffic for a new video randomly to decide whether to promote/demote.
      -  improvement in diversity comes at a cost of accuracy.
      -  Exploration/Exploitation or "Contextual bandits" can make recommendations more "fair" for content creators.
  -  if position matters (like pos. of song on spotify), it can affect feedback.
    **-  can train the model with position encoded and then mark as 1 during inference.
    -  another approach is to train 2 models, 1 model predicts probability that an item will be seen and considered given position, other predicts if an item will be clicked given they considered.???**

51. Error Analysis:
    - ML models can fail silently.
    - Failures can be: theoretical constraincts (e.g. violation of assumptions), poor implementation, bad hyperparaemters, bad data, bad feature engineering,
    - Good practices to avoid problems
      - Start with simplest architecture (like one-layered RNN before expanding), sanity checks, overfit to small training set and evaluate on same (to make sure minimal loss achieved), set random seeds.
    - Models with high bias may not improve from more training data but a high variance model might.
    - Use strategic approach to decide what to tackle first to get highest ROI. e.g. focus on strata giving most error, can further see what specific characterists
    - collecting more data is expensive so do error analysis to confirm if needed/prioritized.
    - estimate bias and variance helps determine next steps
    - analyze errors between training/dev/test sets to determine if there's a variance problem, data mismatch or avoidable bias by comparing to human-level performance. irreducible = bayes error.
    - chain of assumptions in ML: tuning params for lower training error, regularization/bigger train set for lower dev error, bigger dev set for lower test eror, change dev set or cost function for lower real world errors
    - early stopping: one knob affects training and dev so Andrew Ng doesn't like it
    - test set accuracy is not enough. set of disproportionately important examples needs to perform really well.
    - HLP (human level performance) can't beat that. caution if ground truth for HLP is itself by human.
      
45. Observability: is part of monitoring.
  - 3 pillars of monitoring: logs, metrics, traces.
  - Monitoring is about tracking outputs. Monitoring makes no gurantee it will help you find out what went wrong. Monitoring assumes its possible to run tests and let data pass through system to narrow down the problem. Observability makes a stronger assumption that internal states can be inferred using outputs. Allows more fine-grained metrics. Observability and Interpretability go hand-in-hand.
  - Monitoring/obs. is essentially passive.
  - Log important metrics related to network (latency, throughput), hardware (cpu utilization, uptime), output metrics (# user repeat input, # user switches to typing, ctr, prediction dist etc.)
    - ML-related: raw inputs, features, predictions, accuracy. **from harder to easier to monitor. from "less likely to be caused by human errors" to be more closer to business.???**
  - accuracy: explicit feedback or inferred/natural labels. if not ground truth, at least secondary metrics can help detect degradation.
  - predictions: ground truth may have lag. monitor output to see if something is weird (like more 0s). If model is same, diff output can be due to diff input.
  - features: summary stats, business rules etc.
  **- distributed tracing - each process has IDs.??**
  - log analysis (ml-based anomaly detection, prob. of other services being affected)
  - dashboards are helpful for monitoring for both engineers and non. but too many metrics/dashboards can lead to "dashboard rot".
  - Alerts - alert policy (trigger, duration, suppression), channels, description

46. Continual learning:
    - The gain from having fresh data can be determined by running experiments on historic data. Similar whether to do data/model iteration depends on compute/performance gain.
  - a lot about setting up infrastructure.
  - learning with every sample can lead to catastrophic forgetting. less efficient as hardware designed for batch processing, unable to exploid data parallelism. instead update with **micro-batch???** (task dependent).
  _- replica from current champion model. evaluate challenger vs champion._
  - can do stateless or stateful (fine-tuning, incremental learning).
    - stateful requires less data, allows faster convergane, less compute. with stateless, can train from scratch every now and then to realibrate the model. can also combine stateful and staeless models using techniques like **parameter server.???**
  - Data iteration vs model iteration. Model iteration mostly needs stateless but can explore **knowledge transfer and model surgery.???**
  - Combat data distribution shifts and continuous cold start problem (where data does not exist for even existing users because of access styles). Might have to do prediction in real-time within a few minutes on user's first session e.g. tiktok.
  - Challenges of continual leanring
    -  Natural labels often need to be extracted from logged behavioral activities. For faster access, should tap into RT transport instead of waiting for data to arrive in data warehouse (dbx deltalake has milliseconds latency so not nsure if it's accurate).
    - retraining frequency can be limited if A/B tesitng is needed and enough sample size will be reached onyl after a certain time like rare events.
    - some algos (like matrix and tree) based aren't designed or efficient for partial learning. e.g. colab filteirng will require building matrix, performing dim reduction which can be expensdive if done frequently. NNs/logistic are a better choice.
    - need to consider feature preprocessing like scaling as well. running statistics are supported by sklearn but are slow. if computed individual for differenet subsets, they can vary a lot and model trained on one subset might not generalize well.
      - use global statistics (at cost of accuracy). optimized framworks instead of sklearn partial_fit.
    - 4 phases of continual learning:
        1. manual/stateless (adhoc retraining, focus on new mdoels),
        2. automated retraining (gut feeling or idle compute, or even experimenets, need to automate entire process from pulling data, generating labels to deploying, need Scheduler, Data, Model store),
        3. Automated/stateful training (but still on a fixed schedule)
        4. Continual learning (based on model degrapdation/shifts
           - even edge deployments can continually retrain without requiring networking with central server ->  better privacy).

47. Deployment:
 - if updated model on new data distribution, test on recent data "backtesting" in "addition" to static test set. recent data could be corrupted.
 - shadow deployment: expensive coz doubles inference cost
 - a/b testing: ensure randomness in traffic routing. calculate sample size based on power analysis (MDE, alpha rate, power or 1-beta, expected variance). Historic growth will define how long to run the experiment for. Error rate (0.05) means we might just pick a wrong model by chance.
   - if model 1 (A and B) is upstream for model 2, then keep switching A and B (like on alternating days). 
   - **also possible in batch predictions. it is stateless (does not consider model's current performance).???**
 - back testing: b/a test to validate a/b results if it's too optimistic (sanity check)
 - long-running a/b test: to measure user retention. **can also be done via backtest???**
 - Canary release (safe-rollout): simialr to A/B testing but doesn't have to be random. e.g. releasing to less critical/low risk markets first.
 - Interleaving, instead of spliting user groups, serve both models to each user and measure user preferences. May not be tied to actual core performance metrics (not sure what was meant by this statement). Need to make sure there is no position bias, by encoding position or by picking A or B with equal probability. 
 - bandits: each model a slot machine. requests served to the model with best current performance while also exploring along the way. maximizing predicition accuracy for users. requires online, preferable short/explicit feedback, a mechanism to keep track of each model's current performance and routing requests. similar to exploration-exploitation strategy in reinforcement leanring. e.g. e-greedy. Other exploration algos include Thompson Sampling, Upper Confidence Bound.
   - require less data but are complex
   **- Contextual bandits as an exploration strategy - contextual bandits determine payout of each action like item. can have partial feedback problem (badnit feedback). less adopted in industry except top-tech.???**
     
48. Infra/Tooling
  - storage/compute. we cannot stop 1 container in 2-container pod. in addition to RAM, bandwidth/IO is also important. ops are measured in FLOPS (floating point ops per second). If 1 million FLOPS hardware but app/job runs 0.3, then utilization is 30%. but since it's definition is ambiguous, it's not very useful. Often vCPU used (approx. half of physical core). for companies that grow by a lot, cloud costs can be as high as 50% of their revenue which makes them go back to private data centers (cloud repatriation). Companies use hybrid approach. Multi cloud is also popular to avoid "vendor lock-in". Standardizing dev env is critical and cloud-based envs help.
  - resource management. pre-cloud era demanded resource utilization. with cloud/elastic, can just scale up esp. if engineers prodictivity is more important.
  - A Scheduler helps cron by bringing in dependeny management. An Orchestrator is concerned with "where" to get thos eresources. "Schedulers deal with job-type abstractions such as DAGs, priority queues, user-level quotas (i.e., the maximum number of instances a user can use at a given time), etc. Orchestrators deal with lower-level abstractions like machines, instances, clusters, service-level grouping, replication, etc....... ". Schedulers for periodical jobs and orchestrators for services where long-running server responds to requests. Used interchangebly since features oberlap and orchestrators like Airflow have their own schedulers.
    - workflow management tools have schedulers that define tasks and then work with underlying orchestrator to execture jobs. orchestrators often have an instance pool. 
      - airflow was pioneer (config-as-code)and had drawbacks (difficulty setting up diff containers for diff tasks, not parametric, static)
      - perfect (dynamic, parametrized, config-as-code)
      - argo (each step runs natively in diff container but uses YAML). only with k8s and k8s not always available in dev. minikube can stimualt but is messy.
      - Kubeflow and Metaflow - most popular and adnvaced. config as code. kubeflow needs dockerfile and YAML file. Metaflow uses decorators for further abstraction.
  - ml platform: (sagemaker/mlflow) definition varies but set of shared tools for ml adoption and dpeloyment. choice of tool depends on integration with current cloud you're using or wehterh it supports self-hosting/managed service.
- deployment:
- model store: definition (shape, architecture), params (sometimes unified with definition file), featurize/predict funcs, dependencies, data (uri, Data Version Control helps version data), model generation code, experiment artifacts, tags (even including git commit).
- Feature store: management (shareability, definition), computation, storage (acts like a data warehouse), consistency (some platforms help ensure logic is same between training and inference pipelines).
  - dev enviornment (git, ci/cd, ide)
    
49. UX
  -**Consistency-accuracy trade-off???**
  - Human-in-the-loop
  - Smooth failing: use backup/heuristic if new model takes a lot of time
    
50. Responsible AI:
  - Framework
    - Discover sources for bias
      - training data, biased towards underrepresented groups
      - labelling, annotator's subjectivity
      - featuring engineering: **disparate impact, use DisparateImpactRemover??**
      - Model's objective, **does it make sense? does it bias model towards majority????**
      - evaluation
    - Privacy vs accuracy trade-off
      - **Differential privacy: protects individual while sharing group stats???**
    - **Compactness vs fairness trade-off: compression might impact unfairly???**
  - use package slike AI 360 and fairlearn to detect and mitigate bias

52. Degrees of automation: human, shadow, human in the loop (ai assistance (ui to hihglight), partial auto), full automation
53. Data-centric >>> Model-centric
54. PoC: OK for not focusing on reproducability
55. Data pipelines: provenance (where it comes from) and lineage (sequence of steps), metadata

---

# ML Use-cases

## Ads prediction
- downstream systesm: search engine,  sponsored products (query part of contexts), social media
- offline:
   - AUC is calibration insensitive. multiply probs by 2, AUC won't change so need log loss.
   - need calibrated scores in ads.
- online:
   - revenue. bid -> display -> if user clicks -> charge business -> make money. 
   - successful sessions
   - returning users
- action rate (configured by advertiser: click rate, downstream action rate etc)
- counter metrics (hide ad, never see, report)
- actors: user, publisher, item, context (query)
- features:
   - context (region, previous queries), user-ad (embedding sim), user-advertiser (embedding sim), advertiser (hist. egnagement), ad raw terms, ad impression (can cut off for ad engagement), ad negative feedback, ad categ/embedding, region histogram, embedding last k days of ads (just avg them)
   - **ad_id??? advertiser_url (can be used for memorization???)**
   - engagement data can be passed either by histogram bins + current day features OR, a single feature with "today's" engagement rate. first approiach better for NN.
- modelling:
   - selection (context, query/interest)
      - funnel: arbitrarily choose size of output
      - phase 1: index-based filtering. selection criteria by published, user personas
      - phase 2: bid * predicted score. but can use (bid * prior_score) at this point. Score: cost per engagement based on ad, advertiser,
      - phase 3: simple ML model
         - model can be made simply by skipping sparse features in funnel based approach
         - bid * predicted CPE (more accurate the prior approach in phase 2)
   - Scalability:
      -  not sure why selection model was done separately for each shard.
   -  Auction uses engagement probability so need well-calibrating 
        -  recalibration: q = p / (p + [1-p] / w) where w  is negative downstampling rate
        -  factors: bid, engagement score, quality score, budget
        -  CPE = ad rank of ad below / ad rank score + 0.01
  -  pacing:
     - distribute ad over several days
   - ad prediction
      - ads are short-lived, dynamic environment, need rapid updates
      - batch update frequently or a better approach is online learning
         - logistic regression good for online learning
            - online joiner that gets near-real-time features and actions and feeds to
            - training data generator
            - SGD to perform mini-batch update
            - one approach is to use logistic regression but to capture non-linearities/sparse, use leaf nodes/activations from tree/NNs as features
               - prevent catastrophic forgetting
               - NN/tree-based remain fixed, only use for feature computation
               - interpretability/calibration remains intact   
- firing of leaf nodes of tree-based models as features for a downstream model

## Entity Linking
4. Offline: precision/recall/f1. disamb: precision/accuracy = total correct / total entities detected. overall f1-score metric (define FP,FN,TP,TN) for whole system. Macro-avg might be useful if care about class-based or class/entity is imbalanced
5. Online metric: guage perf. of downstream system. 
6. BIO tagging scheema (beginning, inner, non-netity)
7. Downstream systems:
   - search engines
   - VAs
   - the % sesssion/questions success rate
8. Arch:
   - NER (Recognize "terms"/"mentions")
   - Candidate generation (form knowledgebase) (based off wikipedia e.g)
   - disambiguate (link to "entitiy" in knowledgebase)
- Training DATA
   - OS
   - Hand-label
- Modelling:
   - context needs to be accounted. bi-directional.
   - ELMo.
      - either take char-level CNN or init with word2vec static embeddings for words. then train forward and backward layers independetly.
      - resulting word embeddings averaged.
      - can't use simulatenously both direcions.
      - char-level cnn method has advantage of learning "learn" and "learning" similarity and OOV
   - BERT:
      - masking problem solution
      - case/uncased decide based on problem. distil/base ones are faster for inferenace
   - use token-level embeddings as features to a multi-class classifier. Can also fine-tune embeddings based on NER dataset if data is huge and compute/training time permits.
   - disamb: candidate gen. focus on recall
      - indexing:
         - nicknames/alias/subwords/abbrev.
         - anchor text in HTML
         - embedding: for an exisitng index, embed all words in the index. then use/build a model that brings abbrev/alias/syns belonging to same entity together. then use knn to expand index.
   - linking: probability score for each candidate and select highest. focus on precision
      - inputs:
         - embedding for target word
         - then avg of all tokens (can also have separate for entities only) in sentence
         - type of entity (and other detectedn entities' histogram)
         - sentence mebdding of the candidate entity (from knowledgebase)
         - entitiy priors:
         - P (candiadate | term, type, sent)
   - example: [perplexity simulated example](https://www.perplexity.ai/search/help-me-understand-this-the-pr-yR65K3NDSPy3fyeqtHG_NQ)
   - 


## Feed Design
- actors:
   - user, author, content, context, user-author similarity (percentage to nromalize, avg. tfidf/NN embedding of users, social similarity (overlap ), user-content, 
- features:
   - similarity, interaction, influence, relationship, interests, etc.
- Modelling:
  - weighted metric for type of engagements (like, comment etc). better to predict indivudal eventually to be able to fine-tune the knob
  - funnel approach
  - selection:
     - new, unseen tweets, seen tweets with increased engagement, limit recency, tweets by network, suggested and friends' interactions
  - predict prob of engagement
   - Multi-task learning model to predict multiple types of engagement
   - alternatively, can get overall model's score and feed as feature to individual models
   - can stack models in parallel and use a simpler model on the outputs
  - reranker:
    - diversity, repetitiveness (content/user) penality - heuristics based like subtract 0.01
- experimentation: check gain in user engagement metric and p-value
  

## Seach Engine
- Metrics:
  - nDCG, MAP
  - time/queries to success indicates success (lower per session).
- Labels/Data:
  - can assign negative relevance score to a document to mark irrelevant document
  - Define CTR with high dwell time to remove unsuccessful clicks
- actors:
  - searcher (personas)
  - query (intent, historical engagement)
  - context (time/recency, previous queries)
  - doc (pagerank/backlinks, local/global engagement radius)
  - **query-searcher???**
  - query-doc (text match with content/title/metadata, uni/bigram, tfidf (normalized), engagement etc.)
  - searcher-doc (previously accessed, distance for local intent, interest match)
- training data generation:
  - estimate how much training data can be gathered.
- Architecture:
  - query rewriting (spell, expansion vs relaxation to simplify)
  - understnading (intent e.g. local, info, navigational vs informational,)
  - **blender???**
  - 1st stage selctor focused on recall. can do weighted scoring of
    - personalization
    - doc popularity
    - intent match
    - terms match.
    - Google's pagerank has lower weightage and mostly dominated by ML. can assign weights manually or learn by experimentation or learn through ML.
  - ranker itself good be composed of two increasingly complex models/stages.
    - Learning to Rank (LTR) is a supervised machine learning approach to optimize how items are ranked in response to a query. Focus is on assigning scores that yield the correct order, not just correct labels.
      - Stage 1 – Pointwise models: Treats each item independently and predicts a relevance score (e.g., logistic regression, decision trees). Simple but ignores item comparisons.
        - pointwise approach: single doc's relevancy (true ranking at a time). or can "approximate" as relevance/irr using binary classification.  
        - random negative example (result on 50th page)
      - Stage 2 – Pairwise models: Compares item pairs to learn relative orderings (e.g., LambdaRank, LambdaMART). The model learns to adjust scores so that higher-relevance items are ranked above lower-relevance ones, optimizing ranking metrics (e.g., NDCG).
        - human labellers or use historic engagement and infer labels using heuristic.

## Recommendation Engine
- Capacity estimates: users/views
- New users/items constantly added
- recommendation(user_id) -> core function
- model as classification problem. regression if predicting rating.
   - can further re-weight training samples to support business objective to increase session watch time
   - can use all methods and generate candidates. we can also use their raw scores (although not comparable) as features to our 1st stage ranker
- metrics: 
   - mAP., relevant/or not (no relevance score in recommendation), mean of all users, take average precision@k (where relevant).
   - mAR. same calculation for recall
   - f1 based off mAP and mAR
   - RMSE if numeric
- Preprocess
  - normalize matrix by subtracting (user/item) bias using global/user/item avgs **(why subtracting for item makes sense???)**.
- Labels: Implicit (infered like watched) or explicit feedback (rating). Explicit feedback can be biased.
- Modelling:
  - content filtering: if you like ironman, since ironman and ironman2 have same features, recommend ironman 2 (no cold start). user preferences can be boostrapped explicit or using historic. Can caputre a user's niche interests but can't discover. can dot b/w items (weight based on user's score) or build average embedding of user and then find closest items. Cons: hand-engineer features.
  - collab filtering: similar users have simular interests. no domain/hand-engineered features necessary.
    - memory-based, extension of KNN, user-user or item-item.
       - user-user/item-item have different axis of similarities, filter items/users and then custom weighted avg logic to get final recommendations. user prefernces change rapidly so harder to scale. item-based and user-based are static and can be computed offline and served without frequent re-training. sparse matrices, cold start an issue.
    - model-based (SVD, large matrix split into 2 small then multiplied to get missing recommendations.).
       - In SVD, can be done using SGD or WALS (alternately, solve U and V and match to A for error). Hard to include side features (but can extend interaction matrix to add one-hot encoded blocks) and new users/items (a WALS method exists to solve for new item embedding, or use heuristic e.g. avg embedding in a cateogry).
       - Matrix factorization tends to recommend popular items.
  - Softmax DNN, multi-class classification: user query input (dense like watch time, sparse like country), output = softmax over item corpus. Last layer of weights = item embeddings. However, the user embeddings are not learned but system learns for query features. To avoid folding, incorporate negative samples as well (hard negatives, negative pairs with highest error and gradient update). High latency as dynamic user query embeddings need to be computed. cold start for new items.
    - comparison: https://developers.google.com/machine-learning/recommendation/dnn/training
  - Two-tower: learn embeddings for both in input, incorporate side features of both and then predict one pair. **cold start: lack of sufficient examples to learn embeddings.** (i don't think cold start applies to two-tower)
  - hybrid (combine both in a layered approach, weighted approach, to fix cold start, etc.)
- Common arch:
  - Candidate generation (or multiple generators)
     - focus on recall. trending, user interests, genre/cahractersistcs, historical interactions etc.
  - ranker:
     - focus on precision
     - With a smaller pool of candidates, the system can afford to use more features and a more complex model that may better capture context. Scoring can use click-rate, watch time.
  - reranking (diversity, freshness, , business rules, content explicitly disliked by user, exploration/exploitation)
- actors/features: 
   - user (queries, embedding),
   - movies (genre, recency, actors, soundtracks, length, description text embedding),
   - user-movies/genre (historical engagement, similarity),
   - context (time, device, geography),
   - trends,
   - diff. time intervals for metrics,
   - user-actor histogram (normalized)
   - embeddings of past interaction items can be averaged and fed as a feature into a NN model for ranker. both search terms and watched movies.
- once you have embeddings/vectors, it's an ANN problem (e.g. fetch last X entries user has watched)
- Issues:
  - To fix positional bias: Create position-independent rankings or Rank all the candidates as if they are in the top position on the screen.
  - Frequent items have higher norm so dot product metric may dominate. Rare items may not be updated frequently during training so embedding initialization should be carefully done.
  - Cold start
- Evaluation:
  - instead of train/test split, can mask interactions and then predict
- Bootstrapping: by ranking by chronological is fine but trade-off is serving bias (bottom items ignored). Be creative in terms of boostrapping. Heuristic like most engaged feeds, then permuted for randomness.
- Data/Infrastructure 
  - Run inference and data access on the same server to leverage data locality.
  - Use a FAISS-like in memory index to store ANN index and query embeddings .
  - Stores items, users, and caching layers.
  - User interactions are streamed to Kafka, which feeds into a scoring model for real-time updates.
- Inference Optimization Strategies
    - Batch Processing
      - Perform daily batch scoring to precompute recommendations.
      - Limitation: can waste resources by scoring for inactive users; daily updates may be too infrequent.
    - Realtime
      - Precompute user/item embeddings offline.
      - KNN lists may also be precomputed so only have to do scoring for each user (might waste resources)
      - Use Approximate Nearest Neighbors (ANN) for fast similarity search.
      - Use a max-heap to efficiently update a user’s top-K similar items/entities.
      - Consider pre-caching user’s last seen entities to improve latency.

---

# Statistics:

- If effect size to be detected is very small, then to each a given power level, a greater sample size is required.
- In other words, if an effect size = X is observed, then if the sample size was huge, the probabilit of making a Type 2 Error (of not rejecting a false hypothesis) is less as compared to if the effect size X was observed using a small sample.
- Increasing sample size maintains probability of Type 1 error and decreases probability of Type 2 error. The latter makes sense based on the point discussed above.
- Type 1 Error is when you reject a true Null Hypothesis because the difference of the statistics lead to very small chances, that were too small for you to believe it was true (the p-value was too small based on the alpha you set, the chances that didn't matter for you). Regardless of the size of the sample that demonstrated that difference in the statistics, the alpha remains the same, although p-value gets smaller (increasing power).
- alpha is what you define. It's the probability that you'll make a type 1 error. This probability does not change even if you get smaller p-values based on a number of trials / samples. Alpha is the cut-off for p-value.
- Power of a test (1 - beta) depends on effect size, alpha and sample sizes of the two groups. https://medium.com/swlh/why-sample-size-and-effect-size-increase-the-power-of-a-statistical-test-1fc12754c322
- I have yet to come across an example where effect size where we are using a single group / sample to infer for the population. Maybe in the case when we are studying one sample, the other group refers to the poplulation. Maybe.
- Effect sizes like Cohen's D are essentially differences. It's just that they are converted to standardized form using a same standardization (assuming it is same for both groups) or a pooled standard deviation (from both groups). This standardization is actually questionable sometimes. https://rpsychologist.com/cohend/
- Chi-square independence of test: row percentage multiplied by column total (gives you expected frequency)


# Probability
- Arithmetic series (end - start) / step_size -> n_steps
- Law of total probability: P(A) = P(A | S1) P(S1) + P(A | S2) P(S2) + ... (all partitions) pretty much what's used for marginal in bayes theorem
  - depending on the situation, one of the Si's could be A itself 
- Bayes Theorem: P(A | B) = ( P(B | A) . P(A) )/ P(A) - whenever some evidence/prior or observation is given
- Conditional: P(A | B) = P (A and B) / P(B) - whenever a certain observatioin given
- Bernolli Trial (single trial - two outcomes)
- Binomial (# of successes in a series of Bernouli trials)
- Multinomial if more than two outcomes (How many ways can you split 12 people into 3 teams of 4?)
- Combinations nCr
- For assignment problems, think about starting with initial number and reducing options along the way e.g. 10 ways * 9 ways * 8 .... or 12C4 . 8C4 . etc. or even probability (10/10 . 9/10... etc.)
- Geometric Distribution, probability that success occurs at nth trial
- p . (1-p)^(1-x)
  - problems (how many rounds/games/children etc.)
  - expected number of total rounds (1/p)
  - expected nunmber of rounds before success (1/p) - 1
- P(5A | B) = p(A|B)^5
- Always gather data in variables including output, calculate inverse/complement probabilities, can assume generanl knowledge about fair coin/boy or girl, etc.
- For games with dynamic rounds, visualize first set of branches, and assign p of original game. if multiple, can group as p^2.
- For games that recursive after multiple rounds, think about markov chains (calculate steps starting that point onwards with offset indiating number of itinial steps, then for all possible branches write equations and solve them together)
- Sampling from CDF (of a normal dist) is uniform (don't get it exactly but maybe talks about y-axis raange regardless of shape, view vertically)
- Linearity of expectation: E(X) = E(X1) + E(X2) + ....


# SQL
- CTEs/Subqueries and Window functions is a must.
- For funnel analysis/timeline of events, can use UNION ALL.
- Think if self-join is needed.
