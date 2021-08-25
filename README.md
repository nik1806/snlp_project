# snlp_project
Final project for SNLP course summer semester 2021

Please look inside `final_submission` directory for final version of notebook code on English and Bengali language. Here, we present a succint summary of the whole project with analysis of results.

## Introduction
With this project, we aim to estimate OOV words using subword representation. To achieve this we train RNN based language model to artificially generate corpus and compute OOV rate on varying sizes of the generated corpora. Estimating OOV words helps improve the performance of the language model. In this work, we achieved a better OOV rate and perplexity score than the baseline for all three levels of granularity with appropriate hyperparameter tuning.

## Methodology
We begin with preparing the data. The given corpus is segmented into sentences and further split into train and test set in an 80:20 ratio. Segmentation helps SentencePiece to act on the sentence level and implement subwords. All punctuation marks are kept intact to retain richness in the generated text. However, Bengali corpus required manual curation like stripping off English strings, country flag symbols, and repeated occurrence of the same punctuation. Also, the lines are segmented into sentences based on punctuation marks like `?`,`!`,`|`. <br>

Next, subword units are learned with SentencePiece at three different granularity levels, i.e., characters, smaller vocabulary, and a larger vocabulary. These subwords are used to train the RNN language model based on different subword granularity. The goal is also to perform hyper-parameter tuning to improve baseline perplexity scores.
Further, we use the trained RNN based language model to create artificial data for each granularity level. <br>

Finally, we compute the out-of-vocabulary (OOV) rate on the given corpus and compare it with the OOV rate computed after augmenting the training vocabulary.

## Results and observations
We observe that with character level granularity, the generated text is segmented on the character level. With granularity set to smaller subwords units closer to characters, the length of segmented subwords is longer and many words are also considered as subwords. E.g. for Bengali, the tokens are combinations of a few characters but not a complete word. E.g, the word `পাওয়া` (pronounced as `Pā'ōẏā`) is segmented as `▁পা` an `ওয়া`. For larger vocabulary granularity levels,  subwords are longer. Many words are also segmented into single subwords. We also consistently see that longer words are broken into two or more segments.


### English
Upon inspection of artificially generated text using rnnlm trained model:
1. Character-level granularity, there is a structure of the sentence and many generated words are also real. However, the grammar is very bad and some words don't exist in English.
2. Small subword granularity, there are more real words. It still lacks grammar and most sentences don't make sense.
3. Larger subword vocabulary, more words are real. Overall, more information and richer meaning is delivered.

With our experiments following parameters are chosen based on better perplexity and OOV rate (more preferred).
**Parameters table**

| Params\Models | Baseline (fixed) | Character | Smaller Vocab | Larger Vocab |
|---------------|------------------|-----------|---------------|--------------|
| hidden        |               40 |        70 |           100 |          140 |
| bptt          |                3 |         5 |             6 |            1 |
| class         | #vocab_size      |        72 |           650 |         1600 |


**Perplexity scores**

|Model\ Granularity|	Characters	| Smaller vocabulary 	| Larger vocabulary |
|------------|------|---|---|
|Baseline |	8.256309 |	68.993461|	71.323803|
|Tuned	| 7.351029	| 67.478303|	70.26725|


**OOV rate** <br>
OOV on given (original) corpus: 0.1073

|Model\ Granularity|	Characters	| Smaller vocabulary 	| Larger vocabulary |
|------------|------|---|---|
|Baseline	|0.0754|	0.06797|	0.07443|
|Tuned	|0.07482	|**0.06679**	|0.07345|


### Bengali
Upon inspection of artificially generated text using rnnlm trained model:
1. Character-level granularity: The generated text is entirely meaningless and grammatically incorrect. The words in themselves are inexistent and carry no meaning. 
2. Small subword granularity: There seem very few real and meaningful words, but the generated text as in whole carries no meaning.
3. Larger subword granularity: Text seems to be the best of the above, although is partially meaningful. It has more correct words and some of which could be attributed to Bangladeshi Bengali dialect.


We observe that with the increasing number of hidden layers and the number of steps to propagate error (bptt), the perplexity declines as well as OOV improves.

**Parameters table**

| Params\Models | Baseline (fixed) | Character | Smaller Vocab | Larger Vocab |
|---------------|------------------|-----------|---------------|--------------|
| hidden        |               40 |        120|           120 |           120|
| bptt          |                3 |         4 |             4 |            4 |
| class         | #vocab_size      |        52 |           400 |         3000 |


**Perplexity scores**

|Model\ Granularity|	Characters	| Smaller vocabulary 	| Larger vocabulary |
|------------      |          ------|---                    |---                |
|Baseline          |	   10.026335|	           62.794973|	      376.060040|
|Custom	           |        7.311287|              49.141582|	      365.665047|

**OOV rate** <br>
OOV on given corpus: 0.157

|Model\ Granularity|	Characters	   |    Smaller vocabulary |      Larger vocabulary|
|------------      |------             |---                    |                   --- |
|Baseline	       |0.1352             |	             0.1248|	             0.1143|
|Custom	           |0.1297             |                 0.1202|                 0.1133|

As can be concluded from the table above, the hyper-parameters that outperform the baseline results are 120 hidden layers and 4 bptt. It can be mentioned that with 120 hidden layers, the training time is quite significant. We also experimented with lower hidden layers and bptt, for which the OOV rates are less than those of baseline. However, we believed increasing the hidden layers and bptt would be a better choice for achieving an improved OOV rate and lower perplexity.

### Comparison of Perplexity and OOV rate vs Vocabulary size
We selected the following vocab size for English and Bengali.

|Granularity|	Characters	   |    Smaller vocabulary |      Larger vocabulary|
|-----------|------            |---                    |          --- |
|English	       |72             |	             650|	              1600|
|Bengali	       |52             |                 400|                 3000|

- *Table: Comparison of Perplexity and OOV rate vs Vocabulary size for English.*

| Vocab Size |   PPL   | OOV       |
|------------|:-------:|-----------|
|     **72** |   7.351 |    0.07482|
|        250 |	 38.00 |	0.07169|
|        450 |   57.866|	0.07032|
|    **650** |   67.478|	0.06679|
|   **1600** |	 70.267|    0.07345|
|       2000 |	 61.796|    0.07933|
|       2500 |	 47.102|    0.07972|

- *Table: Comparison of Perplexity and OOV rate vs Vocabulary size for Bengali.*


| Vocab Size |   PPL   | OOV |
|------------|:-------:|-----------|
|         **52** |  10.026 |    0.1353 |
|         80 |  16.489 |    0.1347 |
|        100 |  21.925 |    0.1331 |
|        200 |  41.832 |    0.1301 |
|        **400** |  62.795 |    0.1248 |
|        800 | 124.997 |    0.1203 |
|       2000 | 278.289 |    0.1154 |
|       2500 | 326.707 |    0.1147 |
|       **3000** | 376.060 |    0.1143 |
|       4000 | 448.427 |    0.1127 |


The OOV rate decreases as the size of generated corpus ($10^k$) increases, thus the OOV rate is inversely proportional to the size of the corpus. In a practical application, we would prefer, a model with a smaller subword vocabulary. From our experiment, it gets the lowest OOV rate. Also intuitively, character level granularity doesn't make meaningful words and model long-term dependencies. The larger vocabulary subwords become very close to actual words. The smaller vocabulary subwords fit better to close the generative gap between characters and whole words.


### Differences in the result
Our results differ for English and Bengali in the following ways:
1. For larger vocabulary Bengali is found to have much higher perplexity than English. This may be due to the fact Bengali is a morphologically richer language than English.

2. We also observed that while Bengali showed a uniform increase in perplexity and decrease in OOV rate with increasing vocab size. However, this behavior for English is not consistent. Net perplexity decreased by 8.47 when vocab size increased from 1600 to 2000. While the OOV rate is expected to decrease with increasing vocabulary size, it escalated continuously from 0.0619 to 0.0797 while increasing vocabulary size from 650 to 2500.

3. For all granularity levels, English had a much lower OOV rate than Bengali. This is again can be explained as Bengali is a morphologically rich language. 

## Takeaway and future work
We analyzed how subwords can be used as efficient way of estimating OOV words and improve the performance of the language model. The results might vary depending on how morphologically rich is the language. We also observed a uniform relation of `vocab_size` with perplexity and OOV rate i.e. with an increase in vocabulary size, the perplexity escalated, while the OOV rate dipped slowly. However, this behavior is inconsistent for the English language.

Here are few ways we could improve the results:
1. We could try training rnnlm with more hidden layers and bptt but at a cost of increased training complexity.
2. We can also search for optimal vocabulary sizes extensively. Some techniques like grid search with small changes of values could give an improvement in model performance.
3. better neural network architectures like transformers can also improve the results.