# naive_bayes_newsgroup_classifier
simple naive bayes implementation for the classic Usenet newsgroups text classification problem.
<br>(project for Artificial Intelligence class.)

the main equation for the naive Bayes classification is:

![probability equation image](equation.png "naive Bayes classification equation")

where:

V<sub>NB</sub> is the classification that gives the maximum probability of observing the words that were actually found in the document (in this case, the newsgroup category), and of course this is subject to the naive Bayes independence assumption;

**P(a<sub>l</sub>, ... a<sub>lll</sub> | v<sub>j</sub>) = *Π* P(a<sub>i</sub> | v<sub>j</sub>)**

In this case, the independence assumption states that the word probabilities for any single position in the text are independent of the words that occur in other positions, given the newsgroup classification **v**<sub>j</sub>. (Obviously this assumption is incorrect, for example, there is a higher probability of the word "Bayes" occurring if the word "naive" is in the previous position.) However despite making this incorrect indepence assumption, naive Bayes classifiers still do remarkably well.

-----

the equation for TF-IDF (term frequency–inverse document frequency) is:

![### ***w<sub>i,j</sub> = tf<sub>i,j</sub> x log(N/df<sub>i</sub>)***](tfidf-equation.png "term frequency - inverse document frequency equation") 

TF-IDF weights word frequencies (term frequencies) by how *unlikely* each word is in the newsgroup category (inverse document frequency), which often leads to improved results. For example, terms like "car" and "wheel" are more likely to appear in the newsgroup category "rec.autos" than in "rec.sport.baseball".

##### the classifier's final accuracy was 72.97%, using TF-IDF weighting method and m-estimate word probability equations.


