### TIMELY: Improving Labeling Consistency in Medical Imaging for Cell Type Classification
ECAI 2020: http://ecai2020.eu/papers/1174_paper.pdf

Diagnosing diseases such as leukemia or anemia requires reliable counts of blood cells. Hematologists usually label and count microscopy images of blood cells manually. In many cases, however, cells in different maturity states are difficult to distinguish, and in combination with image noise and subjectivity, humans are prone to make labeling mistakes. This results in labels that are often not reproducible, which can directly affect the diagnoses. We introduce TIMELY, a probabilistic model that combines pseudotime inference methods with inhomogeneous hidden Markov trees, which addresses this challenge of label inconsistency. 
We show first on simulation data that TIMELY is able to identify and correct wrong labels with higher precision and recall than baseline methods for labeling correction. We then apply our method to two real-world datasets of blood cell data and show that TIMELY successfully finds inconsistent labels, thereby improving the quality of human-generated labels.

The folder `demo` contains notebooks for the applications to a cell lineage line (`hmm.ipynb`) and a cell lineage tree (`hmt.ipynb`).
Unfortunately, the data used cannot be made publicly available.
