# MetricsForHashing-TensorFlow2
These file implement metrics to evaluate retrieval methods by TensorFlow 2. The retrieval can be done on real data matrix or binary data matrix. Specificly, binary data matrix retrieval refers to hashing. Hashing methods embed real data matrix to binary code matrix. This file implement metrics to evluate hashing methods in TensorFlow 2. Currently, it has two matrics: mean average precision (MAP) and F-measure.
## Requirement:
TensorFlow 2.3.0
Numpy 1.18.5
Although I tested these functions in the above environment, these functions use very elementary functions in TensorFlow and Numpy, so I think other versions, either earlier or later, may work too.
## Mean Average Precision (mAP)
The average precisoin (AP) is defined as:
$$
AP=\frac{1}{n}\sum_{r=1}{R}{P(r)\delta(r)},
$$
where $R$ is the radius of Hamming distance, $P(r)$ is the precision of the top $r$ retrieved data and $\delta(r)=1$ if the $r$-th retrieved datum is a true neighbor of the query, otherwise $\delta(r)=0$. MAP is the mean of APs for all queries.
## F-measure
F-measure is defined as:
$$
F=\beta\frac{precision\cdot recall}{precision+recall},
$$
where $\beta$ is a positive constant generally set as 2. Also, you can specify a (Hamming) radius for retrieving data. The F-measure is calculated only by these retrieved data. Different radius leads to different results.
**NOTE** Although the term 'radius' are used in mAP and F-measure, their meanings are different. For calculating mAP, the radius $R$ refers to top $R$ retrieved data. For calculating F-measuer, the radius refers to an Euclidean distance/Hamming distance that you use to filter neighbors. 
## Usage
The file is self-explained. I think the comments are enough for you to use these two functions.
If you want to retrieve data by real vectors rather binary vectors, just substitute `hamming_dist` with `dist`.
