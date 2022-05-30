# svdpp-variants
PyTorch implementations of [Regularized SVD](https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf), [SVD++](https://people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf) and Bayesian SVD++ algorithms for collaborative filtering.

### Usage
```
cd src/
python main.py --config ../config.yml
```
config.yml allows you to change model, as well as other hyperparameters.


### Performance
Dataset is [MovieLens 1M](https://grouplens.org/datasets/movielens/). Dataset split is 8:1:1 and random with seed 42.
Performance metric is RMSE between ground truth and predicted ratings. Reported performances reflect the validity of each variant's mathematical modelling. Better evaluation is possible by changing random seed 5 times, and reporting mean RMSEs.


| Model      | Test RMSE | Best Validation RMSE     |
| :---        |    :----:   |          :---: |
| Baseline: predict train set mean |   1.1164    |  -  |
| Baseline: predict user mean |   1.0369      | -      |
| Baseline: predict item mean |   0.9809      | -      |
| Regularized SVD |   0.8595      |   0.8561    |
| SVD++ |    0.8492     |   0.8511    |
| Bayesian SVD++ |         |       |