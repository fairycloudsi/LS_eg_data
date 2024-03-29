# LS_eg_data

In this repository we compare randomized Block Coordinate Descent Method and randomized Fast Subspace Descent Method with data matrices generated by Gaussian distribution and UDV matrices. 

The objective function of minimization problem is 

$
f(x) = \frac{1}{M} * \| b - A x\|^2 + \lambda \|x\|_2^2,
$

where $A$ is the data matrix, b is the label and x is the weights we want to figure out. 

## Dimension set up

We generate matrices by dimension $M \times N$, where 

* d = (3,4,5,6,7,8)';
* N = 2.^d -1;
* M = 2*N; 

## Gaussian distributed data matrix 


GenerateAb_LS_L2.m generates data matrix A and label b. 

* A: each row is independently sampled from a N-dim Gaussian distribution with mean 0 and variance $\Sigma$; 
* x: randomly select N/10 entries of x, each of which is independently sampled from a uniform distribution over support (-2,2);
* b: generated by linear model b = Ax + \epsilon, where \epsilon is sampled from an n-variate Gaussian distribution N(0, I_n). 
* $\Sigma$: covariance matrix of data, the diagonal is always 1 and off-diagonal can be chosen from the set {0, 0.5, 0.75, 0.95}. The larger of off-diagonal values, the more ill condition of the data matrix A. 

The data generated by GenerateAb_LS_L2.m are stored in LS_Ab*.mat. 

* $\lambda = 0$, condition number of Hessian and optimum are stores in LS_L2_conda.mat and LS_L2_fopt.m respectively. 

* $\lambda = 0.1/M$ condition number of Hessian and optimum are stores in LS_L2_conda_01.mat and LS_L2_fopt_01.m respectively. 

See detail in eg_LS_L2.m 

## UDV data matrix 


UDV matrix is generated by 
$$
A = U * D * V,
$$
where 
* $U$ is orthogonal matrix of dim $M \times N$
* $V$ is orthgonal matrix with dim $N \times N$ 
* $D$ is a diagonal matrix with elements equal distributed from 1 to c
* c is a real number provided, chosen from {20,40,100,200,300,500}, the larger c, the more ill conditioned of matrix A. 

### $\lambda = 0$

* UDV matrix are stored as structure in LS_UDV_lambda0.mat 
* Condition number are stored in LS_UDV_conda_list_lambda0.mat
* Optimum value are stored in LS_UDV_f_opt_lambda0.mat

### $\lambda = 0.1/M$

* UDV matrix are stored as structure in LS_UDV_lambda01.mat 
* Condition number are stored in LS_UDV_conda_list_lambda01.mat
* Optimum value are stored in LS_UDV_f_opt_lambda01.mat

More details, see eg_LS_UDV1.m

For numerical results, check [here](alg/results_txt/results_LS_Ab_1022). We reduce iteration steps from 13745 to 179.7 in worst case and achieve a performance comparable to AMG. 
