
% This is a function to 
%       1. Generate Gaussian Random matices 
%       2. Save them
%       3. Use amg to solve for x_opt of L2 regularized LS problem 
%           \| A*x - b\|^2 + \lambda \|x\|^2
%       4. Get f_opt and save 
%       5. Solve by rFASD and rCD and compare results 
%  Author: Huiwen Wu, University of California, Irvine
%  Date: 10/22/2019


%% 0. Set up 

d = (9:10)';
M = 2.^d -1;
N = 2.* M;
niter = ones(size(d));
f_opt = ones(size(d));
time_iter = ones(size(d));
conda_list = ones(size(d));

flag = ones(size(d));
relres = ones(size(d));


%% 1. Generate UDV matrices and save 
for i = 1:size(d,1)
    [A,b,~] = GenerateAb_LS_L2(M(i), N(i));
    filename = strcat('data/LS_Ab', num2str(d(i)));
    save(filename, 'A', 'b');    
end

%% 2. Amg solve L2 regularized problem compute the condition number of Hessian 

for i = 1:size(d,1)
    filename = strcat('data/LS_Ab', num2str(d(i)),'.mat');
    data = load(filename);
    A = data.A;
    b = data.b;
    maxiter = (2^(d(i)+1)*1200);
    labd = 0.1/sqrt(M(i));
    [~, conda_list(i)] = Hess_LS_L2(A,labd);

    Hess = A'*A/(M(i)) + 2*labd*eye(N(i));
    bnew = A'*b/M(i);
    [x_opt,~] = amg(Hess, bnew);
    f_opt(i) = LS_L2_obj(A,b,x_opt,labd);
end
    
disp('condition number and f_opt of LS_L2_UDV problem')
T = table;
T.M = M;
T.N = N;
T.fopt = f_opt;
T.conda = conda_list;
display(T);
    
%% 3. Save f_opt and condat_list
filename = strcat('data/LS_Ab_f_opt_lambda01.mat');
save(filename,'f_opt');
filename = strcat('data/LS_Ab_conda_list_lambda01.mat');
save(filename, 'conda_list')

%% 4. Use rFASD and rBCD to solve and compare results 

niter1 = ones(size(d));
f_opt1 = ones(size(d));
time_setup1 = ones(size(d));
time_iter1 = ones(size(d));
niter_dn = ones(size(d));

niter2 = ones(size(d));
f_opt2 = ones(size(d));
time_setup2 = ones(size(d));
time_iter2 = ones(size(d));
niter2_dj = ones(size(d));


for i = 1:size(d,1)
    filename = strcat('data/LS_Ab', num2str(d(i)),'.mat');
    data = load(filename);
    A = data.A;
    b = data.b;
    labd = 0.1/sqrt(M(i));
    x = 0.1*ones(N(i),1); %initial value 
    maxiter = (2^(d(i)+1))*20000;
    order = 'r'; 
    f_ob = f_opt(i);
%     labd = 0;
    [niter1(i),niter_dn(i), time_setup1(i), time_iter1(i), f_opt1(i)] = rBCD_LS_L2(A,b,x,f_ob,labd,maxiter,order);
    [niter2(i),niter2_dj(i), time_setup2(i), time_iter2(i), f_opt2(i)] = rFASD_LS_L2(A,b,x,f_ob,labd,maxiter,d(i));
    
end

for i =1:size(d,1)
    niter_dn(i) = niter_dn(i)/N(i);
end

disp('Performance Comparison of rBCD_LS_L2 and rFASD_LS_L2')
T = table; 
T.M = M;
T.N = N;
T.niter = niter1;
T.niter_dn = niter_dn;
T.niter2 = niter2;
T.niter2_dj = niter2_dj;
T.fopt = f_opt1;
T.fopt2 = f_opt2;
T.fexact = f_opt;
T.time_setup = time_setup1;
T.time_setup2 = time_setup2;
T.time_iter = time_iter1;
T.time_iter2 = time_iter2;
display(T)


    