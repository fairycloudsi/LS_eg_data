
% This is a function using Gradient Descent method to find the optimal
% value of linear regression. 

d = (3:7)';
M = 2.^d -1;
N = 2.* M;
niter = ones(size(d));
f_opt = ones(size(d));
time_iter = ones(size(d));
conda_list = ones(size(d));

for i = 1:size(d,1)
%     [A,b,~] = GenerateAb_LS_L2(M(i), N(i));
%     filename = strcat('data/LS_Ab', num2str(d(i)));
%     save(filename, 'A', 'b');
    filename = strcat('data/LS_Ab', num2str(d(i)),'.mat');
    data = load(filename);
    A = data.A;
    b = data.b;
    maxiter = (2^(d(i)+1)*1200);
    labd = 1/sqrt(M(i));
    [~, conda_list(i)] = Hess_LS_L2(A,labd);
%     [niter(i), time_iter(i), f_opt(i),~] = GD_LS_L2(A,b,labd,maxiter);   
    Anew = A'*A/(M(i)) + 2*labd*eye(N(i));
    bnew = (A)'*b/M(i);
    [x_opt,~] = amg(Anew, bnew);
    f_opt(i) = LS_L2_obj(A, b, x_opt, labd);
end

disp('exact minimal value of LS_L2 using GD')
T = table; 
T.M = M;
T.N = N;
T.niter = niter;
T.fopt = f_opt;
T.time_iter = time_iter;
T.conda = conda_list;
display(T)

