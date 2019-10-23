
% This is a function return the Hessian value of Linear regression with L2
% penalty at point x 
% Input: -- Data Matrix: A
%        -- label: b (not needed)
%        -- penalized coefficient: labd
% Output: -- Hess matrix 
% Author: Huiwen Wu, University of California, Irvine 
% Date: 10/12/2019 

function [Hess,conda] = Hess_LS_L2(A,labd)
    
    [M,N] = size(A);
    Hess = (1/M)*(A'*A) + 2*labd*eye(N);
    conda = cond(Hess);
    
end
