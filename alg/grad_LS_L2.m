

% This is a function return the gradient of Linear regression with L2
% penality at point x
% Input: -- Data matrix: A
%        -- label: b 
%        -- penalized coefficient: labd 
%        -- current point: x
% Output: -- grad at x 
% Author: Huiwen Wu, University of California, Irvine 
% Date: 10/12/2019

function grad = grad_LS_L2(A,b,x,labd)
    
    [M,~] = size(A);
    res = A*x - b;
    grad = 1/sqrt(M)*A'*res + 2* labd*x;
end 