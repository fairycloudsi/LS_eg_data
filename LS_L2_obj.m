
% This is a function return the function value of Linear regression with L2
% penalty. 
% Input: -- Data matrix: A
%        -- label : b 
%        -- current point: x
%        -- penalty coefficient: labd 
% Output: -- function value of objective function 
% Author: Huiwen Wu, University of California, Irvine 
% Date: 10/12/2019


function f_value = LS_L2_obj(A,b,x,labd)

    [M,~] = size(A);
    res = A*x - b;
    f_value = 1/2/M*(res'*res)+ labd*(x'*x);


end