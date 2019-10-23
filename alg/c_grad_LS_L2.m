
% This is a function return coordinate of gradient of Linear Regression
% with L2 penalized problem 
% Author: Huiwen Wu
% Date: 10/12/2019

function grad_j = c_grad_LS_L2(A,b,x,labd,j)

    [M,~] = size(A);
    
    res = A*x - b;
    
    grad_j = 1/M*(A(:,j))'*res + 2*labd*x(j);
    
    
    

end

