
% This is a function use Gradient descent method to solve Linear Regression
% with L2 penalized problem. 
% Author: Huiwen Wu
% Date: 10/12/2019

function [niter, time_iter, f_2,x] = GD_LS_L2(A,b,labd,maxiter)

    % Generate A,b
%     [A,b,x] = GenerateAb_LS_L2(M,N);
    [M,N] = size(A);
    
    % Initialize x 
    x = 1e-2*ones(N,1);
    
    % termination tolerance 
    tol = 1e-6;
    
    
    % regularization parameter
%    labd = 1/sqrt(M);
%     labd = 0;
%     alpha = abs(1/4*(eigs(A'*A,1)/M));
%     alpha = 0.01;
    alpha = min(1e-6,1/4/(eigs(A'*A,1)/M));
    disp('alpha = ')
    disp(alpha)
    
    % maxium number of iteration 
%     maxiter = 100*N;
    
    f_value_list = zeros(maxiter, 1);
    x_norm_list = zeros(maxiter,1);
    
    
    % Initialize gradient norm, optimization vector, iteration counter 
    niter = 1; 
    g_norm_list = zeros(maxiter,1);
    

    % define objective function 
    f_value =@(x) LS_L2_obj(A,b,x,labd);
    f_value_list(1) = f_value(x);
    g_value = @(x) grad_LS_L2(A,b,x,labd);
    g_norm_list(1) = norm(g_value(x));
    x_norm_list(1) = norm(x); 
    
    tic;
    % coordinate gradient descent algorithm 
   while and(g_norm_list(niter)/g_norm_list(1) >= tol, niter <= maxiter)
%     while niter <= maxiter
        niter = niter +1; 
        
        % gradient descent 
        grad = g_value(x);
        x_new = x - alpha*grad;
        
        f_value_list(niter) = f_value(x_new);
        g_norm = norm(grad);
        g_norm_list(niter) = g_norm;
        x_norm_list(niter) = norm(x);
        
        x = x_new;
        if g_norm > 1e8
            break;
        end
    end
    
    time_iter = toc;
    
    f_2 = f_value(x);
    
    figure(1)
    plot(1:niter, f_value_list(1:niter), '-b', 'LineWidth', 2);
    title('Linear Regression Coordinate Descent: fvalue')
    xlabel('number of iterations')
    ylabel('function value')
    name = strcat('rCD_LS_L2', datestr(now));   
    saveas(gcf, name, 'jpg');
    figure(2)
    plot(1:niter, g_norm_list(1:niter), '-r', 'LineWidth', 2);
    title('Linear Regression Coordinate Descent: gnorm')
    xlabel('number of iterations')
    ylabel('norm of gradient')
    name = strcat('rCD_LS_L2_gnorm', datestr(now));   
    saveas(gcf, name, 'jpg');
  
    
    
    
    


end