


% This is a function use Coordinate descent method to solve Linear Regression
% with L2 penalized problem. 
% Author: Huiwen Wu
% Date: 10/12/2019

function [niter,niter_dn, time_setup, time_iter, f_2] = rBCD_LS_L2(A,b,x,f_ob,labd,maxiter,order)

    % Generate A,b
%     [A,b,x] = GenerateAb_LS_L2(M,N);
    
    % Initialize x 
%     x = ones(N,1);
    [M,N] = size(A);
    
    % termination tolerance 
    tol = 1e-1;
%     dxmin = 1e-8;
%     dfmin = 1e-8;
    
    
    % regularization parameter
%    labd = 1/sqrt(M);
%     labd = 0;
    tic;
%     alpha = abs(1/4*(1/(eigs(A'*A,1))/M));
    alpha = 1e-4;
%     alpha = min(1e-4,1/4*(eigs(A'*A,1)/M));
    disp('alpha = ')
    disp(alpha)
    time_setup = toc;
    
    % maxium number of iteration 
%     maxiter = 1000*N;
    
    f_value_list = zeros(maxiter, 1);
    x_norm_list = zeros(maxiter,1);
    
    
    % Initialize gradient norm, optimization vector, iteration counter 
    niter = 1; x_new = x;  
    g_norm = inf;
%     dx = inf; diff = inf;
    g_norm_list = zeros(maxiter,1);
    

    % define objective function 
    f_value =@(x) LS_L2_obj(A,b,x,labd);
    f_value_list(1) = f_value(x);
    g_value = @(x) grad_LS_L2(A,b,x,labd);
    g_norm_list(1) = norm(g_value(x));
%     x_norm_list(1) = norm(x); 
    
    tic;
    % coordinate gradient descent algorithm 
%    while and(and(and(g_norm_list(niter)/g_norm_list(1) >= tol, niter <= maxiter), dx>= dxmin), diff >= dfmin)
    while and(abs(f_value_list(niter) - f_ob) >= tol, niter <= maxiter)
%     while niter <= maxiter
        niter = niter +1; 
        % random pick a coordinate 
        if order == 'r'
            r = randi([1,N]);
        elseif order == 'c'
            r = mod(niter,N);
            if r == 0
                r = N;
            end
        end
        % calculate gradient at r coordinate 
        grad_r = c_grad_LS_L2(A,b,x,labd,r);
        
        % update is the chosen coordinate 
        x_new(r) = x(r) - alpha*grad_r;
%         % gradient descent 
%         grad = g_value(x);
%         x_new = x - alpha*grad;
        f_value_list(niter) = norm(f_value(x_new));
        g_norm = norm(grad_r);
        g_norm_list(niter) = g_norm;
%         x_norm_list(niter) = norm(x);
%         dx = norm(x_new -x)/x_norm_list(1);
%         if niter > 1
%             diff = norm(f_value_list(niter) - f_value_list(niter-1))/f_value_list(1);
%         end 
%         
        x = x_new;
        if g_norm > 1e5
            break;
        end
    end
    
    time_iter = toc;
    
    f_2 = f_value(x);
%     niter_dn = niter/N;
    niter_dn = niter;
    
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