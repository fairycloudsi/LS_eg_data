
% This is a function use rFASD to solve linear regression with L2 penalty
% term. 
% In rFASD method, we divides space R^N into d subspaces using AMG or GMG. 
% In each update, the update direction is given by
% $$ x(k+1) = x(k) - alpha* A_i^{-1} s, $$
% where H(b,b) is the sub-Hessian matrix and s is given by solving the
% linear equation $A_i s_i =  -\nabla_i (f) = g_i$, alpha is computed by 
% $\alpha_i = (g_s, s_i)/ \|s_i\|^2/L.$
% The objective function is linear regression with L2 penalty. 
% $$ f(x) = \frac{1}{2M} \| b - Ax\|^2 + \lambda \|x\|^2, where $\lambda$
% is the regularization parameter, A is data matrix and b is label. 
% Input: M -- number of rows of data matrix 
%        N -- number of cols of data matrix 
%        d -- number of subspaces 
% Output: xopt -- calculated point to achieve optimal value 
%         niter -- number of iterations 
%         g_norm_list -- list of g_norm 
%         fopt -- calculated optimal function value 
%         ti -- elasped CPU time;
% Author: Huiwen Wu, University of California, Irvine
% Date: 10/13/2019

% function [niter, g_norm_list, f_opt] = rFASD_LS_L2(M,N,d)
function [niter, niter_dj, time_setup, time_iter, f_opt] = rFASD_LS_L2(A,b,x0,f_ob,labd,maxiter,d)

    [M,N] = size(A);

    % termination tolerance 
    tol = 1e-1; 
%     dxmin = 1e-8;
%     dfmin = 1e-8;
    
    % maximum number of iterations 
%     maxiter = 50000;
    
    % Create a list to store gradient values 
    g_norm_list = zeros(maxiter,1);
    f_value_list = zeros(maxiter,1);
    x_norm_list = zeros(maxiter,1);
    
    % Generate data matrix A and label b     
%     [A,b,x] = GenerateAb_LS_L2(M,N);
    
    % Set regularization coefficient 
%     labd = sqrt(1/M);
    tic;
    % Generate Hess matrix 
    Hess_f = Hess_LS_L2(A,labd); 
    
    
    
    % Set up subspace matrices
    Ai = cell(d,1);
    Pro = cell(d,1);
    Res = cell(d,1);
    sizei = ones(d,1);
    
    sizei(end) = size(Hess_f,1);
    
    Ai{d} = Hess_f;
    theta = 0.2; % coarsen parameters 
    
    for j = d-1:-1:1
        [isC, As] = coarsenAMGc(Ai{j+1}, theta); 
        [Pro{j}, Res{j}] = interpolationAMGt(As, isC);
        Ai{j} = Res{j}*Ai{j+1}*Pro{j};
        sizei(j) = sizei(j+1) + size(Ai{j},1);
    end 
    
    Pi = zeros(sizei(end), sizei(1));
    A_whole = zeros(sizei(1), sizei(1));
    A_whole(1:sizei(end), 1:sizei(end)) = Hess_f;
    Pi(1:sizei(end), 1:sizei(end)) = speye(sizei(end));
    Pi(:, sizei(end)+1:sizei(end-1)) = Pi(1:sizei(end), 1:sizei(end))*Pro{d-1};
    A_whole(1:sizei(end), sizei(end)+1:sizei(end-1)) = Hess_f*Pro{d-1};
    A_whole(sizei(end)+1: sizei(end-1), 1:sizei(end)) = Res{d-1}*Hess_f;
    A_whole(sizei(end)+1:sizei(end-1), sizei(end)+1:sizei(end-1)) = Res{d-1}*Hess_f*Pro{d-1};
    
    
    for j = d-2:-1:1
        A_whole(sizei(j+1)+1:sizei(j),sizei(j+1)+1:sizei(j)) = Ai{j};
        A_whole(1:sizei(j+1), sizei(j+1)+1:sizei(j)) = A_whole(1:sizei(j+1), sizei(j+2)+1:sizei(j+1))*Pro{j};
        A_whole(sizei(j+1)+1:sizei(j), 1:sizei(j+1)) = Res{j} * A_whole(sizei(j+2)+1:sizei(j+1), 1:sizei(j+1));
        Pi(:, sizei(j+1)+1:sizei(j)) = Pi(:,sizei(j+2)+1:sizei(j+1))*Pro{j};         
    end
    
    
    b_whole = Pi'*(A'*b)/M;
    DL_whole = A_whole(sizei(2)+1: sizei(1), sizei(2)+1:sizei(1));
    
    time_setup = toc; 
    
    
%     figure(3)
%     spy(A_whole);
    % define starting point
%     x0 = ones(N,1);
    
    % Initialize gradient norm, optimization vector, iteration counter, and
    % permutation 
    
    g_norm = inf; niter = 1;
    dx = inf; diff = inf;
    
    
    % define the objective function 
    x_whole = Pi'*x0;
    f_value = @(x) LS_L2_obj(A,b,x,labd);
    
    x_new = x_whole;
    
    
    
    
    g_norm_list(1) = norm(- b_whole + A_whole*x_whole);
    f_value_list(1) = f_value(x0);
    x_norm_list(1) = norm(x_whole); 
    
    tic;
    % randomized fast subspace descent algorithm:
%     while and(g_norm/g_norm_list(1) >= tol, niter <= maxiter)
%     while and(and(and(g_norm_list(niter)/g_norm_list(1) >= tol, niter <= maxiter), dx >= dxmin), diff >= dfmin)
    while and(abs(f_value_list(niter) - f_ob)>= tol, niter <= maxiter) 
        niter = niter +1;
        % calculate gradient 
        grad_f = - b_whole + A_whole*x_whole; % Not true for this example
        
        % Random choose a subspace in uniform distribution 
        r = mod(niter, sizei(1));
%         grad_r = c_grad_LS(A_whole, b_whole, labd, r);
        if and(r>0, r<=sizei(2))
%             g_update = grad_f(r); % Compute gradient individually 
            alpha = 1/A_whole(r,r);
            x_new(r) = x_whole(r) - alpha*grad_f(r);
        elseif or(r ==0, r>sizei(2))
            g_update = grad_f(sizei(2)+1:sizei(1));
%             alpha = inv(DL_whole);
            x_new(sizei(2)+1:sizei(1)) = x_whole(sizei(2)+1:sizei(1)) -...
                DL_whole\g_update;
            
        end
        
        
        g_norm = norm(grad_f);
        g_norm_list(niter) = g_norm;
        x_norm_list(niter) = norm(x_whole);
        f_value_list(niter) = f_value(Pi*x_whole);
%         dx = norm(x_new - x_whole)/x_norm_list(1);
%         if niter > 1
%             diff = norm(f_value_list(niter) - f_value_list(niter -1))/f_value_list(1);
%         end
        
%         if (niter >1)
%             disp(niter);
%             disp(g_norm/g_norm_list(niter-1));
%         end
        
        x_whole = x_new;
        
    end
    time_iter = toc;
    
    f_opt = f_value(Pi*x_new);
    
    niter_dj = niter/sizei(1);
    
    
    figure(1)
    plot(1:niter, f_value_list(1:niter), '-b', 'LineWidth', 2);
    title('Linear Regression rFASD: fvalue')
    xlabel('number of iterations')
    ylabel('function value')
    name = strcat('rFASD_LS_L2', datestr(now));   
    saveas(gcf, name, 'jpg');
    figure(2)
    plot(1:niter, g_norm_list(1:niter), '-r', 'LineWidth', 2);
    title('Linear Regression rFASD: gnorm')
    xlabel('number of iterations')
    ylabel('norm of gradient')
    name = strcat('rFASD_LS_L2_gnorm', datestr(now));   
    saveas(gcf, name, 'jpg');
    
end


            
            
        
    
    
    
    
    
    
    
    
    
    
    
    


