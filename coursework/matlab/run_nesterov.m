function h = run_nesterov(p, x0, alpha, max_iter, rtol, atol, beta_const)
%RUN_NESTEROV Nesterov 加速梯度法
%   y_k     = x_k + beta_k (x_k - x_{k-1})          外推点
%   x_{k+1} = y_k - alpha * grad f(y_k)
%   beta_k 缺省取 (k-1)/(k+2)（一般凸问题求 O(1/k^2) 用的），
%   若给 beta_const 则改用常数动量。
%   梯度取自外推点，高频分量被放大，故稳定步长上限退回 alpha < 1/lambda_max。

if nargin < 4 || isempty(max_iter), max_iter = 1e5; end
if nargin < 5 || isempty(rtol), rtol = 1e-10; end
if nargin < 6 || isempty(atol), atol = 0; end
if nargin < 7, beta_const = []; end

f = zeros(max_iter + 1, 1);
err = zeros(max_iter + 1, 1);
tt = zeros(max_iter + 1, 1);
betas = zeros(max_iter, 1);

x = x0;
x_prev = x0;
nd0 = norm(x0 - p.x_star);
k = 0;
f(1) = p_f(p, x);
err(1) = 1.0;
tol = max(atol, rtol);

t_start = tic;
converged = false;
while k < max_iter
    if isempty(beta_const)
        beta = k / (k + 3);          % (k-1)/(k+2)，k 从 0 起算
        if k == 0, beta = 0; end
    else
        beta = beta_const;
    end
    y = x + beta * (x - x_prev);
    g = p.Q * y - p.c;
    x_prev = x;
    x = y - alpha * g;
    k = k + 1;
    betas(k) = beta;
    % 记录时直接用新点算 f，保证 f 与 x 对应
    f(k + 1) = p_f(p, x);
    err(k + 1) = norm(x - p.x_star) / nd0;
    tt(k + 1) = toc(t_start);
    if (isnan(err(k + 1)) | isinf(err(k + 1))) || err(k + 1) > 1e30
        break;
    end
    if err(k + 1) <= tol
        converged = true;
        break;
    end
end

h.f = f(1:k + 1);
h.err = err(1:k + 1);
h.time = tt(1:k + 1);
h.iters = k;
h.converged = converged;
h.info = struct('alpha', alpha, 'beta_const', beta_const, 'betas', betas(1:k), 'tol', tol);
end
