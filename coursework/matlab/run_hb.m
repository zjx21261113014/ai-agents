function h = run_hb(p, x0, alpha, beta, max_iter, rtol, atol)
%RUN_HB Heavy Ball 重球法
%   x_{k+1} = x_k - alpha * grad f(x_k) + beta * (x_k - x_{k-1})
%   取 x_{-1} = x_0，即第一步不含动量项。
%   二次函数上每个特征分量满足递推 z_{k+1} = (1+beta-alpha*lam) z_k - beta z_{k-1}，
%   稳定条件是 0 < alpha < 4/lambda_max = 2L，比 GD 的 0 < alpha < L 宽一倍。

if nargin < 5 || isempty(max_iter), max_iter = 1e5; end
if nargin < 6 || isempty(rtol), rtol = 1e-10; end
if nargin < 7 || isempty(atol), atol = 0; end

f = zeros(max_iter + 1, 1);
err = zeros(max_iter + 1, 1);
tt = zeros(max_iter + 1, 1);

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
    g = p.Q * x - p.c;
    x_new = x - alpha * g + beta * (x - x_prev);
    x_prev = x;
    x = x_new;
    k = k + 1;
    f(k + 1) = p_f(p, x);
    err(k + 1) = norm(x - p.x_star) / nd0;
    tt(k + 1) = toc(t_start);
    if ~isfinite(err(k + 1)) || err(k + 1) > 1e30
        break;      % 发散，直接退出（外推步长过大时会走到这里）
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
h.info = struct('alpha', alpha, 'beta', beta, 'tol', tol);
end
