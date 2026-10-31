function h = run_sd(p, x0, max_iter, rtol, atol)
%RUN_SD 最速下降法：每步做精确线搜索
%   alpha_k = (g'g) / (g'Q g)   —— 二次函数上的精确线搜索有闭式解
%   注意二次函数上相邻两步的梯度正交，这正是它慢（锯齿）的原因。

if nargin < 3 || isempty(max_iter), max_iter = 1e6; end
if nargin < 4 || isempty(rtol), rtol = 1e-10; end
if nargin < 5 || isempty(atol), atol = 0; end

f = zeros(max_iter + 1, 1);
err = zeros(max_iter + 1, 1);
tt = zeros(max_iter + 1, 1);
al = zeros(max_iter, 1);

x = x0;
nd0 = norm(x0 - p.x_star);
k = 0;
f(1) = p_f(p, x);
err(1) = 1.0;
tol = max(atol, rtol);

t_start = tic;
converged = false;
while k < max_iter
    g = p.Q * x - p.c;
    Qg = p.Q * g;
    alpha = (g' * g) / (g' * Qg);
    x = x - alpha * g;
    k = k + 1;
    al(k) = alpha;
    f(k + 1) = p_f(p, x);
    err(k + 1) = norm(x - p.x_star) / nd0;
    tt(k + 1) = toc(t_start);
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
h.info = struct('alphas', al(1:k), 'tol', tol);
end
