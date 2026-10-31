function h = run_bb(p, x0, variant, max_iter, rtol, atol, alpha0)
%RUN_BB Barzilai-Borwein 方法
%   s_k = x_k - x_{k-1},  y_k = g_k - g_{k-1}
%   BB1: alpha_k = (s's) / (s'y) = 1 / R_Q(s)
%   BB2: alpha_k = (s'y) / (y'y) = R_Q(s) / R_{Q^2}(s)
%   二次函数上 s'y = s'Qs > 0，两种步长都自动落在 [1/lambda_max, 1/lambda_min] 内，
%   右端是 GD 稳定上界 2/lambda_max 的 kappa/2 倍，这就是 BB 快但 f 不单调的原因。
%   首步取 alpha0 = 1/lambda_max，与固定步长 GD 的第一步相同。

if nargin < 4 || isempty(max_iter), max_iter = 4e5; end
if nargin < 5 || isempty(rtol), rtol = 1e-10; end
if nargin < 6 || isempty(atol), atol = 0; end
if nargin < 7 || isempty(alpha0), alpha0 = 1 / p.lam_max; end

f = zeros(max_iter + 1, 1);
err = zeros(max_iter + 1, 1);
tt = zeros(max_iter + 1, 1);
al = zeros(max_iter, 1);

x = x0;
g = p.Q * x - p.c;
nd0 = norm(x0 - p.x_star);
k = 0;
f(1) = p_f(p, x);
err(1) = 1.0;
tol = max(atol, rtol);

alpha = alpha0;
t_start = tic;
converged = false;
while k < max_iter
    x_new = x - alpha * g;
    g_new = p.Q * x_new - p.c;
    s = x_new - x;
    yv = g_new - g;
    sy = s' * yv;
    if sy > 0
        if variant == 1
            alpha = (s' * s) / sy;
        else
            alpha = sy / (yv' * yv);
        end
    end
    x = x_new;
    g = g_new;
    k = k + 1;
    al(k) = alpha;
    f(k + 1) = p_f(p, x);
    err(k + 1) = norm(x - p.x_star) / nd0;
    tt(k + 1) = toc(t_start);
    if (isnan(err(k + 1)) | isinf(err(k + 1)))
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
h.info = struct('variant', variant, 'alpha0', alpha0, 'alphas', al(1:k), 'tol', tol);
end
