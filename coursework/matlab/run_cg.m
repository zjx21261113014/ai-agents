function h = run_cg(p, x0, max_iter, rtol)
%RUN_CG 共轭梯度法（第 2 节要求的方法），解 Q x = c 等价于 min f(x)
%   r_0 = c - Q x_0,  d_0 = r_0
%   alpha_k = (r'r) / (d'Q d)
%   x_{k+1} = x_k + alpha_k d_k
%   r_{k+1} = r_k - alpha_k Q d_k
%   beta_k  = (r_{k+1}' r_{k+1}) / (r_k' r_k)        Fletcher-Reeves
%   d_{k+1} = r_{k+1} + beta_k d_k
%   理论上 n 步内精确收敛（有限步终止），实际因舍入需略多几步。
%   这里用残差范数作为停机判据（相对初始残差）。

if nargin < 3 || isempty(max_iter), max_iter = 3 * p.n; end
if nargin < 4 || isempty(rtol), rtol = 1e-14; end

n = p.n;
mx = max(max_iter, 1);
f = zeros(mx + 1, 1);
err = zeros(mx + 1, 1);
tt = zeros(mx + 1, 1);
res = zeros(mx + 1, 1);

x = x0;
r = p.c - p.Q * x;
d = r;
rr = r' * r;
nd0 = norm(x0 - p.x_star);
res0 = sqrt(rr);

k = 0;
f(1) = p_f(p, x);
err(1) = 1.0;
res(1) = 1.0;

t_start = tic;
converged = false;
while k < mx
    Qd = p.Q * d;
    alpha = rr / (d' * Qd);
    x = x + alpha * d;
    r = r - alpha * Qd;
    rr_new = r' * r;
    k = k + 1;
    f(k + 1) = p_f(p, x);
    err(k + 1) = norm(x - p.x_star) / nd0;
    res(k + 1) = sqrt(rr_new) / res0;
    tt(k + 1) = toc(t_start);
    if sqrt(rr_new) <= rtol * res0
        converged = true;
        break;
    end
    beta = rr_new / rr;
    d = r + beta * d;
    rr = rr_new;
end

h.f = f(1:k + 1);
h.err = err(1:k + 1);
h.time = tt(1:k + 1);
h.iters = k;
h.converged = converged;
h.info = struct('res', res(1:k + 1), 'max_iter', mx);
end
