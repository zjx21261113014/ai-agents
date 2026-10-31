function h = run_gd(p, x0, alpha, max_iter, rtol, atol)
%RUN_GD 固定步长梯度下降  x_{k+1} = x_k - alpha * grad f(x_k)
%
%   h 的字段
%     f          目标函数值序列，h.f(k+1) 对应 k = 0,1,...
%     err        相对误差 ||x_k - x*|| / ||x_0 - x*||
%     time       累计 CPU 时间（秒）
%     iters      实际步数
%     converged  是否达到容差
%     info       附加信息（步长、梯度范数等）

if nargin < 4 || isempty(max_iter), max_iter = 1e6; end
if nargin < 5 || isempty(rtol), rtol = 1e-10; end
if nargin < 6 || isempty(atol), atol = 0; end

% 目标函数值用恒等式 f(x) = f* + 1/2 (x-x*)' grad f(x) 算，
% 省掉一次 A*x 的乘法，避免在上百万步的循环里多做无用功。
f = zeros(max_iter + 1, 1);
err = zeros(max_iter + 1, 1);
tt = zeros(max_iter + 1, 1);

x = x0;
nd0 = norm(x0 - p.x_star);

k = 0;
f(1) = p_f(p, x);
err(1) = 1.0;
tol = max(atol, rtol * err(1));

t_start = tic;
converged = false;
while k < max_iter
    g = p.Q * x - p.c;
    x = x - alpha * g;
    k = k + 1;
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
h.info = struct('alpha', alpha, 'tol', tol, 'alphas', alpha);
end
