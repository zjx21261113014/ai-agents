function [xs, cpu] = traj2(p, x0, kind, alpha, beta, n_max, rtol)
%TRAJ2 按方法递推并记录所有迭代点（含 x0），返回 2 x (K+1) 的点阵与 CPU 时间
%   kind: 'gd' | 'hb' | 'nes' | 'bb1'
%   这里为了画图要保留 {x_k} 本身，而 run_*.m 里的 History 只存 f / err / 时间。

x0 = x0(:);
xs = x0;
e0 = norm(x0 - p.x_star);
tol = max(0, rtol * e0);

x_prev = x0;
x = x0;
a = alpha;
if isempty(a)
    a = 1 / p.lam_max;
end

t0 = tic;
for k = 1:n_max
    if any((isnan(x) | isinf(x)))
        break;
    end
    g = p_grad(p, x);
    if norm(x - p.x_star) <= tol
        break;
    end

    switch kind
        case 'gd'
            x_new = x - a * g;
        case 'hb'
            x_new = x - a * g + beta * (x - x_prev);
        case 'nes'
            y = x + beta * (x - x_prev);
            x_new = y - a * p_grad(p, y);
        case 'bb1'
            s = -a * g;
            x_try = x + s;
            yg = p_grad(p, x_try) - g;
            sy = s' * yg;
            if sy > 0
                a = (s' * s) / sy;
            end
            x_new = x_try;
        otherwise
            error('traj2: unknown kind %s', kind);
    end

    x_prev = x;
    x = x_new;
    xs(:, end + 1) = x;
    if any((isnan(x) | isinf(x)))
        break;
    end
end
cpu = toc(t0);
end
