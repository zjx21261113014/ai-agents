function out = mom_longrun(p, alpha, beta, scheme, steps)
%MOM_LONGRUN 在特征坐标下长程仿真动量法，看 rho_k 是否真的趋于谱半径
%   二次函数上动量法的每个特征分量都满足二阶线性递推
%       z_{k+1} = pp * z_k - qq * z_{k-1}
%   直接迭代这 n 条标量递推比跑原算法快得多，可以在算法早已达到容差之后
%   继续观察渐近行为。
%
%   scheme = 'heavy_ball' : pp = 1 + beta - alpha*lam,  qq = beta
%   scheme = 'nesterov'   : pp = (1+beta)(1 - alpha*lam), qq = beta*(1 - alpha*lam)

if nargin < 5, steps = 20000; end

[V, D] = eig(p.Q);
lam = diag(D);
c0 = V' * (zeros(p.n, 1) - p.x_star);      % x_0 = 0 时的初始误差系数

if strcmpi(scheme, 'heavy_ball') || strcmpi(scheme, 'hb')
    pp = 1 + beta - alpha * lam;
    qq = beta * ones(size(lam));
elseif strcmpi(scheme, 'nesterov')
    pp = (1 + beta) * (1 - alpha * lam);
    qq = beta * (1 - alpha * lam);
else
    error('scheme 只能取 heavy_ball 或 nesterov');
end

z_prev = c0;          % z_{-1} = z_0
z = c0;
ratios = zeros(steps, 1);
for k = 1:steps
    z_next = pp .* z - qq .* z_prev;
    z_prev = z;
    z = z_next;
    ratios(k) = norm(z) / norm(z_prev);
end

out = struct();
out.ratios = ratios;
out.w1 = median(ratios(11:2000));
out.w2 = median(ratios(2001:5000));
out.w3 = median(ratios(5001:20000));
end
