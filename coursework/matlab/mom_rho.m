function rho = mom_rho(eigvals, alpha, beta, scheme)
%MOM_RHO 动量法在二次问题上的精确谱半径
%   在特征坐标下每个分量满足二阶线性递推
%     Heavy Ball: z_{k+1} = (1 + beta - alpha*lam) z_k - beta z_{k-1}
%     Nesterov  : z_{k+1} = (1+beta)(1 - alpha*lam) z_k - beta (1 - alpha*lam) z_{k-1}
%   单步收缩率的上确界就是这些二阶递推的特征根模在整条谱上的最大值。
%   递推形式为 z_{k+1} = p z_k - q z_{k-1}，特征方程 t^2 - p t + q = 0。

lam = eigvals(:);
if strcmpi(scheme, 'heavy_ball') || strcmpi(scheme, 'hb')
    pp = 1 + beta - alpha * lam;
    qq = beta * ones(size(lam));
else
    pp = (1 + beta) * (1 - alpha * lam);
    qq = beta * (1 - alpha * lam);
end

disc = pp.^2 - 4 * qq;
r = zeros(size(lam));
% 复根：两根模相同且等于 sqrt(q)
neg = disc < 0;
if any(neg)
    r(neg) = sqrt(abs(qq(neg)));
end
% 实根：取模较大的那个
pos = ~neg;
if any(pos)
    sq = sqrt(disc(pos));
    r1 = abs((pp(pos) + sq) / 2);
    r2 = abs((pp(pos) - sq) / 2);
    r(pos) = max(r1, r2);
end
rho = max(r);
end
