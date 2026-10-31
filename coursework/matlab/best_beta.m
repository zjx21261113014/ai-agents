function out = best_beta(eigvals, alpha, scheme, n_grid, n_refine)
%BEST_BETA 在给定步长 alpha 下，一维搜索使谱半径最小的动量系数 beta
%   动量法的经典 beta 是配自己的最优步长推出来的，搬到别的步长上并非最优；
%   本实验要求三种方法用同一个 alpha，所以每种动量法都按其 alpha 重新取 beta。
%   先用粗网格定位，再在相邻区间上做黄金分割细化。

if nargin < 4, n_grid = 2001; end
if nargin < 5, n_refine = 60; end

bgrid = linspace(0, 0.999999, n_grid);
rr = zeros(size(bgrid));
for i = 1:n_grid
    rr(i) = mom_rho(eigvals, alpha, bgrid(i), scheme);
end
[~, im] = min(rr);
lo = bgrid(max(1, im - 1));
hi = bgrid(min(n_grid, im + 1));

% 黄金分割（rr 在局部是单峰的，实测有效）
gr = (sqrt(5) - 1) / 2;
for i = 1:n_refine
    m1 = hi - gr * (hi - lo);
    m2 = lo + gr * (hi - lo);
    if mom_rho(eigvals, alpha, m1, scheme) < mom_rho(eigvals, alpha, m2, scheme)
        hi = m2;
    else
        lo = m1;
    end
end
b = (lo + hi) / 2;
out.beta = b;
out.rho = mom_rho(eigvals, alpha, b, scheme);
end
