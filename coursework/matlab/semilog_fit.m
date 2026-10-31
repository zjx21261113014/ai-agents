function out = semilog_fit(k, e, start_frac, end_frac)
%SEMILOG_FIT 在 (k, ln e) 上做最小二乘直线拟合，判据 1 的定量版本
%   斜率 s < 0 且 R^2 接近 1 说明 semilogy 图上是直线，即 R-线性收敛；
%   收敛因子 rho = exp(s)。
%   只取序列的 [start_frac, end_frac] 段，避免起步暂态和末尾的舍入噪声。

if nargin < 3, start_frac = 0.30; end
if nargin < 4, end_frac = 0.95; end

k = k(:);
e = e(:);
m = isfinite(e) & (e > 0);
k = k(m);
e = e(m);
N = numel(e);

out = struct('slope', -Inf, 'rate', 1.0, 'r2', 0.0, 'nfit', 0);
if N < 4
    return;
end

i0 = max(1, ceil(start_frac * N));
i1 = min(N, floor(end_frac * N));
if i1 - i0 < 2
    i0 = 1;
    i1 = N;
end
ks = k(i0:i1);
ys = log(e(i0:i1));

c = polyfit(ks, ys, 1);
s = c(1);
yhat = polyval(c, ks);
ss_res = sum((ys - yhat).^2);
ss_tot = sum((ys - mean(ys)).^2);
if ss_tot > 0
    r2 = 1 - ss_res / ss_tot;
else
    r2 = 1.0;
end

out.slope = s;
out.rate = exp(s);
out.r2 = r2;
out.nfit = i1 - i0 + 1;
end
