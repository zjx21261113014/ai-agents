function out = loglog_fit(k, e, start_frac, end_frac)
%% LOGLOG_FIT 在 (ln k, ln e) 上做最小二乘直线拟合，判据 4 的定量版本
%   若斜率在整段上大致为常数 -p，则 e_k ~ C k^(-p) 是代数收敛；
%   若斜率随时间变陡（本实验中 -25 到 -40 之类的量级），则是几何收敛，
%   "代数收敛阶" 这个概念不适用。

if nargin < 3, start_frac = 0.30; end
if nargin < 4, end_frac = 0.95; end

k = k(:);
e = e(:);
m = isfinite(e) & (e > 0) & (k > 0);
k = k(m);
e = e(m);
N = numel(e);

out = struct('slope', -Inf, 'order', 0.0, 'r2', 0.0);
if N < 4
    return;
end

i0 = max(1, ceil(start_frac * N));
i1 = min(N, floor(end_frac * N));
if i1 - i0 < 2
    i0 = 1;
    i1 = N;
end
ks = log(k(i0:i1));
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
out.order = -s;
out.r2 = r2;
end
