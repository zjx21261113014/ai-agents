function m = tail_median(v, tail_frac)
%TAIL_MEDIAN 取尾部若干比例有限值的中位数（对锯齿状序列比平均更稳健）
if nargin < 2, tail_frac = 0.1; end
v = v(:);
v = v(isfinite(v));
if isempty(v)
    m = 0;
    return;
end
nt = max(1, ceil(tail_frac * numel(v)));
m = median(v(end - nt + 1:end));
end
