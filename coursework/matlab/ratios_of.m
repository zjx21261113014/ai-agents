function r = ratios_of(v)
%RATIOS_OF r_k = v_k / v_(k-1)，丢弃模量过小（下溢、有效位数不足）的比值
%   f - f* 在收敛末期会下溢到 1e-26 量级，此时的比值由舍入噪声决定，没有意义。

v = v(:);
pos = v(~isnan(v) & ~isinf(v) & v > 0);
if isempty(pos)
    r = [];
    return;
end
floor_v = 1e-14 * pos(1);

r = nan(numel(v) - 1, 1);
for i = 1:numel(v) - 1
    if v(i) > floor_v
        r(i) = v(i + 1) / max(v(i), 1e-300);
    end
end
r = r(~isnan(r) & ~isinf(r));
end
