function hh = plot_subsampled(k, y, col, lw)
%PLOT_SUBSAMPLED 半对数画图，点数过多时等间隔抽稀
%   上百万个点时 gnuplot 画不动，抽稀到约 4000 点，曲线形状不受影响。

n = numel(y);
st = max(1, ceil(n / 4000));
idx = 1:st:n;
if idx(end) ~= n
    idx(end + 1) = n;
end
hh = semilogy(k(idx), max(y(idx), realmin), '-', 'color', col, 'linewidth', lw);
end
