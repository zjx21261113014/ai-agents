%% 第 6 题：两种 Barzilai-Borwein（BB）方法 vs 梯度下降法（n = 50, seed = 0）
%   BB 的步长由最近两步的信息给出（s_k = x_k - x_(k-1), y_k = g_k - g_(k-1)）：
%     BB1: alpha_k = (s's)/(s'y) = 1 / R_Q(s)
%     BB2: alpha_k = (s'y)/(y'y) = R_Q(s) / R_(Q^2)(s)
%   二次函数上 s'y = s'Qs > 0，两种步长都自动落在 [1/lam_max, 1/lam_min] 内，
%   右端是 GD 稳定上界 2/lam_max 的 kappa/2 倍 —— 这正是 BB 快、但 f 不单调的原因。
%   首步取 alpha0 = 1/lam_max，与固定步长 GD 的第一步相同，保证起点公平。

clear; clc;
C = set_style();

N = 50;
SEED = 0;
RTOL = 1e-10;
TOLS = [1e-4, 1e-6, 1e-8, 1e-10];

p = mk_problem(N, SEED);
x0 = zeros(N, 1);

L = p.L;
alpha_gd = L / 2;             % = 1/lam_max，与第 5 题的主对比步长一致
alpha_star = p.alpha_opt;     % = 2/(lam_min+lam_max)

rho_gd_half = max(abs(1 - alpha_gd * p.eigvals));
rho_gd_star = max(abs(1 - alpha_star * p.eigvals));

lo = 1 / p.lam_max;
hi = 1 / p.lam_min;

fprintf('n = %d, seed = %d, kappa(Q) = %.4e, sqrt(kappa) = %.4f\n', N, SEED, p.kappa, sqrt(p.kappa));
fprintf('lam_min = %.6e, lam_max = %.6e\n', p.lam_min, p.lam_max);
fprintf('GD 步长：alpha = L/2 = 1/lam_max = %.6e（rho 理论 = %.8f）；alpha* = %.6e（rho 理论 = %.8f）\n', ...
        alpha_gd, rho_gd_half, alpha_star, rho_gd_star);
fprintf('BB 步长的理论区间：[1/lam_max, 1/lam_min] = [%.6e, %.6e]，\n', lo, hi);
fprintf('  其右端是 GD 稳定上界 2/lam_max = %.6e 的 %.1f 倍\n\n', L, p.kappa / 2);

fprintf('开始运行 ...\n');
h_gd_half = run_gd(p, x0, alpha_gd, 800000, RTOL);
h_gd_opt = run_gd(p, x0, alpha_star, 800000, RTOL);
h_bb1 = run_bb(p, x0, 1, 400000, RTOL);
h_bb2 = run_bb(p, x0, 2, 400000, RTOL);
fprintf('  GD(L/2)  %7d 步\n', h_gd_half.iters);
fprintf('  GD(a*)   %7d 步\n', h_gd_opt.iters);
fprintf('  BB1      %7d 步\n', h_bb1.iters);
fprintf('  BB2      %7d 步\n\n', h_bb2.iters);

nm = {'GD(L/2)', 'GD(a*)', 'BB1', 'BB2'};
hists = {h_gd_half, h_gd_opt, h_bb1, h_bb2};
recs = cell(1, 4);
for i = 1:4
    recs{i} = describe_bb(nm{i}, hists{i}, p, TOLS);
end
ref_iters = recs{1}.iters;

%% 表 6
HEADER = {'方法', 'rho 拟合', 'R^2', 'rho 尾部中位数', 'rho^f 中位数', 'rho^f>1', ...
          '几何平均速率', '1e-4', '1e-6', '1e-8', '1e-10', '步数', '对 GD(L/2)', 'CPU'};
rows = cell(4, 14);
for i = 1:4
    rows(i, :) = row_of_bb(recs{i}, ref_iters, TOLS);
end
print_table(HEADER, rows, ...
    sprintf('表 6  BB1 / BB2 与梯度下降法的数值对比（n=%d, seed=%d, 同一 x0=0）', N, SEED));

fprintf('判读要点：\n');
fprintf('  1. BB 的误差并不单调：按 f（即 Q-范数）口径有 %.1f%%（BB1）/ %.1f%%（BB2）的步子是变大的，\n', ...
        100 * recs{3}.frac_rho_f_gt1, 100 * recs{4}.frac_rho_f_gt1);
fprintf('     f 分别上升过 %d / %d 次；GD 严格单调。\n', recs{3}.n_f_increase, recs{4}.n_f_increase);
fprintf('     所以对 BB 不能用单步 rho_k 判断好坏，必须看几何平均速率 (e_N/e_0)^(1/N)。\n');
fprintf('  2. 几何平均速率：GD(L/2) %.6f，GD(a*) %.6f，BB1 %.6f，BB2 %.6f。\n', ...
        recs{1}.geomean_rate, recs{2}.geomean_rate, recs{3}.geomean_rate, recs{4}.geomean_rate);
fprintf('     BB 的平均速率明显更小 → 步数少 %.1f 倍（BB1）/ %.1f 倍（BB2）。\n', ...
        ref_iters / max(recs{3}.iters, 1), ref_iters / max(recs{4}.iters, 1));
fprintf('     按 f 口径 BB1 的几何平均速率为 %.6f，BB2 为 %.6f。\n', ...
        recs{3}.energy_geomean_rate, recs{4}.energy_geomean_rate);
fprintf('  3. semilog 拟合的 R^2 只有 %.3f / %.3f（GD(L/2) 为 %.3f）：BB 不是几何收敛，\n', ...
        recs{3}.semilog_r2, recs{4}.semilog_r2, recs{1}.semilog_r2);
fprintf('     ln e_k 呈阶梯下台阶，每次下台阶对应步长被"重新校准"到 lam_min 方向。\n');
fprintf('  4. 两种口径的反差：BB2 的 2-范数几乎从不上升（仅 %.2f%% 的步），\n', ...
        100 * recs{4}.frac_rho_gt1);
fprintf('     但 f（Q-范数）上升了 %.0f%% 的步。说明 BB 的过冲主要发生在高曲率（lam_max）方向：\n', ...
        100 * recs{4}.frac_rho_f_gt1);
fprintf('     Q-范数被这一步放大，而 2-范数不受影响。选用哪个量度"单调"必须以结论所需为准。\n');
fprintf('  5. BB1 / BB2 的步数比为 %d : %d = %.2f。\n', recs{3}.iters, recs{4}.iters, ...
        recs{4}.iters / max(recs{3}.iters, 1));
fprintf('     BB1 的 alpha_k = 1/R_Q(s) 直接取瑞利商的倒数，当 s 主要含 lam_min 方向分量时 alpha_k 可以很大，\n');
fprintf('     一次性消掉慢方向，故台阶宽、总步数少；BB2 的 alpha_k 对高曲率方向更敏感，步长偏保守。\n\n');

%% 图 5
fig = figure('position', [0 0 1340 920]);
XMAX = 600000;
r0 = p.A * x0 - p.b;
f0 = 0.5 * (r0' * r0);

kk = unique(round(logspace(0, log10(XMAX), 2500)));
kk = kk(kk >= 1 & kk <= XMAX);
kk1 = kk + 1;                 % f(k) / err(k) 存在下标 k+1 处

colv = {C.GD, C.HB, C.BB1, C.BB2};

% (a) 步数 — f(x_k)
subplot(2, 2, 1);
hold on;
hl = zeros(1, 4);
for i = 1:4
    v = hists{i}.f;
    n = min(numel(v), XMAX + 1);
    s = kk1(kk1 <= n);
    hl(i) = loglog(s - 1, max(v(s), realmin), '-', 'color', colv{i}, 'linewidth', 1.2);
end
set(gca, 'xscale', 'log', 'yscale', 'log'); grid on;
xlim([1, XMAX]); ylim([1e-28, 3 * f0]);
set(gca, 'ytick', 10.^(-24:4:0));   % 刻度必须递增，MATLAB 不接受递减向量
xlabel('迭代步数 k（对数轴）'); ylabel('f(x_k)');
title('(a) 步数 — f(x_k)：BB 非单调、整体更快');
legend(hl, nm, 'location', 'southwest');
text(3e3, 3e-26, '锯齿 = f 非单调', 'fontsize', 9);

% (b) 步数 — ||x_k - x*||
subplot(2, 2, 2);
hold on;
hl = zeros(1, 4);
for i = 1:4
    v = hists{i}.err;
    n = min(numel(v), XMAX + 1);
    s = kk1(kk1 <= n);
    hl(i) = loglog(s - 1, max(v(s), realmin), '-', 'color', colv{i}, 'linewidth', 1.2);
end
set(gca, 'xscale', 'log', 'yscale', 'log'); grid on;
xlim([1, XMAX]); ylim([1e-12, 1e2]);
xlabel('迭代步数 k（对数轴）'); ylabel('||x_k - x^*||_2');
title('(b) 步数 — ||x_k - x^*||_2：阶梯式下降');
legend(hl, nm, 'location', 'southwest');
text(2e4, 3e-11, '台阶 = 步长校准到最慢方向', 'fontsize', 9);

% (c) BB 的自适应步长
subplot(2, 2, 3);
hold on;
hb1 = plot([1, XMAX], [hi, hi], '--', 'color', C.ref, 'linewidth', 1.0);
hb2 = plot([1, XMAX], [L, L], '--', 'color', C.GD, 'linewidth', 1.0);
hb3 = plot([1, XMAX], [alpha_star, alpha_star], ':', 'color', C.HB, 'linewidth', 1.3);
ha = zeros(1, 2);
for j = 1:2
    i = 2 + j;
    a = hists{i}.info.alphas(:);
    n = min(numel(a), XMAX);
    s = kk(kk <= n);
    ha(j) = loglog(s, a(s), '-', 'color', colv{i}, 'linewidth', 0.7);
end
set(gca, 'xscale', 'log', 'yscale', 'log'); grid on;
xlim([1, XMAX]); ylim([lo / 2, hi * 1.45]);
xlabel('迭代步数 k（对数轴）'); ylabel('步长 alpha_k');
title('(c) BB 的自适应步长：剧烈摆动但不越界');
legend([hb1, hb2, hb3, ha], {'1/lam_min（上界）', 'L = 2/lam_max', ...
       'alpha*', 'BB1', 'BB2'}, 'location', 'northeast', 'fontsize', 8);
text(1.2, 2.9e-3, sprintf('1/lam_min 是 L 的 %.0f 倍', p.kappa / 2), 'fontsize', 8);

% (d) 判据 2：rho^f_k 的分布
subplot(2, 2, 4);
hold on;
edges = linspace(-3, 3, 61);
xc = edges(1:end-1) + 0.05;
sel = [2, 3, 4];
hd = zeros(1, 3);
for j = 1:3
    i = sel(j);
    energy = hists{i}.f(:) - p.f_star;
    rho = ratios_of(energy);
    rho = rho(rho > 0);
    lg = log10(rho);
    cnt = histc(lg, edges);
    cnt = cnt(1:end-1);
    w = cnt / max(sum(cnt), 1);
    w(w <= 0) = NaN;
    hd(j) = plot(xc, w, '-', 'color', colv{i}, 'linewidth', 1.6);
end
hv = plot([0, 0], [2e-3, 2], ':', 'color', C.ref, 'linewidth', 1.1);
set(gca, 'yscale', 'log'); grid on;
xlim([-3, 3]); ylim([2e-3, 2]);
set(gca, 'xtick', -3:1:3);
set(gca, 'ytick', 10.^(-3:1:0));
xlabel('log_{10} rho^f_k（x > 0 即该步 f 上升）');
ylabel('步数占比');
title('(d) 判据 2：单步收缩率的分布（x>0 即该步 f 上升）');
legend([hd, hv], {nm{2}, nm{3}, nm{4}, 'rho^f = 1'}, 'location', 'northeast', 'fontsize', 8);

%% 结果
P = struct();
P.n = N; P.seed = SEED; P.rtol = RTOL;
P.lam_min = p.lam_min; P.lam_max = p.lam_max; P.kappa = p.kappa;
P.alpha_gd_half = alpha_gd; P.alpha_star = alpha_star;
P.rho_gd_half = rho_gd_half; P.rho_gd_star = rho_gd_star;
P.bb_step_band = [lo, hi];
for i = 1:4
    P.methods.(sprintf('m%d', i)) = recs{i};
    P.speedup.(sprintf('m%d', i)) = ref_iters / max(recs{i}.iters, 1);
end

save_fig(fig, 'fig5_bb');
fprintf('图 5 已保存：figures/fig5_bb.png\n');
save_json('q6_bb', P);
fprintf('已写出 results/q6_bb.json\n');
