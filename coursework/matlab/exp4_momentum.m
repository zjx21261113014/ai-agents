%% 第 5 题：在相同步长 alpha 与相同初值 x0 下对比 GD / Heavy Ball / Nesterov
%   主对比  alpha = L/2 = 1/lam_max（三种方法都稳定，也是 Nesterov 常数动量的最优步长）
%   极端    alpha = L = 2/lam_max（GD 的收敛上界）：GD 停滞，Heavy Ball 仍收敛
% 公平性：题目要求三种方法同一个 alpha。Heavy Ball 的经典 beta 是配 alpha = 2L 推导的，
% 搬到 alpha = L/2 上已非最优，故每种动量法都在给定 alpha 下重新一维搜索取最优 beta，
% 另附"经典 beta"作对照。

clear; clc;
C = set_style();

N = 50;
SEED = 0;
RTOL = 1e-10;
TOLS = [1e-4, 1e-6, 1e-8, 1e-10];

p = mk_problem(N, SEED);
x0 = zeros(N, 1);
L = p.L;
lam_min = p.lam_min;
lam_max = p.lam_max;
kappa = p.kappa;
sq = sqrt(kappa);

alpha_main = L / 2;          % = 1/lam_max
alpha_edge = L;              % = 2/lam_max

beta_hb_classic = ((sq - 1) / (sq + 1))^2;    % 配 alpha_HB = 4/(sqrt(lam_max)+sqrt(lam_min))^2
beta_nes_classic = (sq - 1) / (sq + 1);       % 配 alpha = 1/lam_max

hb_tune = best_beta(p.eigvals, alpha_main, 'heavy_ball');
nes_tune = best_beta(p.eigvals, alpha_main, 'nesterov');
hb_tune_edge = best_beta(p.eigvals, alpha_edge, 'heavy_ball');
nes_tune_edge = best_beta(p.eigvals, alpha_edge, 'nesterov');

rho_gd_main = max(abs(1 - alpha_main * lam_min), abs(1 - alpha_main * lam_max));
rho_gd_edge = max(abs(1 - alpha_edge * lam_min), abs(1 - alpha_edge * lam_max));

fprintf('n = %d, seed = %d, kappa(Q) = %.4e, sqrt(kappa) = %.4f\n', N, SEED, kappa, sq);
fprintf('L = 2/lam_max = %.6e\n', L);
fprintf('主对比 alpha = L/2 = %.6e；极端 alpha = L = %.6e\n', alpha_main, alpha_edge);
fprintf('GD 的 rho：alpha=L/2 时 1-1/kappa = %.8f；alpha=L 时 |1-2| = %.8f\n', ...
        rho_gd_main, rho_gd_edge);
fprintf('动量系数（alpha=L/2）：HB 经典 %.6f -> rho=%.8f；HB 最优 %.6f -> rho=%.8f\n', ...
        beta_hb_classic, mom_rho(p.eigvals, alpha_main, beta_hb_classic, 'heavy_ball'), ...
        hb_tune.beta, hb_tune.rho);
fprintf('                      Nes 经典 %.6f -> rho=%.8f；Nes 最优 %.6f -> rho=%.8f\n', ...
        beta_nes_classic, mom_rho(p.eigvals, alpha_main, beta_nes_classic, 'nesterov'), ...
        nes_tune.beta, nes_tune.rho);
fprintf('动量系数（alpha=L）  ：HB 经典 %.6f -> rho=%.8f；HB 最优 %.6f -> rho=%.8f\n', ...
        beta_hb_classic, mom_rho(p.eigvals, alpha_edge, beta_hb_classic, 'heavy_ball'), ...
        hb_tune_edge.beta, hb_tune_edge.rho);
fprintf('                      Nes 经典 %.6f -> rho=%.8f（大于 1，发散）；Nes 最优 %.6f\n\n', ...
        beta_nes_classic, mom_rho(p.eigvals, alpha_edge, beta_nes_classic, 'nesterov'), ...
        nes_tune_edge.beta);
fprintf('理论加速比：GD 每步约 1-1/kappa，动量法约 1-1/sqrt(kappa)，步数比约 sqrt(kappa) = %.1f\n\n', sq);

nm = {'GD', 'HB 经典', 'HB 最优', 'Nes 常数', 'Nes 默认'};
HEADER = {'方法', 'beta', 'rho 理论', 'rho 实测', 'R^2', '1e-4', '1e-6', '1e-8', ...
          '1e-10', '步数', '加速', 'CPU'};

%% 第一部分：alpha = L/2 的主对比
beta_main = {'—', sprintf('%.6f', beta_hb_classic), sprintf('%.6f', hb_tune.beta), ...
             sprintf('%.6f', beta_nes_classic), '(k-1)/(k+2)'};
rho_th_main = [rho_gd_main, mom_rho(p.eigvals, alpha_main, beta_hb_classic, 'heavy_ball'), ...
               hb_tune.rho, mom_rho(p.eigvals, alpha_main, beta_nes_classic, 'nesterov'), NaN];

fprintf('第一部分：alpha = L/2，开始运行 ...\n');
h_main = cell(1, 5);
h_main{1} = run_gd(p, x0, alpha_main, 800000, RTOL);
h_main{2} = run_hb(p, x0, alpha_main, beta_hb_classic, 100000, RTOL);
h_main{3} = run_hb(p, x0, alpha_main, hb_tune.beta, 100000, RTOL);
h_main{4} = run_nesterov(p, x0, alpha_main, 100000, RTOL, 0, beta_nes_classic);
h_main{5} = run_nesterov(p, x0, alpha_main, 800000, RTOL);
for i = 1:5
    fprintf('  %-10s %7d 步\n', nm{i}, h_main{i}.iters);
end
fprintf('\n');

rec_main = cell(1, 5);
rows = cell(5, 12);
for i = 1:5
    rec_main{i} = describe_run(nm{i}, beta_main{i}, rho_th_main(i), h_main{i}, TOLS);
    rows(i, :) = row_of(rec_main{i}, rec_main{1}.iters);
end
print_table(HEADER, rows, ...
    sprintf('表 4  同一 alpha = L/2 下三种方法的数值对比（n=%d, seed=%d, 相对容差 %.0e）', ...
            N, SEED, RTOL));

gd_iters = rec_main{1}.iters;
fprintf('注：rho 实测为 semilogy 尾段（30%%~95%%）的拟合因子；加速 = GD 步数 / 该方法步数。\n');
fprintf('注：alpha = L/2 恰是 Nesterov 常数动量的最优步长，故其经典 beta 已是该步长下的最优 beta；\n');
fprintf('    Heavy Ball 的经典 beta 是配 alpha = 2L 推出的，搬到 alpha = L/2 上需要重新取 beta。\n');
fprintf('注：Nesterov 默认 beta_k 在强凸问题上只比 GD 快 %.1f 倍，R^2 仅 %.2f，误差不是严格几何衰减。\n\n', ...
        gd_iters / rec_main{5}.iters, rec_main{5}.semilog_r2);

%% 第二部分：alpha = L 的极端对比
beta_edge = {'—', sprintf('%.6f', beta_hb_classic), sprintf('%.6f', hb_tune_edge.beta), ...
             sprintf('%.6f', beta_nes_classic), '(k-1)/(k+2)'};
rho_th_edge = [rho_gd_edge, mom_rho(p.eigvals, alpha_edge, beta_hb_classic, 'heavy_ball'), ...
               hb_tune_edge.rho, mom_rho(p.eigvals, alpha_edge, beta_nes_classic, 'nesterov'), NaN];

fprintf('第二部分：alpha = L，开始运行 ...\n');
h_edge = cell(1, 5);
h_edge{1} = run_gd(p, x0, alpha_edge, 60000, RTOL);
h_edge{2} = run_hb(p, x0, alpha_edge, beta_hb_classic, 100000, RTOL);
h_edge{3} = run_hb(p, x0, alpha_edge, hb_tune_edge.beta, 100000, RTOL);
h_edge{4} = run_nesterov(p, x0, alpha_edge, 150, RTOL, 0, beta_nes_classic);
h_edge{5} = run_nesterov(p, x0, alpha_edge, 150, RTOL);
for i = 1:5
    fprintf('  %-10s %7d 步\n', nm{i}, h_edge{i}.iters);
end
fprintf('\n');

rec_edge = cell(1, 5);
rows = cell(5, 12);
for i = 1:5
    rec_edge{i} = describe_run(nm{i}, beta_edge{i}, rho_th_edge(i), h_edge{i}, TOLS);
    rows(i, :) = row_of(rec_edge{i}, rec_edge{1}.iters);
end
print_table(HEADER, rows, ...
    sprintf('表 5  极端步长 alpha = L = 2/lam_max 下三种方法的数值对比（n=%d, seed=%d）', N, SEED));

fprintf('注：alpha = L 时 GD 的 rho = |1-2| = 1（lam_max 方向分量永不衰减），60000 步后相对误差仍有 %.3f；\n', ...
        rec_edge{1}.final_err);
fprintf('    Heavy Ball 的稳定区间是 0 < alpha < 4/lam_max = 2L，故仍收敛。\n');
fprintf('注：Nesterov 的梯度取自外推点 y_k，外推放大高频分量，稳定步长上限退回 alpha < 1/lam_max；\n');
fprintf('    alpha = L 时任何 beta>0 都发散（rho = %.4f），最优 beta 只能是 0（退化为 GD）。\n\n', ...
        rho_th_edge(4));

%% 图 4
fig = figure('position', [0 0 1340 920]);
XMAX = 600000;
XE = 6000;
XR = 4000;
r0 = p.A * x0 - p.b;
f0 = 0.5 * (r0' * r0);

kk = unique(round(logspace(0, log10(XMAX), 2500)));
kk = kk(kk >= 1 & kk <= XMAX);
kk1 = kk + 1;                  % f(k) / err(k) 存在下标 k+1 处（下标 1 对应 k = 0）

colv = {C.GD, C.HB, C.HB, C.Nesterov, C.Nesterov};
sty = {'-', '--', '-', '-', '--'};

% (a) 步数 — f(x_k)
subplot(2, 2, 1);
hold on;
hl = [];
for i = 1:5
    v = h_main{i}.f;
    n = min(numel(v), XMAX + 1);
    s = kk1(kk1 <= n);
    if i == 5
        hl(end + 1) = loglog(s - 1, max(cummax(v(s)), realmin), '--', 'color', colv{i}, 'linewidth', 1.1);
        hl(end + 1) = loglog(s - 1, max(cummin(v(s)), realmin), ':', 'color', colv{i}, 'linewidth', 1.1);
    else
        hl(end + 1) = loglog(s - 1, max(v(s), realmin), sty{i}, 'color', colv{i}, 'linewidth', 1.5);
    end
end
set(gca, 'xscale', 'log', 'yscale', 'log'); grid on;
xlim([1, XMAX]); ylim([1e-28, 3 * f0]);
set(gca, 'ytick', 10.^(0:-4:-24));
xlabel('迭代步数 k（对数轴）'); ylabel('f(x_k)');
title('(a) 同一 alpha = L/2：步数 — f(x_k)');
legend(hl, {'GD', 'HB 经典 beta', 'HB 最优 beta', 'Nes 常数 beta', ...
            'Nes 默认 beta_k（上包络）', 'Nes 默认 beta_k（下包络）'}, 'location', 'southwest');

% (b) 步数 — ||x_k - x*||
subplot(2, 2, 2);
hold on;
hl = [];
for i = 1:5
    v = h_main{i}.err;
    n = min(numel(v), XMAX + 1);
    s = kk1(kk1 <= n);
    if i == 5
        hl(end + 1) = loglog(s - 1, max(cummax(v(s)), realmin), '--', 'color', colv{i}, 'linewidth', 1.1);
        hl(end + 1) = loglog(s - 1, max(cummin(v(s)), realmin), ':', 'color', colv{i}, 'linewidth', 1.1);
    else
        hl(end + 1) = loglog(s - 1, max(v(s), realmin), sty{i}, 'color', colv{i}, 'linewidth', 1.5);
    end
end
set(gca, 'xscale', 'log', 'yscale', 'log'); grid on;
xlim([1, XMAX]); ylim([1e-12, 1e2]);
xlabel('迭代步数 k（对数轴）'); ylabel('||x_k - x^*||_2');
title('(b) 同一 alpha = L/2：步数 — ||x_k - x^*||');
legend(hl, {'GD', 'HB 经典 beta', 'HB 最优 beta', 'Nes 常数 beta', ...
            'Nes 默认 beta_k（上包络）', 'Nes 默认 beta_k（下包络）'}, 'location', 'southwest');

% (c) rho_k 演化与精确谱半径
subplot(2, 2, 3);
hold on;
selc = [1, 3, 4];
colc = {C.GD, C.HB, C.Nesterov};
for j = 1:3
    i = selc(j);
    e = h_main{i}.err;
    n = min(numel(e), XR + 1);
    e = e(1:n);
    rk = e(2:end) ./ max(e(1:end-1), 1e-300);
    kx = (1:n-1)';
    m = isfinite(rk) & rk > 0.985 & rk < 1.012;
    plot(kx(m), rk(m), '.', 'color', colc{j}, 'markersize', 3);
end
h1 = plot([0, XR], [rho_gd_main, rho_gd_main], ':', 'color', C.GD, 'linewidth', 1.2);
h2 = plot([0, XR], [hb_tune.rho, hb_tune.rho], ':', 'color', C.HB, 'linewidth', 1.2);
h3 = plot([0, XR], [nes_tune.rho, nes_tune.rho], ':', 'color', C.Nesterov, 'linewidth', 1.2);
grid on;
xlim([0, XR]); ylim([0.988, 1.004]);
xlabel('迭代步数 k'); ylabel('\rho_k = e(k+1)/e(k)');
title('(c) 判据 2：\rho_k 与精确谱半径（alpha = L/2）');
legend([h1, h2, h3], {sprintf('GD 理论 rho = %.6f', rho_gd_main), ...
       sprintf('HB 最优 rho = %.6f', hb_tune.rho), ...
       sprintf('Nes 常数 rho = %.6f', nes_tune.rho)}, 'location', 'east');
text(0.02 * XR, 0.9885, sprintf('动量法 rho_k 有阻尼振荡，最终落在 1-1/sqrt(kappa) = %.6f 附近', ...
     1 - 1 / sq), 'fontsize', 9);

% (d) 极端情形 alpha = L
subplot(2, 2, 4);
hold on;
lss = {'-', '--', '-', '-', '--'};
hl = zeros(1, 5);
for i = 1:5
    e = h_edge{i}.err;
    n = min(numel(e), XE + 1);
    hl(i) = semilogy((0:n-1)', min(max(e(1:n), 1e-16), 1e12), lss{i}, ...
                     'color', colv{i}, 'linewidth', 1.5);
end
set(gca, 'yscale', 'log'); grid on;
xlim([0, XE]); ylim([1e-12, 1e3]);
set(gca, 'ytick', 10.^(-12:3:3));
xlabel('迭代步数 k'); ylabel('||x_k - x^*||_2');
title('(d) 极端情形 alpha = L = 2/lam_max：GD 停滞，动量法仍收敛');
legend(hl, {'GD', 'HB 经典 beta', 'HB 最优 beta', 'Nes 常数 beta', 'Nes 默认 beta_k'}, ...
       'location', 'southwest');
text(0.42 * XE, 5e2, 'GD 停滞在 1.2e-02，lam_max 分量不衰减', 'fontsize', 9);
text(0.42 * XE, 5e1, 'Nes：外推点取梯度，alpha = L 时发散', 'fontsize', 9);
text(0.42 * XE, 5e0, 'HB：稳定区间 0 < alpha < 2L，仍收敛', 'fontsize', 9);

%% 结果
P = struct();
P.n = N; P.seed = SEED; P.rtol = RTOL;
P.kappa = kappa; P.L = L;
P.alpha_main = alpha_main; P.alpha_edge = alpha_edge;
P.beta_hb_classic = beta_hb_classic;
P.beta_nes_classic = beta_nes_classic;
P.hb_tuned_main = hb_tune;
P.nes_tuned_main = nes_tune;
P.hb_tuned_edge = hb_tune_edge;
P.nes_tuned_edge = nes_tune_edge;
P.rho_gd_main = rho_gd_main;
P.rho_gd_edge = rho_gd_edge;
P.rho_1_minus_inv_sqrt_kappa = 1 - 1 / sq;
P.speedup_sqrt_kappa = sq;
for i = 1:5
    P.main.(sprintf('m%d', i)) = rec_main{i};
    P.edge.(sprintf('m%d', i)) = rec_edge{i};
end

save_fig(fig, 'fig4_momentum');
fprintf('图 4 已保存：figures/fig4_momentum.png\n');
save_json('q5_momentum', P);
fprintf('已写出 results/q5_momentum.json\n');
