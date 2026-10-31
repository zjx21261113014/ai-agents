%% 第 4 题：怎么用画图看出收敛阶 / 收敛速率
% 把"阶"这件抽象的事拆成四个可视判据：
%   (a) semilogy(e_k — k) 是否为直线        -> 是否 R-线性（几何收敛）
%   (b) rho_k = e_{k+1}/e_k 是否趋于常数    -> 是否 Q-线性，常数即渐近压缩因子
%   (c) p_k = ln(e_{k+1}/e_k)/ln(e_k/e_{k-1}) 是否趋于 1 -> 线性阶
%   (d) loglog(e_k — k) 的斜率是否为常数    -> 代数收敛与几何收敛的区分
% 同一张表给出由四个判据得到的定量结论，并与理论 rho 对照。

clear; clc;
C = set_style();

N = 50;
SEED = 0;
RTOL = 1e-10;
XMAX = 5000;         % 快方法三千步内就收敛，前 5000 步够看清所有方法的形态

p = mk_problem(N, SEED);
x0 = zeros(N, 1);
L = p.L;
lam_min = p.lam_min;
lam_max = p.lam_max;
kappa = p.kappa;
sq = sqrt(kappa);

alpha_opt = p.alpha_opt;
rho_gd = max(abs(1 - alpha_opt * lam_min), abs(1 - alpha_opt * lam_max));
rho_sd = (kappa - 1) / (kappa + 1);          % 经典界（f 的界是它的平方）

% Heavy Ball：alpha = 4/(sqrt(lam_max)+sqrt(lam_min))^2, beta = ((sqrt(k)-1)/(sqrt(k)+1))^2
alpha_hb = 4 / (sqrt(lam_max) + sqrt(lam_min))^2;
beta_hb = ((sq - 1) / (sq + 1))^2;
% Nesterov 常数动量：alpha = 1/lam_max, beta = (sqrt(k)-1)/(sqrt(k)+1)
alpha_nes = 1 / lam_max;
beta_nes = (sq - 1) / (sq + 1);

rho_hb = mom_rho(p.eigvals, alpha_hb, beta_hb, 'heavy_ball');
rho_nes = mom_rho(p.eigvals, alpha_nes, beta_nes, 'nesterov');

fprintf('n = %d, seed = %d, kappa = %.4e\n', N, SEED, kappa);
fprintf('L = 2/lam_max = %.6e;  alpha* = %.6e;  alpha_HB = %.6e;  alpha_Nes = %.6e\n', ...
        L, alpha_opt, alpha_hb, alpha_nes);
fprintf('理论 rho：GD(alpha*) = %.8f; SD = %.8f; HB = %.8f; Nesterov = %.8f\n\n', ...
        rho_gd, rho_sd, rho_hb, rho_nes);

% 动量法到容差只要两三千步，可能还在暂态内；用特征坐标长程仿真核实
hb_long = mom_longrun(p, alpha_hb, beta_hb, 'heavy_ball', 20000);
nes_long = mom_longrun(p, alpha_nes, beta_nes, 'nesterov', 20000);
fprintf('动量法长程仿真的 rho_k 中位数：\n');
fprintf('  Heavy Ball  谱半径 = %.6f | [10,2000] %.6f  [2000,5000] %.6f  [5000,20000] %.6f\n', ...
        rho_hb, hb_long.w1, hb_long.w2, hb_long.w3);
fprintf('  Nesterov    谱半径 = %.6f | [10,2000] %.6f  [2000,5000] %.6f  [5000,20000] %.6f\n\n', ...
        rho_nes, nes_long.w1, nes_long.w2, nes_long.w3);

%% 六种方法
names = {'GD(alpha=alpha*)', '最速下降', 'Heavy Ball', 'Nesterov', 'BB1', 'CG'};
cols = {C.GD, C.SD, C.HB, C.Nesterov, C.BB1, C.CG};

fprintf('开始运行六种方法 ...\n');
hists = cell(1, 6);
hists{1} = run_gd(p, x0, alpha_opt, 400000, RTOL);
fprintf('  GD(alpha*) 完成，%d 步\n', hists{1}.iters);
hists{2} = run_sd(p, x0, 300000, RTOL);
fprintf('  最速下降   完成，%d 步\n', hists{2}.iters);
hists{3} = run_hb(p, x0, alpha_hb, beta_hb, 20000, RTOL);
fprintf('  Heavy Ball 完成，%d 步\n', hists{3}.iters);
hists{4} = run_nesterov(p, x0, alpha_nes, 20000, RTOL, 0, beta_nes);
fprintf('  Nesterov   完成，%d 步\n', hists{4}.iters);
hists{5} = run_bb(p, x0, 1, 50000, RTOL);
fprintf('  BB1        完成，%d 步\n', hists{5}.iters);
hists{6} = run_cg(p, x0, 200, 1e-14);
fprintf('  CG         完成，%d 步\n\n', hists{6}.iters);

rho_theory = [rho_gd, rho_sd, rho_hb, rho_nes, NaN, NaN];

%% 四项数值判据
rows_a = cell(6, 5);
rows_b = cell(6, 5);
keys = {'gd', 'sd', 'hb', 'nesterov', 'bb1', 'cg'};
P = struct();

for i = 1:6
    e = hists{i}.err;
    k = (0:numel(e) - 1)';
    fit = semilog_fit(k, e, 0.30, 0.95);
    lls = loglog_fit(k, e, 0.30, 0.95);
    [kk, pk] = coc_p(e);
    p_tail = tail_median(pk, 0.3);

    rk = e(2:end) ./ max(e(1:end-1), 1e-300);
    rho_tail = tail_median(rk, 0.1);

    if isfinite(p_tail) && abs(p_tail - 1) < 0.05
        verdict = 'Q-线性，阶 1';
    elseif i == 6
        verdict = '有限步终止';
    else
        verdict = '仅 R-线性';
    end

    if isfinite(rho_theory(i))
        theo_str = sprintf('%.8f', rho_theory(i));
    else
        theo_str = '—';
    end

    rows_a(i, :) = {names{i}, theo_str, sprintf('%.6f', fit.rate), ...
                    sprintf('%.6f', fit.r2), sprintf('%d', hists{i}.iters)};
    rows_b(i, :) = {names{i}, sprintf('%.6f', rho_tail), sprintf('%.3f', p_tail), ...
                    sprintf('%.3f', lls.slope), verdict};

    S = struct();
    S.iters = hists{i}.iters;
    S.converged = hists{i}.converged;
    S.cpu_time = hists{i}.time(end);
    if isfinite(rho_theory(i))
        S.rho_theory = rho_theory(i);
    else
        S.rho_theory = [];
    end
    S.rho_semilog_fit = fit.rate;
    S.semilog_slope = fit.slope;
    S.semilog_r2 = fit.r2;
    S.rho_tail_median = rho_tail;
    S.coc_tail_median = p_tail;
    S.loglog_slope = lls.slope;
    S.final_relative_err = e(end);
    S.verdict = verdict;
    P.(keys{i}) = S;
end

print_table({'方法', 'rho 理论', 'rho 拟合', 'R^2', '步数'}, rows_a, ...
    sprintf('表 3a  收敛速率与拟合质量（n=%d, seed=%d, 相对容差 %.0e）', N, SEED, RTOL));
print_table({'方法', 'rho_k 尾部中位数', 'COC p_k', 'loglog 斜率', '判定'}, rows_b, ...
    '表 3b  逐点判据与结论');

fprintf(['\n判据说明：R^2 接近 1 且 rho<1 说明 semilogy 是直线，即 R-线性；\n' ...
         '          rho_k 趋于常数 rho 说明 Q-线性；p_k 趋于 1 说明是线性阶；\n' ...
         '          loglog 斜率不是常数（随 k 变陡）说明是几何收敛而不是代数收敛。\n']);
[~, pk_bb] = coc_p(hists{5}.err);
fprintf('注：BB1 的 p_k 尾部中位数为 %.3f，rho_k 尾部中位数为 %.6f；\n', ...
        tail_median(pk_bb, 0.3), tail_median(hists{5}.err(2:end) ./ hists{5}.err(1:end-1), 0.1));
fprintf('    其 rho_k 呈锯齿状而非平滑趋于常数，仅在平均意义下才是 R-线性。\n');
fprintf('注：CG 在二次问题上 %d 步内达到机器精度，属有限步终止，“阶”不适用。\n', hists{6}.iters);

%% 图 3
fig = figure('position', [0 0 1340 900]);
orderly = [1, 2, 3, 4];

% (a) semilogy：是不是直线
subplot(2, 2, 1);
hold on;
hl = zeros(1, 6);
for i = 1:6
    e = hists{i}.err;
    n = min(numel(e), XMAX + 1);
    e = max(e(1:n), realmin);
    hl(i) = semilogy((0:n-1)', e, '-', 'color', cols{i}, 'linewidth', 1.4);
end
set(gca, 'yscale', 'log'); grid on;
xlim([0, XMAX]); ylim([1e-12, 4]);
set(gca, 'ytick', 10.^(0:-3:-12));
xlabel('迭代步数 k'); ylabel('相对误差 ||x_k-x^*|| / ||x_0-x^*||');
title('(a) 判据 1：semilogy 为直线即 R-线性');
legend(hl, {'GD(\alpha*)', '最速下降', 'Heavy Ball', 'Nesterov', 'BB1', 'CG'}, ...
       'location', 'southeast');
text(0.03 * XMAX, 1e-11, 'GD 与最速下降 5000 步只降 0.6 倍', 'fontsize', 9);

% (b) rho_k
subplot(2, 2, 2);
hold on;
for i = orderly
    e = hists{i}.err;
    n = min(numel(e), XMAX + 1);
    e = e(1:n);
    rk = e(2:end) ./ e(1:end-1);
    kk = (1:n-1)';
    m = isfinite(rk) & rk > 0.95 & rk < 1.06;
    plot(kk(m), rk(m), '.', 'color', cols{i}, 'markersize', 3);
end
h1 = plot([0, XMAX], [rho_gd, rho_gd], '--', 'color', C.GD, 'linewidth', 1.1);
h2 = plot([0, XMAX], [rho_sd, rho_sd], ':', 'color', C.SD, 'linewidth', 1.4);
h3 = plot([0, XMAX], [rho_hb, rho_hb], '--', 'color', C.HB, 'linewidth', 1.1);
h4 = plot([0, XMAX], [rho_nes, rho_nes], '--', 'color', C.Nesterov, 'linewidth', 1.1);
grid on;
xlim([0, XMAX]); ylim([0.95, 1.06]);
xlabel('迭代步数 k'); ylabel('\rho_k = e(k+1) / e(k)');
title('(b) 判据 2：\rho_k 趋于常数即 Q-线性');
legend([h1, h2, h3, h4], {sprintf('rho_GD = %.6f', rho_gd), ...
       sprintf('rho_SD = %.6f', rho_sd), sprintf('rho_HB = %.6f', rho_hb), ...
       sprintf('rho_Nes = %.6f', rho_nes)}, 'location', 'northeast', 'fontsize', 9);
text(0.02 * XMAX, 0.956, '动量法实测略高于谱半径，是暂态与拍频所致', 'fontsize', 9);

% (c) 计算型收敛阶
subplot(2, 2, 3);
hold on;
hp = zeros(1, 2);
for i = [orderly, 5]
    [kk, pk] = coc_p(hists{i}.err);
    m = isfinite(pk) & abs(pk) < 6 & kk <= XMAX;
    hp(1) = plot(kk(m), min(max(pk(m), -1.5), 5), '.', 'color', cols{i}, 'markersize', 3);
end
hp(2) = plot([0, XMAX], [1, 1], '--', 'color', [0.3 0.3 0.3], 'linewidth', 1.3);
grid on;
xlim([0, XMAX]); ylim([0.9, 2.6]);
xlabel('迭代步数 k'); ylabel('p_k = ln[e(k+1)/e(k)] / ln[e(k)/e(k-1)]');
title('(c) 判据 3：p_k 趋于 1 即线性收敛阶');
legend(hp, {'GD / SD / HB / Nesterov / BB1', '线性阶 p = 1'}, 'location', 'northeast');

% (d) loglog：几何还是代数
subplot(2, 2, 4);
hold on;
sel = [1, 5, 6];
for i = sel
    e = hists{i}.err;
    n = min(numel(e), XMAX + 1);
    hh = loglog((1:n-1)', max(e(2:n), realmin), '-', 'color', cols{i}, 'linewidth', 1.4);
end
kref = logspace(0.5, 3.7, 40);
loglog(kref, 10 ./ kref, '--', 'color', [0.45 0.45 0.45], 'linewidth', 1.1);
loglog(kref, 1e4 ./ kref .^ 2, ':', 'color', [0.45 0.45 0.45], 'linewidth', 1.1);
set(gca, 'xscale', 'log', 'yscale', 'log'); grid on;
xlim([1, XMAX]);
xlabel('迭代步数 k（对数轴）'); ylabel('相对误差（对数轴）');
title('(d) 判据 4：loglog 斜率非常数即几何收敛');
text(1.3, 1e-9, '虚线 k^{-1}，点线 k^{-2}', 'fontsize', 9);

P.theory = struct('alpha_opt', alpha_opt, 'rho_gd', rho_gd, 'rho_sd', rho_sd, ...
    'alpha_hb', alpha_hb, 'beta_hb', beta_hb, 'rho_hb_spectral', rho_hb, ...
    'alpha_nesterov', alpha_nes, 'beta_nesterov', beta_nes, ...
    'rho_nesterov_spectral', rho_nes, 'kappa', kappa, ...
    'longrun_hb_w1', hb_long.w1, 'longrun_hb_w2', hb_long.w2, 'longrun_hb_w3', hb_long.w3, ...
    'longrun_nes_w1', nes_long.w1, 'longrun_nes_w2', nes_long.w2, 'longrun_nes_w3', nes_long.w3);

save_fig(fig, 'fig3_order');
fprintf('\n图 3 已保存：figures/fig3_order.png\n');

save_json('q4_order', P);
fprintf('已写出 results/q4_order.json\n');
