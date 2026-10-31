%% exp7_cg.m  第 2 节：共轭梯度法，并与最速下降 / 固定步长 GD / Nesterov 对比
%
%   题目要求：编写共轭梯度法求解 min f(x) = 1/2 ||Ax - b||^2，并与最速下降法、
%   步长 alpha = L/2 的梯度下降法、Nesterov 加速法比较。取 n = 50 与 n = 1000，
%   分别画 (a) 步数 - ln f(x_k)、(b) 步数 - ln ||x_k - x*||、
%   (c) CPU 时间 - ln ||x_k - x*||。
%
%   停止准则统一为 ||x_k - x*|| <= RTOL * ||x_0 - x*||（CG 用相对残差，二者同阶）；
%   一阶方法按规模设迭代上限（CAPS），跑不到精度的在图和表里都标成"未收敛"。
%
%   计时用 tic/toc 取进程实际时间。为让 CPU 时间可重复，BLAS 必须是单线程：
%   启动 Octave 前设置 OPENBLAS_NUM_THREADS=1（8 线程时 1000x1000 的 matvec
%   反而慢 6 倍，因为线程同步开销远大于计算本身）。

%% ------------------------------------------------------------------ 设置
C = set_style();

SEED = 0;
RTOL = 1e-10;
CASES = [50, 1000];

% 每个方法的迭代上限（按规模）：只为控制 CPU 开销
CAPS = struct( ...
    'n',   {50,   1000}, ...
    'sd',  {2000000, 20000}, ...
    'gd',  {3500000, 20000}, ...
    'nes', {300000, 150000});

KEYS = {'cg', 'sd', 'gd_half', 'nes'};
PNAME = {'CG', '最速下降', 'GD (alpha = L/2)', 'Nesterov'};
LEGLAB = {'CG', '最速下降（精确线搜索）', 'GD (alpha = L/2)', ...
          'Nesterov (alpha = 1/lam_max)'};
COL = {C.CG, C.SD, C.GD, C.Nesterov};

%% ------------------------------------------------------------------ 计算
D = struct([]);
for ic = 1:numel(CASES)
    n = CASES(ic);
    cap = CAPS(ic);

    p = mk_problem(n, SEED);
    x0 = zeros(n, 1);
    nd0 = norm(x0 - p.x_star);

    sig = sqrt(p.kappa);
    beta_std = (sig - 1) / (sig + 1);     % 强凸问题的经典动量系数
    alpha_nes = 1 / p.lam_max;            % Nesterov 格式要求 alpha <= 1/L
    alpha_gd = 0.5 * p.L;                 % = 1/lam_max

    rho_gd = max(abs(1 - alpha_gd * p.eigvals));
    rho_sd = (p.kappa - 1) / (p.kappa + 1);          % 精确线搜索的渐近速率
    rho_nes = mom_rho(p.eigvals, alpha_nes, beta_std, 'nesterov');

    fprintf('n = %d：lam_min = %.4e，lam_max = %.4e，kappa = %.4e，sqrt(kappa) = %.2f\n', ...
            n, p.lam_min, p.lam_max, p.kappa, sig);
    fprintf('        理论谱半径：GD(L/2) %.8f，SD %.8f，Nesterov %.8f（解析 1-1/sqrt(k) = %.8f）\n', ...
            rho_gd, rho_sd, rho_nes, 1 - 1 / sig);

    h = cell(1, 4);
    t0 = tic;
    h{1} = run_cg(p, x0, 20 * n, RTOL);
    fprintf('        CG         %6d 步  %7.3f s  %s\n', ...
            h{1}.iters, h{1}.time(end), yn(h{1}.converged));
    t0 = tic;
    h{2} = run_sd(p, x0, cap.sd, RTOL);
    fprintf('        最速下降   %6d 步  %7.3f s  %s\n', ...
            h{2}.iters, h{2}.time(end), yn(h{2}.converged));
    t0 = tic;
    h{3} = run_gd(p, x0, alpha_gd, cap.gd, RTOL);
    fprintf('        GD (L/2)   %6d 步  %7.3f s  %s\n', ...
            h{3}.iters, h{3}.time(end), yn(h{3}.converged));
    t0 = tic;
    h{4} = run_nesterov(p, x0, alpha_nes, cap.nes, RTOL, [], beta_std);
    fprintf('        Nesterov   %6d 步  %7.3f s  %s\n\n', ...
            h{4}.iters, h{4}.time(end), yn(h{4}.converged));

    D(ic).n = n;
    D(ic).p = p;
    D(ic).h = h;
    D(ic).nd0 = nd0;
    D(ic).meta = struct('lambda_min', p.lam_min, 'lambda_max', p.lam_max, ...
        'kappa', p.kappa, 'sqrt_kappa', sig, 'L', p.L, ...
        'alpha_gd', alpha_gd, 'alpha_nesterov', alpha_nes, ...
        'beta_nesterov', beta_std, 'rho_gd_theory', rho_gd, ...
        'rho_sd_theory', rho_sd, 'rho_nes_theory', rho_nes, ...
        'rho_nes_analytic', 1 - 1 / sig, 'err0', nd0, 'rtol', RTOL);
end

%% ------------------------------------------------------------------ 表 8
HEADER = {'规模', '方法', '步数（->相对误差 1e-10）', '收敛', ...
          '末步 ln|x_k-x*|', 'CPU (s)', '相对 CG 的时间倍数'};
rows = cell(2 * numel(CASES), 7);
r = 0;
for ic = 1:numel(CASES)
    cg_time = D(ic).h{1}.time(end);
    for im = 1:4
        r = r + 1;
        hh = D(ic).h{im};
        rows{r, 1} = sprintf('n = %d', D(ic).n);
        rows{r, 2} = PNAME{im};
        if hh.converged
            rows{r, 3} = sprintf('%d', hh.iters);
        else
            rows{r, 3} = sprintf('>%d（到上限）', hh.iters);
        end
        rows{r, 4} = yn(hh.converged);
        rows{r, 5} = sprintf('%.3f', log(max(hh.err(end) * D(ic).nd0, 1e-300)));
        rows{r, 6} = sci(hh.time(end), 3);
        if im == 1
            rows{r, 7} = '1.0';
        else
            rows{r, 7} = sprintf('%.1f', hh.time(end) / cg_time);
        end
    end
end
print_table(HEADER, rows, ...
    '表 8  CG 与三种一阶方法的代价对比（停止准则：相对误差 1e-10）');

%% ------------------------------------------------------------------ 图 7
ylabs = {'ln f(x_k)', 'ln |x_k - x^{*}|_2', 'ln |x_k - x^{*}|_2'};
xlabs = {'迭代步数 k（对数轴）', '迭代步数 k（对数轴）', 'CPU 时间 (s，对数轴)'};
pname3 = {'步数 — ln f', '步数 — ln |x_k-x^{*}|', 'CPU — ln |x_k-x^{*}|'};

fig = figure('position', [0 0 1620 1080]);

for ic = 1:numel(CASES)
    nd0 = D(ic).nd0;
    for j = 1:3
        subplot(2, 3, (ic - 1) * 3 + j);
        hold on;

        xmin = inf; xmax = 0; ymin = inf; ymax = -inf;
        for im = 1:4
            hh = D(ic).h{im};
            if j == 3
                x = hh.time(:);
                keep = x > 1e-9;
            else
                x = (0:numel(hh.err) - 1)';
                keep = x >= 1;
            end
            if j == 1
                y = log(max(hh.f, 1e-300));
            else
                y = log(max(hh.err * nd0, 1e-300));
            end
            x = x(keep); y = y(keep);

            % 点数太多 gnuplot 画不动：按对数等距抽稀
            m = numel(x);
            if m > 2500
                idx = unique(round(logspace(0, log10(m), 2500)));
            else
                idx = 1:m;
            end
            plot(x(idx), y(idx), '-', 'color', COL{im}, 'linewidth', 1.5);

            xmin = min(xmin, x(1)); xmax = max(xmax, x(end));
            ymin = min(ymin, min(y)); ymax = max(ymax, max(y));
        end

        xlo = xmin / 1.4; xhi = xmax * 1.6;
        dy = max(ymax - ymin, 1);
        ylo = ymin - 0.12 * dy; yhi = ymax + 0.08 * dy;
        set(gca, 'xscale', 'log');
        xlim([xlo, xhi]); ylim([ylo, yhi]);
        grid on;
        xlabel(xlabs{j});
        ylabel(ylabs{j});
        title(sprintf('(%s) n = %d：%s', char('a' + (ic - 1) * 3 + j - 1), ...
                      CASES(ic), pname3{j}));
    end
end

% 图例放整幅图最上方通栏一行，避免压在曲线上
axl = axes('position', [0.0, 0.961, 1.0, 0.032]);
axis(axl, 'off');
hold(axl, 'on');
hp = zeros(1, 4);
for im = 1:4
    hp(im) = plot(axl, NaN, NaN, '-', 'color', COL{im}, 'linewidth', 1.5);
end
hl = legend(axl, hp, LEGLAB, 'orientation', 'horizontal');
set(hl, 'fontsize', 8.0);

save_fig(fig, 'fig7_cg');
fprintf('图 7 已保存：figures/fig7_cg.png\n\n');

%% ------------------------------------------------------------------ JSON
S = struct();
S.seed = SEED;
S.rtol = RTOL;
S.note_blas = 'OPENBLAS_NUM_THREADS=1';
for ic = 1:numel(CASES)
    nn = sprintf('n%d', CASES(ic));
    S.cases.(nn) = D(ic).meta;
    S.cases.(nn).max_iter = struct('sd', CAPS(ic).sd, 'gd_half', CAPS(ic).gd, ...
                                   'nes', CAPS(ic).nes);
    for im = 1:4
        hh = D(ic).h{im};
        rec = struct();
        rec.iters = hh.iters;
        rec.converged = hh.converged;
        for t = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
            rec.iters_to_rel.(sprintf('t%02d', round(-log10(t)))) = iter_to(hh.err, t);
        end
        rec.final_f = hh.f(end);
        rec.final_err = hh.err(end) * D(ic).nd0;
        rec.final_rel_err = hh.err(end);
        rec.cpu_time = hh.time(end);
        rec.cpu_per_iter = hh.time(end) / max(hh.iters, 1);
        S.methods.(sprintf('%s_%s', nn, KEYS{im})) = rec;
    end
end
save_json('sec2_cg', S);
fprintf('结果已写出：results/sec2_cg.json\n\n');

%% ------------------------------------------------------------------ 判读
fprintf('判读要点：\n');
for ic = 1:numel(CASES)
    hh = D(ic).h;
    fprintf('  n = %d：CG %d 步 / %.3f s（%s）；SD %d 步 / %.2f s（%s）；', ...
            CASES(ic), hh{1}.iters, hh{1}.time(end), yn(hh{1}.converged), ...
            hh{2}.iters, hh{2}.time(end), yn(hh{2}.converged));
    fprintf('GD %d 步 / %.2f s（%s）；Nesterov %d 步 / %.2f s（%s）\n', ...
            hh{3}.iters, hh{3}.time(end), yn(hh{3}.converged), ...
            hh{4}.iters, hh{4}.time(end), yn(hh{4}.converged));
end
