%% 第 2、3 题：固定步长的收敛区间与不同步长的实验
% Q2  推出步长收敛区间 (0, 2/lambda_max)，并数值确认上界；
% Q3  在其中取 alpha = L/3, L/2, 3L/4, L（L = 2/lambda_max）做实验，
%     画出"迭代步数 — f(x_k)"与"迭代步数 — ||x_k - x*||"。

clear; clc;
C = set_style();

N = 50;
SEED = 0;
RTOL = 1e-10;

p = mk_problem(N, SEED);
x0 = zeros(N, 1);
L = p.L;

names = {'L/3', 'L/2', '3L/4', 'L'};
alphas = [L / 3, L / 2, 3 * L / 4, L];
caps = [2e6, 2e6, 2e6, 6e4];
cols = {C.HB, C.GD, C.Nesterov, C.BB1};

hist = cell(1, 4);
rows = cell(4, 13);
P = struct();
tols = [1e-4, 1e-6, 1e-8, 1e-10];

fprintf('alpha 的收敛区间：(0, 2/lambda_max) = (0, %.6e)\n', 2 / p.lam_max);
fprintf('L = 2/lambda_max = %.6e, alpha* = 2/(lambda_min+lambda_max) = %.6e\n\n', ...
        L, p.alpha_opt);

for i = 1:4
    h = run_gd(p, x0, alphas(i), caps(i), RTOL);
    hist{i} = h;

    rho_th = max(abs(1 - alphas(i) * p.lam_min), abs(1 - alphas(i) * p.lam_max));
    ratio = h.err(2:end) ./ h.err(1:end-1);
    rho_obs = tail_median(ratio, 0.05);

    it = -ones(1, 4);
    for j = 1:4
        it(j) = iter_to(h.err, tols(j));
    end
    it_str = cell(1, 4);
    for j = 1:4
        if it(j) < 0
            it_str{j} = '未达到';
        else
            it_str{j} = sprintf('%d', it(j));
        end
    end

    if h.converged
        conv_str = '是';
    else
        conv_str = '否';
    end

    rows(i, :) = [{names{i}}, {sci(alphas(i))}, {sprintf('%.8f', rho_th)}, ...
        {sprintf('%.6f', rho_obs)}, it_str, {sprintf('%d', h.iters)}, ...
        {conv_str}, {sci(h.f(end))}, {sci(h.err(end))}, ...
        {sprintf('%.2f s', h.time(end))}];

    P.(sprintf('c%d', i)) = struct('label', names{i}, 'alpha', alphas(i), ...
        'rho_theory', rho_th, 'rho_observed', rho_obs, 'iters', h.iters, ...
        'converged', h.converged, 'final_f', h.f(end), 'final_err', h.err(end), ...
        'cpu_time', h.time(end), 'iters_to_1e_4', it(1), 'iters_to_1e_6', it(2), ...
        'iters_to_1e_8', it(3), 'iters_to_1e_10', it(4));
end

print_table({'alpha', 'alpha 数值', 'rho 理论', 'rho 实测', '步数→1e-4', '步数→1e-6', ...
             '步数→1e-8', '步数→1e-10', '实际步数', '达到容差', 'f(x_K)', ...
             '||x_K-x*||', 'CPU'}, rows, ...
            sprintf('表 2  固定步长梯度下降（n=%d, seed=%d, 相对容差 %.0e）', N, SEED, RTOL));

P.theory = struct('L', L, 'alpha_opt', p.alpha_opt, 'rho_opt', p.rho_opt, ...
    'lambda_min', p.lam_min, 'lambda_max', p.lam_max, ...
    'upper_bound_2_over_lambda_max', 2 / p.lam_max);

%% 图 2
fig = figure('position', [0 0 1320 880]);

% (a) f(x_k)
subplot(2, 2, 1);
hold on;
hc = zeros(1, 4);
for i = 1:4
    h = hist{i};
    k = (0:h.iters)';
    hc(i) = plot_subsampled(k, h.f, cols{i}, 1.5);
end
set(gca, 'yscale', 'log', 'ytick', 10.^(-15:3:3)); grid on;   % 刻度必须递增
xlabel('迭代步数 k'); ylabel('f(x_k)');
title('(a) 目标函数值（几何衰减，故为直线）');
legend(hc, {sprintf('\\alpha = L/3 = %.3e', alphas(1)), ...
        sprintf('\\alpha = L/2 = %.3e', alphas(2)), ...
        sprintf('\\alpha = 3L/4 = %.3e', alphas(3)), ...
        sprintf('\\alpha = L = %.3e', alphas(4))}, 'location', 'southwest');

% (b) ||x_k - x*|| 与实测参考斜率
subplot(2, 2, 2);
hold on;
hc = zeros(1, 4);
for i = 1:4
    h = hist{i};
    k = (0:h.iters)';
    hc(i) = plot_subsampled(k, h.err, cols{i}, 1.5);
    ratio = h.err(2:end) ./ h.err(1:end-1);
    rho = tail_median(ratio, 0.05);
    if rho > 0 && rho < 1
        kk = linspace(0.35 * h.iters, h.iters, 40);
        semilogy(kk, h.err(1) * rho .^ kk, ':', 'color', cols{i}, 'linewidth', 1.1);
    end
end
set(gca, 'yscale', 'log', 'ytick', 10.^(-12:3:0)); grid on;   % 刻度必须递增
xlabel('迭代步数 k'); ylabel('||x_k - x^*||_2');
title('(b) 误差范数（点线为实测 \rho^k 参考斜率）');
legend(hc, {'L/3', 'L/2', '3L/4', 'L'}, 'location', 'southwest');

% (c) 理论步数曲线与实测
subplot(2, 2, 3);
hold on;
aa = linspace(0.02 * L, (1 - 1e-7) * L, 1500);
rho_a = max(abs(1 - aa * p.lam_min), abs(1 - aa * p.lam_max));
n_pred = log(RTOL) ./ log(rho_a);
hth = plot(aa / L, n_pred, '-', 'color', [0.35 0.35 0.35], 'linewidth', 1.8);
hstar = semilogy(p.alpha_opt / L, log(RTOL) / log(p.rho_opt), '*', ...
                'markersize', 13, 'color', C.CG);
hp = []; hx = [];
for i = 1:4
    rho = max(abs(1 - alphas(i) * p.lam_min), abs(1 - alphas(i) * p.lam_max));
    if rho < 1
        hp(end + 1) = semilogy(alphas(i) / L, log(RTOL) / log(rho), 'o', ...
            'markersize', 7, 'color', cols{i}, 'markerfacecolor', cols{i});
        hx(end + 1) = semilogy(alphas(i) / L, hist{i}.iters, 'x', ...
            'markersize', 9, 'color', cols{i}, 'linewidth', 1.8);
    else
        text(alphas(i) / L * 0.97, 1.2e6, '\alpha = L：不收敛', ...
             'horizontalalignment', 'right', 'fontsize', 10, 'color', cols{i});
    end
end
set(gca, 'yscale', 'log'); grid on;
ylim([1e5, 3e7]);
set(gca, 'ytick', 10.^(5:7));
xlabel('\alpha / L'); ylabel('达到 ||x_k - x^*|| <= 1e-10 ||x_0 - x^*|| 的步数');
title('(c) 理论步数与实测值（x 为实测）');
legend([hth, hstar, hp(1), hx(1)], {'理论步数 \ln(\epsilon)/\ln\rho(\alpha)', ...
        '\alpha^*（理论最优）', '理论值（o）与实测值（x）'}, ...
       'location', 'northeast', 'fontsize', 9);

% (d) alpha = L 的停滞水平
subplot(2, 2, 4);
hL = hist{4};
% eig 的返回顺序是 [特征向量, 特征值对角阵]，别写反
[V_eig, D_eig] = eig(p.Q);
[~, imax] = max(diag(D_eig));
c_max = V_eig(:, imax)' * (x0 - p.x_star);
floor_f = 0.5 * p.lam_max * c_max^2;
hold on;
plot_subsampled((0:hL.iters)', max(hL.f, realmin), cols{4}, 1.4);
semilogy([0, hL.iters], [floor_f, floor_f], '--', 'color', [0.4 0.4 0.4], 'linewidth', 1.3);
set(gca, 'yscale', 'log'); grid on;
xlabel('迭代步数 k'); ylabel('f(x_k)');
title('(d) \alpha = L：收敛到停滞水平后不再下降');
legend({'\alpha = L 时的 f(x_k)', sprintf('理论停滞水平 %.4f', floor_f)}, ...
       'location', 'northeast');

P.L_stall = struct('stall_floor_f', floor_f, 'c_max', c_max);

save_fig(fig, 'fig2_stepsizes');
fprintf('\n图 2 已保存：figures/fig2_stepsizes.png\n');

save_json('q2_q3_stepsizes', P);
fprintf('已写出 results/q2_q3_stepsizes.json\n');
