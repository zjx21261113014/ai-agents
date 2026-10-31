%% 第 7 题：n = 2 时把四种方法的迭代点画在 f 的等值线上
%   题目要求：取 n = 2，在同样初值下，把 BB 方法（这里用 BB1）、梯度下降法、
%   Heavy Ball 方法与 Nesterov 方法的 {x_k} 分别描述在 f(x) 的等值线上。
%
%   统一设定：四种方法用同一个初值 x0；步长取各自的标准值
%     GD / HB : alpha = alpha* = 2/(lam_min+lam_max)（二次问题的最优固定步长）
%     Nesterov: alpha = 1/lam_max（梯度 Lipschitz 常数的倒数）
%     BB1     : alpha_0 = 1/lam_max 起步
%   HB 的 beta 用 best_beta 在 alpha = alpha* 下数值搜索；Nesterov 用经典 beta。
%   注：alpha = alpha* 时 Nesterov 对任何常数 beta 都不稳定（前瞻点过冲），
%   所以只能取 alpha = 1/lam_max，这也是 Nesterov 基本格式的标准前提。

clear; clc;
C = set_style();

N = 2;
SEED = 0;   % seed = 0 时 kappa = 69.1（sqrt(kappa) = 8.3），拉长的"香蕉谷"足够明显
X0 = [-1; 2];
RTOL = 1e-10;
N_MAX = 20000;
N_SHOW_GD = 40;              % 梯度下降的锯齿太密，图上只画前 40 步

p = mk_problem(N, SEED);
lmin = p.lam_min;
lmax = p.lam_max;
kappa = p.kappa;
alpha = p.alpha_opt;         % = 2/(lam_min+lam_max)
alpha_nes = 1 / lmax;

bb = best_beta(p.eigvals, alpha, 'heavy_ball');
beta_hb = bb.beta;
rho_hb = bb.rho;
beta_nes = ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^2;

rho_gd = max(abs(1 - alpha * p.eigvals));
rho_nes = mom_rho(p.eigvals, alpha_nes, beta_nes, 'nesterov');

fprintf('n = %d, seed = %d\n', N, SEED);
fprintf('lam_min = %.6e, lam_max = %.6e, kappa = %.4f, sqrt(kappa) = %.4f\n', ...
        lmin, lmax, kappa, sqrt(kappa));
fprintf('x0 = [%.1f, %.1f]'', x* = [%.6f, %.6f]'', ||x0-x*|| = %.4f\n', ...
        X0(1), X0(2), p.x_star(1), p.x_star(2), norm(X0 - p.x_star));
fprintf('GD 与 HB：alpha = alpha* = %.6e，HB 的 beta = %.6f（rho = %.6f）\n', alpha, beta_hb, rho_hb);
fprintf('Nesterov ：alpha = 1/lam_max = %.6e，beta = %.6f，rho = %.6f\n\n', ...
        alpha_nes, beta_nes, rho_nes);

%% 四条轨迹
keys = {'bb1', 'gd', 'hb', 'nes'};
nm = {'BB1', '梯度下降（alpha = alpha*）', 'Heavy Ball（alpha = alpha*）', ...
      'Nesterov（alpha = 1/lam_max）'};
xs = cell(1, 4);
cpu = zeros(1, 4);
[xs{1}, cpu(1)] = traj2(p, X0, 'bb1', [], [], N_MAX, RTOL);
[xs{2}, cpu(2)] = traj2(p, X0, 'gd', alpha, [], N_MAX, RTOL);
[xs{3}, cpu(3)] = traj2(p, X0, 'hb', alpha, beta_hb, N_MAX, RTOL);
[xs{4}, cpu(4)] = traj2(p, X0, 'nes', alpha_nes, beta_nes, N_MAX, RTOL);

e0 = norm(X0 - p.x_star);
tol = RTOL * e0;
iters = zeros(1, 4);
final_err = zeros(1, 4);
conv = false(1, 4);
for i = 1:4
    iters(i) = size(xs{i}, 2) - 1;
    final_err(i) = norm(xs{i}(:, end) - p.x_star);
    conv(i) = final_err(i) <= tol;
end

%% 表 7
HEADER = {'方法', '步数', '理论 rho', '末步误差', 'CPU (ms)', '轨迹形状'};
shape = {'一步下到谷底，再沿谷底大步命中 x*', ...
         '沿陡方向来回锯齿，同时沿谷底缓慢漂移', ...
         '冲过谷底再折回（过冲），螺旋进入 x*', ...
         '沿谷底几乎是直线（陡方向一步清零）'};
rho_th = [NaN, rho_gd, rho_hb, rho_nes];
rows = cell(4, 6);
for i = 1:4
    rows{i, 1} = nm{i};
    rows{i, 2} = sprintf('%d', iters(i));
    if ~isnan(rho_th(i)) & ~isinf(rho_th(i))
        rows{i, 3} = sprintf('%.4f', rho_th(i));
    else
        rows{i, 3} = '—（自适应）';
    end
    rows{i, 4} = sprintf('%.2e', final_err(i));
    rows{i, 5} = sprintf('%.2f', 1e3 * cpu(i));
    rows{i, 6} = shape{i};
end
print_table(HEADER, rows, ...
    sprintf('表 7  n = %d 时四种方法（各取标准参数，同一初值 x0）的迭代步数', N));

%% 等值线网格
allx = [xs{1}, xs{2}, xs{3}, xs{4}, p.x_star, X0];
cx = 0.5 * (min(allx(1, :)) + max(allx(1, :)));
cy = 0.5 * (min(allx(2, :)) + max(allx(2, :)));
side = 1.15 * max([max(allx(1, :)) - min(allx(1, :)), ...
                   max(allx(2, :)) - min(allx(2, :)), 2.4]);
xlo = cx - 0.5 * side; xhi = cx + 0.5 * side;
ylo = cy - 0.5 * side; yhi = cy + 0.5 * side;

ng = 240;
gx = linspace(xlo, xhi, ng);
gy = linspace(ylo, yhi, ng);
[GX, GY] = meshgrid(gx, gy);

Q = p.Q;
dx1 = GX - p.x_star(1);
dx2 = GY - p.x_star(2);
energy = 0.5 * (Q(1, 1) * dx1.^2 + 2 * Q(1, 2) * dx1 .* dx2 + Q(2, 2) * dx2.^2);
Z = log10(max(energy, 1e-300));
% 等值线只取最外层往下 6 个数量级：再往下就挤在 x* 附近，看不出椭圆的拉长
zmax = max(Z(:));
zlo = zmax - 4;
Z = max(Z, zlo);
levels = linspace(zlo, zmax, 11);

% 主方向：过 x* 的两条特征方向
[V, D] = eig(Q);
v_min = V(:, 1); v_min = v_min / norm(v_min);
v_max = V(:, 2); v_max = v_max / norm(v_max);

cmap = [linspace(0.46, 0.99, 64)', linspace(0.46, 0.99, 64)', linspace(0.48, 0.99, 64)'];

colv = {C.BB1, C.GD, C.HB, C.Nesterov};
pname = {'(a) BB1', '(b) 梯度下降', '(c) Heavy Ball', '(d) Nesterov'};
ptitle = {'(a) BB1：alpha_0 = 1/lam_max', ...
          '(b) 梯度下降：alpha = alpha*，无动量', ...
          sprintf('(c) Heavy Ball：alpha = alpha*，beta = %.3f', beta_hb), ...
          sprintf('(d) Nesterov：alpha = 1/lam_max，beta = %.3f', beta_nes)};

%% 图 6
fig = figure('position', [0 0 1240 1180]);

for i = 1:4
    subplot(2, 2, i);
    hold on;

    contourf(GX, GY, Z, levels);
    colormap(cmap);
    contour(GX, GY, Z, levels, 'linewidth', 0.5);

    % 两条主轴
    tt = [-2, 2] * side;
    plot(p.x_star(1) + tt * v_min(1), p.x_star(2) + tt * v_min(2), '--', ...
         'color', C.ref, 'linewidth', 0.65);
    plot(p.x_star(1) + tt * v_max(1), p.x_star(2) + tt * v_max(2), '--', ...
         'color', C.ref, 'linewidth', 0.65);

    % 轨迹
    X = xs{i};
    if i == 2 && size(X, 2) - 1 > N_SHOW_GD
        X = X(:, 1:N_SHOW_GD + 1);
    end
    plot(X(1, :), X(2, :), '-', 'color', colv{i}, 'linewidth', 1.4);
    plot(X(1, :), X(2, :), '.', 'color', colv{i}, 'markersize', 6);

    % 初值与最优点
    plot(X0(1), X0(2), 's', 'markersize', 8, 'markerfacecolor', 'w', ...
         'markeredgecolor', 'k', 'linewidth', 1.3);
    plot(p.x_star(1), p.x_star(2), 'p', 'markersize', 15, ...
         'markerfacecolor', C.Nesterov, 'markeredgecolor', 'w');
    text(X0(1) - 0.10 * side, X0(2) - 0.075 * side, 'x_0', 'fontsize', 12);
    text(p.x_star(1) + 0.05 * side, p.x_star(2) + 0.03 * side, 'x*', ...
         'fontsize', 12, 'color', C.Nesterov);

    % 主轴名称
    xu = p.x_star + 0.40 * side * v_max;
    text(min(max(xu(1), xlo + 0.07 * side), xhi - 0.07 * side), ...
         min(max(xu(2), ylo + 0.07 * side), yhi - 0.07 * side), 'q_{max}', ...
         'fontsize', 11, 'color', [0.3 0.3 0.3], 'horizontalalignment', 'center');
    xv = p.x_star - 0.40 * side * v_min;
    text(min(max(xv(1), xlo + 0.07 * side), xhi - 0.07 * side), ...
         min(max(xv(2), ylo + 0.07 * side), yhi - 0.07 * side), 'q_{min}', ...
         'fontsize', 11, 'color', [0.3 0.3 0.3], 'horizontalalignment', 'center');

    % 图内只留一行步数；四行推导写进正文，5 cm 宽的面板里印不下
    if i == 2 && iters(i) > N_SHOW_GD
        lab = sprintf('%d 步到 1e-10（只画前 %d 步）', iters(i), N_SHOW_GD);
    else
        lab = sprintf('%d 步到 1e-10', iters(i));
    end
    text(xlo + 0.03 * side, ylo + 0.09 * side, lab, 'fontsize', 10);

    xlim([xlo, xhi]); ylim([ylo, yhi]);
    daspect([1 1 1]);
    grid on;
    xlabel('x^(1)'); ylabel('x^(2)');
    title(ptitle{i});
end

save_fig(fig, 'fig6_contour');
fprintf('图 6 已保存：figures/fig6_contour.png\n\n');

%% 判读要点
fprintf('判读要点：\n');
fprintf('  1. 等高线是长轴比 sqrt(kappa) = %.1f 的椭圆族：沿 q_max（陡方向）f 变得极快，\n', sqrt(kappa));
fprintf('     沿 q_min（谷底方向）几乎不变 —— 这就是条件数造成的困难。\n');
fprintf('  2. GD(alpha=alpha*) 的轨迹沿陡方向来回锯齿（1-alpha*lam_max = %.4f < 0，交替变号），\n', ...
        1 - alpha * lmax);
fprintf('     同时沿谷底缓慢漂移；锯齿的衰减率正是 (kappa-1)/(kappa+1) = %.4f。\n', ...
        (kappa - 1) / (kappa + 1));
fprintf('  3. Heavy Ball 用动量累加位移，第一步就冲到谷底另一侧（过冲），之后螺旋进入 x*；\n');
fprintf('     谱半径从 %.4f 降到 %.4f，步数相应从 %d 降到 %d。\n', ...
        rho_gd, rho_hb, iters(2), iters(3));
fprintf('  4. Nesterov 取 alpha = 1/lam_max：1-alpha*lam_max = 0，陡方向分量一步清零，\n');
fprintf('     所以看不到锯齿，点几乎沿谷底推进；慢方向仍需 %d 步，比 GD 的 %d 步少。\n', ...
        iters(4), iters(2));
fprintf('  5. BB1 用 s''s/s''y 自适应放大步长，n = 2 时几乎一眼看穿：%d 步就到 1e-10，\n', iters(1));
fprintf('     轨迹是一条折线，只在谷底拐一下。\n\n');

%% 结果
P = struct();
P.n = N; P.seed = SEED;
P.x0 = X0(:)'; P.x_star = p.x_star(:)';
P.lambda_min = lmin; P.lambda_max = lmax;
P.kappa = kappa; P.sqrt_kappa = sqrt(kappa);
P.alpha_gd_hb = alpha;
P.beta_heavy_ball = beta_hb; P.rho_heavy_ball = rho_hb;
P.alpha_nesterov = alpha_nes; P.beta_nesterov = beta_nes;
P.rtol = RTOL;
P.n_show_gd = N_SHOW_GD;
for i = 1:4
    r = struct();
    r.name = nm{i};
    r.iters = iters(i);
    r.final_err = final_err(i);
    r.converged = conv(i);
    r.rho_theory = rho_th(i);
    P.methods.(sprintf('m%d', i)) = r;
end

save_json('q7_contour', P);
fprintf('已写出 results/q7_contour.json\n');
