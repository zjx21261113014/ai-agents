%% 第 1 题：问题本身的性质
% 对应作业第 1 节第 1 小问。要确认三件事：
%   (1) Q = A'A 对称正定，f 是严格凸二次函数，最优解唯一；
%   (2) 最优解可以用正规方程 x* = Q\c 直接求出，并与 lstsq、inv 两种算法对照；
%   (3) 由 f、grad f、Hessian 三者的数值行为验证解析式没有写错。
% 同时给出 n = 50 与 n = 1000 两个规模下 Q 的谱分布，说明条件数的量级。

clear; clc;
C = set_style();

N = 50;         % 报告正文与后续各题统一用 n = 50
SEED = 0;
N_BIG = 1000;   % 只用于比较谱分布

p = mk_problem(N, SEED);
p_big = mk_problem(N_BIG, SEED);
x0 = zeros(N, 1);

fprintf('n = %d, seed = %d\n', N, SEED);
fprintf('lambda_min = %.6e, lambda_max = %.6e\n', p.lam_min, p.lam_max);
fprintf('kappa(Q) = %.6e, sqrt(kappa) = %.4f\n', p.kappa, sqrt(p.kappa));
fprintf('||x*||_2 = %.6e, f* = %.6e\n', norm(p.x_star), p.f_star);
fprintf('L = 2/lambda_max = %.6e, alpha* = 2/(lambda_min+lambda_max) = %.6e\n', ...
        p.L, p.alpha_opt);
fprintf('rho(alpha*) = (kappa-1)/(kappa+1) = %.8f\n\n', p.rho_opt);

%% 1. 正定性：Cholesky 是否能做，特征值是否全正
[~, chol_flag] = chol(p.Q);
sym_err = norm(p.Q - p.Q', 'fro') / norm(p.Q, 'fro');
n_pos = sum(p.eigvals > 0);

fprintf('Q 的对称性相对误差 ||Q-Q''||_F/||Q||_F = %.3e\n', sym_err);
fprintf('Cholesky 分解成功：%d（0 表示成功）\n', chol_flag);
fprintf('Q 的正特征值个数：%d / %d\n', n_pos, N);

%% 2. 最优解的三种算法对照
xs_backslash = p.Q \ p.c;
xs_inv = inv(p.Q) * p.c;
xs_lstsq = p.A \ p.b;
rel_bs = norm(xs_backslash - p.x_star) / norm(p.x_star);
rel_inv = norm(xs_inv - p.x_star) / norm(p.x_star);
rel_ls = norm(xs_lstsq - p.x_star) / norm(p.x_star);

g_star = p.Q * p.x_star - p.c;
fprintf('\nx* 的三种算法与 Q\\c 的相对差异：\n');
fprintf('  lstsq (A\\b)   : %.3e\n', rel_ls);
fprintf('  inv(Q)*c      : %.3e\n', rel_inv);
fprintf('  两种直接法对比: %.3e\n', rel_bs);
fprintf('||grad f(x*)||_2 = %.3e （应为 0）\n', norm(g_star));

%% 3. 解析式的数值校验
d = randn(N, 1);
d = d / norm(d);
xx = x0;
t = 1e-2;

% (a) 三点恒等式：f(x+td) - 2f(x) + f(x-td) = t^2 d'Qd
lhs = p_f(p, xx + t * d) - 2 * p_f(p, xx) + p_f(p, xx - t * d);
rhs = t^2 * (d' * (p.Q * d));
fprintf('\n三点恒等式：左端 %.12e，右端 %.12e，相对误差 %.3e\n', ...
        lhs, rhs, abs(lhs - rhs) / abs(rhs));

% (b) 中心差分 Hessian 与 Q 比较
h_step = 1e-5;
H_fd = zeros(N, N);
for j = 1:N
    ej = zeros(N, 1); ej(j) = 1;
    H_fd(:, j) = (p_grad(p, xx + h_step * ej) - p_grad(p, xx - h_step * ej)) / (2 * h_step);
end
fprintf('中心差分 Hessian 与 Q 的相对误差：%.3e\n', norm(H_fd - p.Q, 'fro') / norm(p.Q, 'fro'));

% (c) 凸性：中点处函数值应严格小于两端平均
y1 = randn(N, 1);
y2 = randn(N, 1);
mid = 0.5 * (y1 + y2);
gap = 0.5 * (p_f(p, y1) + p_f(p, y2)) - p_f(p, mid);
gap_theory = 0.125 * (y1 - y2)' * p.Q * (y1 - y2);
fprintf('凸性中点间隙：实测 %.6e，理论 %.6e，相对误差 %.3e\n', ...
        gap, gap_theory, abs(gap - gap_theory) / gap_theory);

%% 4. 两个规模下的谱
fprintf('\n规模     lambda_min      lambda_max      kappa\n');
fprintf('n=%-6d %.4e      %.4e      %.4e\n', N, p.lam_min, p.lam_max, p.kappa);
fprintf('n=%-6d %.4e      %.4e      %.4e\n', N_BIG, p_big.lam_min, p_big.lam_max, p_big.kappa);

%% 图 1
fig = figure('position', [0 0 1280 580]);

subplot(1, 2, 1);
idx = (1:N)';
semilogy(idx / N, p.eigvals / p.lam_max, '-', 'color', C.GD, 'linewidth', 1.6);
hold on;
idx2 = (1:N_BIG)';
semilogy(idx2 / N_BIG, p_big.eigvals / p_big.lam_max, '-', 'color', C.SD, 'linewidth', 1.6);
grid on;
xlabel('特征值序号 j/n');
ylabel('\lambda_j / \lambda_{max}');
title('(a) Q 的谱：n=50 与 n=1000');
legend({'n = 50', 'n = 1000'}, 'location', 'southwest');
text(0.06, 0.35, sprintf('n=50: \\kappa = %.2e', p.kappa), 'fontsize', 10);
text(0.06, 0.15, sprintf('n=1000: \\kappa = %.2e', p_big.kappa), 'fontsize', 10);

subplot(1, 2, 2);
d2 = randn(N, 1); d2 = d2 / norm(d2);
ts = linspace(-1.5, 1.5, 61);
fv = zeros(size(ts));
for i = 1:numel(ts)
    fv(i) = p_f(p, x0 + ts(i) * d2);
end
% 二次模型：f(x0) + t g'd + t^2/2 d'Qd
g0 = p_grad(p, x0);
model = p_f(p, x0) + ts * (g0' * d2) + 0.5 * ts.^2 * (d2' * (p.Q * d2));
plot(ts, fv, 'o', 'color', C.GD, 'markersize', 4);
hold on;
plot(ts, model, '-', 'color', C.Nesterov, 'linewidth', 1.6);
grid on;
xlabel('t');
ylabel('f(x_0 + t d)');
title('(b) 沿方向的二次性检验');
legend({'f(x_0+td) 实测', '二次模型'}, 'location', 'north');
dev = max(abs(fv - model)) / abs(mean(fv));
text(-1.4, 0.30 * max(fv - min(fv)) + min(fv), ...
     sprintf('最大偏差 %.2e', dev), 'fontsize', 9, 'backgroundcolor', 'w');

save_fig(fig, 'fig1_problem');
fprintf('\n图 1 已保存：figures/fig1_problem.png\n');

%% 5. 条件数、残差与随机 Rayleigh 商探测
sa = svd(p.A);
kappa_A = max(sa) / min(sa);
nb = norm(p.b);
res_ax = norm(p.A * p.x_star - p.b);
grad_rel = norm(g_star) / norm(p.c);

% 用随机单位向量做 500 次 Rayleigh 商探测，看能否碰到谱的两端
n_probe = 500;
rq = zeros(n_probe, 1);
for i = 1:n_probe
    v = randn(N, 1); v = v / norm(v);
    rq(i) = v' * (p.Q * v);
end
fprintf('\nkappa(A) = %.6e（= sqrt(kappa(Q))）\n', kappa_A);
fprintf('||b||_2 = %.6f, ||Ax*-b||_2 = %.3e\n', nb, res_ax);
fprintf('||grad f(x*)||_2 / ||c||_2 = %.3e\n', grad_rel);
fprintf('随机 Rayleigh 商探测（%d 次）：[%.4e, %.4e]\n', ...
        n_probe, min(rq), max(rq));

%% 结果落盘
S = struct();
S.n = N;
S.seed = SEED;
S.lam_min = p.lam_min;
S.lam_max = p.lam_max;
S.kappa = p.kappa;
S.x_star_norm = norm(p.x_star);
S.f_star = p.f_star;
S.L = p.L;
S.alpha_opt = p.alpha_opt;
S.rho_opt = p.rho_opt;
S.symmetry_rel_err = sym_err;
S.chol_flag = chol_flag;
S.n_positive_eig = n_pos;
S.grad_norm_at_x_star = norm(g_star);
S.rel_diff_lstsq = rel_ls;
S.rel_diff_inv = rel_inv;
S.rel_diff_backslash = rel_bs;
S.three_point_rel_err = abs(lhs - rhs) / abs(rhs);
S.hessian_fd_rel_err = norm(H_fd - p.Q, 'fro') / norm(p.Q, 'fro');
S.convexity_gap = gap;
S.convexity_gap_theory = gap_theory;
S.kappa_A = kappa_A;
S.norm_b = nb;
S.residual_ax_minus_b = res_ax;
S.grad_rel = grad_rel;
S.rayleigh_probe_min = min(rq);
S.rayleigh_probe_max = max(rq);
S.rayleigh_n_probe = n_probe;
S.n_big = N_BIG;
S.kappa_big = p_big.kappa;
S.lam_min_big = p_big.lam_min;
S.lam_max_big = p_big.lam_max;
save_json('q1_problem', S);
fprintf('已写出 results/q1_problem.json\n');
