function p = mk_problem(n, seed)
%MK_PROBLEM 按题目要求生成 A = randn(n,n)、b = randn(n)，
%   并算出所有派生量（谱、最优解、步长相关的理论量）。
%
%   min f(x) = 1/2 * ||A x - b||_2^2
%   grad f(x) = Q x - c,  Q = A'*A,  c = A'*b,  Hessian 恒为 Q
%
%   输入
%     n    矩阵阶数
%     seed 随机数种子（默认 0）
%   输出 p 结构体，字段见下。
%
%   注意：题目给的收敛区间上端 L = 2/lambda_max，它与 Lipschitz 常数
%   lambda_max 本身不是一回事，报告正文对此做了区分。

if nargin < 2, seed = 0; end

% 固定随机流：MATLAB 与 Octave 都认 rng；老版本退回 randn('state',seed)
if exist('rng', 'builtin') || exist('rng', 'file')
    rng(seed, 'twister');
else
    randn('state', seed);
end

A = randn(n, n);
b = randn(n, 1);   % 题目写成 b=randn(n)，但在 MATLAB 里那是 n 阶矩阵，
                  % 右端项应为 n 维列向量，故明确写 randn(n,1)

p.A = A;
p.b = b;
p.n = n;
p.seed = seed;

p.Q = A' * A;
p.c = A' * b;

% 解析最优解：解正规方程 Q x = c
p.x_star = p.Q \ p.c;
r = A * p.x_star - b;
p.f_star = 0.5 * (r' * r);

% 谱信息（Q 对称，用 eig 后取实部即可）
lam = sort(real(eig(p.Q)));
p.eigvals = lam;
p.lam_min = lam(1);
p.lam_max = lam(end);
p.kappa = p.lam_max / p.lam_min;

% 步长相关的理论量
p.L = 2 / p.lam_max;                          % 收敛区间 (0, L) 的上端点
p.alpha_opt = 2 / (p.lam_min + p.lam_max);    % 最优固定步长
p.rho_opt = (p.kappa - 1) / (p.kappa + 1);
end
