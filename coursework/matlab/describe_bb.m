function rec = describe_bb(name, h, p, TOLS)
%DESCRIBE_BB 把一个 History 归纳成表 6 的一行 + 一份 JSON 记录
%   同时统计两种口径的收缩率：2-范数的 e_k，以及 f_k - f*（恰为 Q-范数平方的一半）。
%   BB 在这两种口径下的单调性差别极大，必须分开看。

e = h.err(:);
k = (0:numel(e) - 1)';
fit = semilog_fit(k, e, 0.30, 0.95);
n_step = max(h.iters, 1);

rho_e = ratios_of(e);
energy = h.f(:) - p.f_star;
rho_f = ratios_of(energy);

e0 = energy(1);
valid = energy(1:end-1) > 1e-14 * e0;
dE = diff(energy) > 0;
n_f_increase = sum(dE(valid));

rec = struct();
rec.name = name;
rec.rho_fit = fit.rate;
rec.semilog_r2 = fit.r2;
rec.rho_tail_median = tail_median(rho_e, 0.1);
if isempty(rho_e)
    rec.frac_rho_gt1 = NaN;
else
    rec.frac_rho_gt1 = mean(rho_e > 1);
end
rec.rho_f_tail_median = tail_median(rho_f, 0.1);
if isempty(rho_f)
    rec.frac_rho_f_gt1 = NaN;
else
    rec.frac_rho_f_gt1 = mean(rho_f > 1);
end
if e(end) > 0
    rec.geomean_rate = (e(end) / e(1))^(1 / n_step);
else
    rec.geomean_rate = 0;
end
rec.energy_geomean_rate = (energy(end) / energy(1))^(1 / n_step);
rec.n_f_increase = n_f_increase;
rec.n_valid = sum(valid);
rec.iters = h.iters;
rec.converged = h.converged;
rec.final_f = h.f(end);
rec.final_err = e(end);
rec.cpu_time = h.time(end);
rec.iters_to = zeros(1, numel(TOLS));
for i = 1:numel(TOLS)
    rec.iters_to(i) = iter_to(e, TOLS(i));
end
end
