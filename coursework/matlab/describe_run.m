function rec = describe_run(name, beta_label, rho_theory, h, TOLS)
%DESCRIBE_RUN 把一个 History 归纳成一行表格 + 一份 JSON 记录
e = h.err(:);
k = (0:numel(e) - 1)';
fit = semilog_fit(k, e, 0.30, 0.95);
rk = e(2:end) ./ max(e(1:end-1), 1e-300);

rec = struct();
rec.name = name;
rec.beta_label = beta_label;
rec.rho_theory = rho_theory;
rec.rho_fit = fit.rate;
rec.semilog_r2 = fit.r2;
rec.rho_tail_median = tail_median(rk, 0.1);
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
