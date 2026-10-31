function r = row_of(rec, gd_iters)
%ROW_OF 把 describe_run 的记录整理成表 4 / 表 5 的一行
TOLS = [1e-4, 1e-6, 1e-8, 1e-10];

r = cell(1, 12);
r{1} = rec.name;
r{2} = rec.beta_label;
if ~isnan(rec.rho_theory) & ~isinf(rec.rho_theory)
    r{3} = sprintf('%.8f', rec.rho_theory);
else
    r{3} = '—';
end
if ~isnan(rec.rho_fit) & ~isinf(rec.rho_fit)
    r{4} = sprintf('%.6f', rec.rho_fit);
else
    r{4} = '—';
end
if ~isnan(rec.semilog_r2) & ~isinf(rec.semilog_r2)
    r{5} = sprintf('%.6f', rec.semilog_r2);
else
    r{5} = '—';
end
for i = 1:numel(TOLS)
    if rec.iters_to(i) < 0
        r{5 + i} = '未达到';
    else
        r{5 + i} = sprintf('%d', rec.iters_to(i));
    end
end
if rec.converged
    r{10} = sprintf('%d', rec.iters);
    r{11} = sprintf('%.1fx', gd_iters / max(rec.iters, 1));
else
    r{10} = sprintf('>%d', rec.iters);
    r{11} = '—';
end
r{12} = [sci(rec.cpu_time, 2) ' s'];
end
