function r = row_of_bb(rec, gd_iters, TOLS)
%ROW_OF_BB 把 describe_bb 的记录整理成表 6 的一行（14 列）

r = cell(1, 14);
r{1} = rec.name;

if ~isnan(rec.rho_fit) & ~isinf(rec.rho_fit)
    r{2} = sprintf('%.6f', rec.rho_fit);
else
    r{2} = '—';
end
if ~isnan(rec.semilog_r2) & ~isinf(rec.semilog_r2)
    r{3} = sprintf('%.6f', rec.semilog_r2);
else
    r{3} = '—';
end
if ~isnan(rec.rho_tail_median) & ~isinf(rec.rho_tail_median)
    r{4} = sprintf('%.6f', rec.rho_tail_median);
else
    r{4} = '—';
end
if ~isnan(rec.rho_f_tail_median) & ~isinf(rec.rho_f_tail_median)
    r{5} = sprintf('%.6f', rec.rho_f_tail_median);
else
    r{5} = '—';
end
if ~isnan(rec.frac_rho_f_gt1) & ~isinf(rec.frac_rho_f_gt1)
    r{6} = sprintf('%.0f%%', 100 * rec.frac_rho_f_gt1);
else
    r{6} = '—';
end
if ~isnan(rec.geomean_rate) & ~isinf(rec.geomean_rate)
    r{7} = sprintf('%.5f', rec.geomean_rate);
else
    r{7} = '—';
end
for i = 1:numel(TOLS)
    if rec.iters_to(i) < 0
        r{7 + i} = '未达到';
    else
        r{7 + i} = sprintf('%d', rec.iters_to(i));
    end
end
if rec.converged
    r{12} = sprintf('%d', rec.iters);
    r{13} = sprintf('%.1fx', gd_iters / max(rec.iters, 1));
else
    r{12} = sprintf('>%d', rec.iters);
    r{13} = '—';
end
r{14} = [sci(rec.cpu_time, 2) ' s'];
end
