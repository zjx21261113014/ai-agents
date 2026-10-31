function s = note_of(ic, j, D)
%NOTE_OF 图 7 每个面板下方的一句结论
%   数字全部取自本次运行结果，不写死。
%   ic = 1 / 2 对应 n = 50 / 1000；j = 1 / 2 / 3 对应 (a)(b)(c) 三个面板。
%   注意两点：
%     1) sprintf 会把反斜杠开头的字符当转义，TeX 命令必须写双反斜杠；
%     2) 面板不宽，句子要短，否则右边会被裁掉。

hh = D(ic).h;
nd0 = D(ic).nd0;
mt = D(ic).meta;

if ic == 1 && j == 1
    s = sprintf('CG %d 步把 ln f 压到 %.0f', ...
                hh{1}.iters, log(max(hh{1}.f(end), 1e-300)));
elseif ic == 1 && j == 2
    s = sprintf('SD 的 |ln \\rho| 是 GD 的 %.1f 倍', ...
                (1 - mt.rho_sd_theory) / (1 - mt.rho_gd_theory));
elseif ic == 1 && j == 3
    s = sprintf('CG %.3f s，SD %.2f s（%.0f 倍）', ...
                hh{1}.time(end), hh{2}.time(end), ...
                hh{2}.time(end) / hh{1}.time(end));
elseif ic == 2 && j == 1
    s = sprintf('\\kappa = %.1e，%d 步后 ln f 几乎没动', ...
                mt.kappa, hh{2}.iters);
elseif ic == 2 && j == 2
    if hh{4}.converged
        s = sprintf('Nesterov %d 步达 1e-10，GD 仍不动', hh{4}.iters);
    else
        s = sprintf('Nesterov 只降 %.1f 个量级', ...
                    log(nd0) - log(max(hh{4}.err(end) * nd0, 1e-300)));
    end
else
    s = sprintf('CG %.2f s，SD %.0f s、GD %.0f s', ...
                hh{1}.time(end), hh{2}.time(end), hh{3}.time(end));
end
end
