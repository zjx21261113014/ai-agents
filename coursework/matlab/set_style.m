function C = set_style()
%SET_STYLE 统一的绘图与命令行输出风格
%   在 MATLAB 与 GNU Octave 下都能用。Octave 无图形界面时自动退回 gnuplot。

if exist('OCTAVE_VERSION', 'builtin')
    try
        graphics_toolkit('qt');
    catch
        try
            graphics_toolkit('gnuplot');
        catch
        end
    end
    % 无窗口时 figure 一旦被"显示"，gnuplot 会把整幅图以字符画形式吐到终端，
    % 把正常的数值输出淹没。默认不可见即可避免。
    set(0, 'defaultfigurevisible', 'off');
end

set(0, 'defaultaxesfontsize', 12);
set(0, 'defaulttextfontsize', 12);
set(0, 'defaultlinelinewidth', 1.5);
set(0, 'defaultaxeslinewidth', 0.8);
set(0, 'defaultaxesbox', 'on');
set(0, 'defaultaxesgridlinestyle', ':');

% 与报告图例一致的配色
C.GD = [31 119 180] / 255;
C.SD = [255 127 14] / 255;
C.HB = [44 160 44] / 255;
C.Nesterov = [214 39 40] / 255;
C.BB1 = [148 103 189] / 255;
C.BB2 = [140 86 75] / 255;
C.CG = [23 190 207] / 255;
C.ref = [0.45 0.45 0.45];
end
