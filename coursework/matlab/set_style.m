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
end

% 无窗口时 figure 一旦被"显示"，gnuplot 会把整幅图以字符画形式吐到终端，
% 把正常的数值输出淹没；MATLAB 下则会把七个窗口一个个弹出来打断批处理。
% 默认不可见即可避免。要看得见就手动 figure(n); set(n, 'visible', 'on')。
set(0, 'defaultfigurevisible', 'off');

% 图上文字是中文。MATLAB 的默认字体在部分 Windows 机器上不含 CJK，会渲染成方框；
% 按优先级挑一个系统里真实存在的字体，都没有就退回默认（不报错）。
cand = {'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', ...
        'Source Han Sans SC', 'WenQuanYi Micro Hei', 'STHeiti', 'PingFang SC'};
try
    fonts = listfonts;
    if ischar(fonts), fonts = {fonts}; end
    for ii = 1:numel(cand)
        if any(strcmpi(fonts, cand{ii}))
            set(0, 'defaultaxesfontname', cand{ii});
            set(0, 'defaulttextfontname', cand{ii});
            break;
        end
    end
catch
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
