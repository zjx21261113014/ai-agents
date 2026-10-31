function out = save_fig(fig, name, dpi)
%SAVE_FIG 把图存成 figures/<name>.png
%   Octave 用 -dpngcairo 设备（抗锯齿、中文正常），MATLAB 用 -dpng。

if nargin < 3, dpi = 170; end
here = fileparts(mfilename('fullpath'));
dir_out = fullfile(here, 'figures');
if ~exist(dir_out, 'dir'), mkdir(dir_out); end
out = fullfile(dir_out, [name '.png']);

if exist('OCTAVE_VERSION', 'builtin')
    print(fig, out, '-dpngcairo', sprintf('-r%d', dpi));
else
    print(fig, out, '-dpng', sprintf('-r%d', dpi));
end
end
