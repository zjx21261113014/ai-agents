function save_json(name, S)
%SAVE_JSON 把结果写成 results/<name>.json，供报告逐项核对
here = fileparts(mfilename('fullpath'));
dir_out = fullfile(here, 'results');
if ~exist(dir_out, 'dir'), mkdir(dir_out); end
out = fullfile(dir_out, [name '.json']);

txt = jsonencode(S);
% Octave 的 jsonencode 会把非有限数写成 NaN/Inf，换成 null 保证 JSON 合法
txt = strrep(txt, ':NaN', ':null');
txt = strrep(txt, ':Inf', ':null');
txt = strrep(txt, ':-Inf', ':null');

fid = fopen(out, 'w');
if fid < 0
    error('无法写入 %s', out);
end
fprintf(fid, '%s', txt);
fclose(fid);
end
