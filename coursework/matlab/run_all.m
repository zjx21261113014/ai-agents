function run_all()
%RUN_ALL  一键复现报告中的全部实验、图与 JSON 结果。
%
%   run_all
%
%   在 MATLAB 或 GNU Octave 下均可运行。依次执行 exp1..exp7，每个脚本
%   自己清空工作区、自己写 results/*.json 与 figures/*.png，所以这里用
%   evalin('base', ...) 把脚本放到基础工作区里跑：脚本开头的 clear 不
%   会影响本文件的控制变量。
%
%   某个脚本出错时打印错误并继续，最后给出汇总。

here = fileparts(mfilename('fullpath'));
if isempty(here)
    here = pwd;
end

if ~exist(fullfile(here, 'results'), 'dir')
    mkdir(fullfile(here, 'results'));
end
if ~exist(fullfile(here, 'figures'), 'dir')
    mkdir(fullfile(here, 'figures'));
end

list = {'exp1_problem', 'exp2_stepsizes', 'exp3_order', 'exp4_momentum', ...
        'exp5_bb', 'exp6_contour', 'exp7_cg'};
label = {'第 1 题 对称正定二次型与谱', ...
         '第 2、3 题 固定步长梯度下降', ...
         '第 4 题 收敛阶的数值判定', ...
         '第 5 题 Heavy Ball 与 Nesterov', ...
         '第 6 题 Barzilai-Borwein 步长', ...
         '第 7 题 n = 2 的等值线与迭代轨迹', ...
         '作业第 2 节 共轭梯度法'};

t0 = tic;
failed = {};
for i = 1:numel(list)
    fprintf('== [%d/%d] %s（%s）==\n', i, numel(list), label{i}, list{i});
    ts = tic;
    try
        evalin('base', sprintf('cd(''%s''); %s;', here, list{i}));
        fprintf('== [%d/%d] 完成，用时 %.1f s ==\n\n', i, numel(list), toc(ts));
    catch err
        failed{end + 1} = list{i}; %#ok<AGROW>
        fprintf(2, '== [%d/%d] 失败：%s ==\n', i, numel(list), err.message);
        fprintf(2, '%s\n\n', getReport(err, 'basic'));
    end
end

%% 汇总
fprintf('================ 汇总 ================\n');
fprintf('总用时 %.1f s，成功 %d 个，失败 %d 个\n', ...
        toc(t0), numel(list) - numel(failed), numel(failed));
if ~isempty(failed)
    fprintf('失败的脚本：%s\n', strjoin(failed, ', '));
end

fprintf('\n结果文件（results/）：\n');
show_files(fullfile(here, 'results'), '*.json');
fprintf('图形文件（figures/）：\n');
show_files(fullfile(here, 'figures'), '*.png');
fprintf('======================================\n');

end

% ---------------------------------------------------------------- 子函数
function show_files(d, pat)
f = dir(fullfile(d, pat));
if isempty(f)
    fprintf('  （无）\n');
    return;
end
for k = 1:numel(f)
    fprintf('  %-24s %8.1f KB\n', f(k).name, f(k).bytes / 1024);
end
end
