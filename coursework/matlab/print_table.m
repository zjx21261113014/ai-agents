function print_table(header, rows, title)
%PRINT_TABLE 在命令行打印一张纯文本表（报告里的表格即由此抄录）

if nargin >= 3 && ~isempty(title)
    fprintf('\n%s\n', title);
end
ncol = numel(header);
w = zeros(1, ncol);
for j = 1:ncol
    w(j) = disp_width(header{j});
    for i = 1:size(rows, 1)
        w(j) = max(w(j), disp_width(rows{i, j}));
    end
end

line = repmat('-', 1, sum(w) + 3 * ncol);
fprintf('%s\n', line);
fprintf('%s\n', fmt_row(header, w));
fprintf('%s\n', line);
for i = 1:size(rows, 1)
    fprintf('%s\n', fmt_row(rows(i, :), w));
end
fprintf('%s\n', line);

    function s = fmt_row(cells, w)
        parts = cell(1, numel(cells));
        for jj = 1:numel(cells)
            c = cells{jj};
            if isnumeric(c), c = num2str(c); end
            pad = w(jj) - disp_width(c);
            parts{jj} = [c repmat(' ', 1, max(pad, 0))];
        end
        s = strjoin(parts, ' | ');
    end

    function n = disp_width(s)
        % 中文按两格宽计算，等宽字体下才能对齐。
        % 注意 UTF-8 下 char 是按字节存的，不能直接数 numel，
        % 要按首字节判断这是几个字节的字符，续字节不再计数。
        if ~ischar(s), s = num2str(s); end
        b = double(s);
        n = 0;
        i = 1;
        while i <= numel(b)
            c = b(i);
            if c < 128
                n = n + 1;
                i = i + 1;
            elseif c < 224
                n = n + 1;      % 两字节字符（拉丁扩展、希腊字母等）
                i = i + 2;
            elseif c < 240
                n = n + 2;      % 三字节字符（汉字、全角标点）
                i = i + 3;
            else
                n = n + 2;      % 四字节字符（极少见）
                i = i + 4;
            end
        end
    end
end
