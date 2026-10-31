function s = sci(x, digits)
%SCI 把数写成 1.23e-05 这样的短形式（报告里的表格都用这个格式）
if nargin < 2, digits = 3; end
if x == 0
    s = '0';
    return;
end
e = floor(log10(abs(x)));
m = x / 10^e;
s = sprintf(['%.' num2str(digits) 'fe%+03d'], m, e);
end
