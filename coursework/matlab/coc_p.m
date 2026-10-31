function [kk, pk] = coc_p(e)
%COC_P 计算型收敛阶（computational order of convergence）
%   p_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1})
%   线性收敛时 p_k -> 1；二次收敛时 p_k -> 2。
%   返回 p_{k+1} 对应的步号 kk 与数值 pk。

e = e(:);
N = numel(e);
if N < 3
    kk = zeros(0, 1);
    pk = zeros(0, 1);
    return;
end

pk = zeros(N - 2, 1);
kk = (2:N - 1)';
for k = 2:N - 1
    a = e(k) / e(k - 1);
    b = e(k + 1) / e(k);
    if a > 0 && b > 0 && a ~= 1
        pk(k - 1) = log(b) / log(a);
    else
        pk(k - 1) = NaN;
    end
end
end
