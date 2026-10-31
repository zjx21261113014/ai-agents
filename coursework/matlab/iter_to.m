function kk = iter_to(err, tol)
%ITER_TO 首次满足 err(k) <= tol 的步数 k（k 从 0 起算）；未达到则返回 -1。
idx = find(err <= tol, 1, 'first');
if isempty(idx)
    kk = -1;
else
    kk = idx - 1;   % err(1) 对应 k = 0
end
end
