function v = p_f(p, x)
%P_F 目标函数值 f(x) = 1/2 ||A x - b||^2
r = p.A * x - p.b;
v = 0.5 * (r' * r);
end
