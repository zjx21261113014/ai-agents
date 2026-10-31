function g = p_grad(p, x)
%P_GRAD 梯度 grad f(x) = Q x - c = A'(A x - b)
g = p.Q * x - p.c;
end
