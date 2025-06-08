function x = lm_tCG_solver(Afun, b, M, x0, tol, maxit)
    x = M.zerovec(x0);
    r = b-Afun(x);
    p = r;
    rsold = M.inner(x0, r, r);
    for i = 1:maxit
        Ap = Afun(p);
        alpha = rsold / M.inner(x0, p, Ap);
        
        x = M.lincomb(x0, 1, x, alpha, p);
        r = M.lincomb(x0, 1, r, -alpha, Ap);
        rsnew = M.inner(x0, r, r);
        if sqrt(rsnew) < tol %do not delete --> see boumal's
            break; 
        end
        p = M.lincomb(x0,1,r,rsnew/rsold, p);
        p = M.tangent(x, p);
        rsold = rsnew;
    end
end
