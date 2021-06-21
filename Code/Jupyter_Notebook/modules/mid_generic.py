import numpy as np
import sympy as sp
import cxroots as cx
import scipy.special as spsp


def compute_root_spectrum_mid_generic(n, m, s0, value_tau):
    global a_coeff_mid_generic
    global b_coeff_mid_generic
    s = sp.symbols('s')
    tau = sp.symbols('tau')
    a = sp.symbols(["a{:d}".format(i) for i in range(n)], real=True)
    alpha = sp.symbols(["alpha{:d}".format(i) for i in range(m + 1)],
                       real=True)
    polynomial = s ** n + np.array(a).dot([s ** i for i in range(n)])
    delayed = np.array(alpha).dot([s ** i for i in range(m + 1)]) * sp.exp(
        -s * tau)
    q = polynomial + delayed
    sysderivatif = [q]
    for i in range(n + m + 1):
        dernierederivee = sysderivatif[-1]
        sysderivatif.append(dernierederivee.diff(s))
    sol = sp.linsolve(sysderivatif[:-1], alpha + a).args[0]
    solnum = sol.subs({s: s0})
    solnum = solnum.subs({tau: value_tau})
    a_num = list(solnum[m + 1:])
    a_coeff_mid_generic = a_num.copy()
    a_coeff_mid_generic.append(1)
    alpha_num = list(solnum[:m + 1])
    b_coeff_mid_generic = alpha_num.copy()
    qnumerique = s ** n + np.array(a_num).dot([s ** i for i in range(n)]) + \
        np.array(alpha_num).dot([s ** i for i in range(m + 1)]) * \
        sp.exp(-s * tau)
    qnumerique = qnumerique.subs(tau, value_tau)
    sysrootfinding = [qnumerique, qnumerique.diff(s)]
    sysfunc = [sp.lambdify(s, i) for i in sysrootfinding]
    rect = cx.Rectangle([-100, 10], [-100, 100])
    roots = rect.roots(sysfunc[0], sysfunc[1], rootErrTol=1e-5, absTol=1e-5,
                       M=n + m + 1)
    xroot = np.real(roots[0])
    yroot = np.imag(roots[0])
    return xroot, yroot, qnumerique


def factorization_integral_latex(n, m, s0, tau):
    factor = str(tau ** (m + 1) / spsp.factorial(m))
    parenthesis = "(s + " + str(-s0) + ")"
    power = "^{" + str(n + m + 1) + "}"
    return r"\$" + "\\Delta(s) = " + factor + parenthesis + power + \
           r"\int_0^1 t^{" + str(m) + r"} (1 - t)^{" + str(n) + \
           "} e^{-" + str(tau) + "t" + parenthesis + \
           "} \\mathrm{d}t" + "$"


def factorization_1f1_latex(n, m, s0, tau):
    factor = str(
        tau ** (m + 1) * spsp.factorial(n) / spsp.factorial(n + m + 1))
    parenthesis = "(s + " + str(-s0) + ")"
    power = "^{" + str(n + m + 1) + "}"
    return r"\$" + "\\Delta(s) = " + factor + parenthesis + power + \
           r" {}_1 F_1(" + str(m + 1) + r", " + str(n + m + 2) + ", -" + \
           str(tau) + parenthesis + ")" + "$"

