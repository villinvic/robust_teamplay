# from background_population.bg_population import BackgroundPopulation
# from environments.matrix_form.repeated_prisoners import RepeatedPrisonersDilemmaEnv
#
# if __name__ == '__main__':
#
#
#     bg = BackgroundPopulation(RepeatedPrisonersDilemmaEnv(2))
#
#     print(bg.build_randomly(4))

import numpy as np

def project_onto_simplex(v, z=1):
    n_features = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


x = np.ones(3) / 3

loss = np.array([1, 2, np.nan])
loss -= np.nanmean(loss)

loss[np.isnan(loss)] = np.nanmean(loss)

next_x = x + loss * 0.1

x = project_onto_simplex(next_x)
print(x, x.sum())