import numpy as np
import math
# from icecream import ic  # simple debugger

pi, e = math.pi, math.e


def fast_gaussian(x, mean, variance):
    # Implementazione veloce del valore (x) di una gaussiana
    # con media (mean) e varianza (variance). Equivalente di:
    # scipy.stats.norm.pdf(x, mean, variance)
    return ((1/((2*pi*variance)**0.5)) * (e ** (((-0.5)*((x-mean)/variance) ** 2))))


def fast_2D_distance(x_1, y_1, x_2, y_2):
    # Implementazione veloce della distanza 2D tra due punti.
    # Equivalente di:
    # numpy.linalg.norm(numpy.array([x_1, y_1]) - numpy.array([x_2, y_2]))
    return ((((x_1-x_2)**2) + ((y_1-y_2)**2)) ** 0.5)


def init_particles_2D(grid_size_x, grid_size_y, n_particles):
    # Devo restituire (n_particles) con x distribuita uniformemente
    # tra 0 e (grid_size_x) e la y tra 0 e (grid_size_y).

    points = np.ndarray((2, n_particles))

    for i in range(n_particles):
        points[0, i] = np.random.default_rng().uniform(0, grid_size_x)
        points[1, i] = np.random.default_rng().uniform(0, grid_size_y)

    return points


def sample_motion_model(particle, movement, variance):
    # Aggiorno il punto con la sua propagazione lungo gli assi, con errore gaussiano.
    # Da ricordare che il secondo parametro di .normal() è la deviazione standard,
    # e non la varianza definita nello script principale; ne prendiamo la radice.

    particle[0] = particle[0] + movement[0] + \
        np.random.default_rng().normal(0, np.sqrt(variance))
    particle[1] = particle[1] + movement[1] + \
        np.random.default_rng().normal(0, np.sqrt(variance))

    return particle


def measurement_model(Z, particle, M, variance, iteration):

    # Prima implementazione (classica):

    # dist_1 = np.linalg.norm(particle - M[:, 0])
    # dist_2 = np.linalg.norm(particle - M[:, 1])
    # dist_3 = np.linalg.norm(particle - M[:, 2])

    # mu_1 = dist_1 - Z[0]
    # mu_2 = dist_2 - Z[1]
    # mu_3 = dist_3 - Z[2]

    # w_1 = fast_gaussian(mu_1, 0, variance)
    # w_2 = fast_gaussian(mu_2, 0, variance)
    # w_3 = fast_gaussian(mu_3, 0, variance)

    # return w_1 * w_2 * w_3

    # Seconda implementazione: più generica ed efficiente.

    if(Z.shape[0] != M.shape[1]):
        raise ValueError(f'Il numero di misure Z ({Z.shape[0]}) al timestamp {iteration}' +
                         f' e\' diverso dal numero di beacon ({M.shape[1]}).')

    mu_array, p_array = np.ndarray(Z.shape[0]), np.ndarray(Z.shape[0])

    for i in range(Z.shape[0]):
        mu_array[i] = Z[i] - fast_2D_distance(
            particle[0], particle[1], M[0, i], M[1, i])

        p_array[i] = fast_gaussian(mu_array[i], 0, variance)

    return np.prod(p_array)


def resample(particles, weights):
    # Uso l'algoritmo "Low variance sampler" descritto in PR.

    # Per prima cosa normalizzo i pesi per avere una CDF valida.
    weights = weights / np.sum(weights)

    new_particles = np.zeros_like(particles)
    n_particles = len(weights)

    r = np.random.default_rng().uniform(0, 1/n_particles)
    c = weights[0]
    i = 0
    for m in range(n_particles):
        u = r + (m) * (1/n_particles)  # rimosso il (-1) dallo pseudocodice
        while u > c:
            i = i + 1
            c = c + weights[i]

        new_particles[:, m] = particles[:, i]

    return new_particles
