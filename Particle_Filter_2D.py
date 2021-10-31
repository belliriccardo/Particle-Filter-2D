from data import *
from models import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer


if __name__ == '__main__':

    grid_size_x, grid_size_y = 20, 20
    bounding = 1  # bordi extra (nel plot)

    n_particles = 100

    iterations = 30

    var_mm = 0.5
    var_z = 1.5

    particles = init_particles_2D(grid_size_x, grid_size_y, n_particles)

    weights = np.ones(n_particles) * (1/n_particles)

    times = []

    for iteration in range(iterations):

        plt.plot(particles[0, :], particles[1, :], 'o',
                 mfc='none', label='particles', markersize=5)
        plt.plot(M[0, :], M[1, :], 'rx', label='beacons')
        plt.plot(np.mean(particles[0, :]), np.mean(
            particles[1, :]), 'k^', label='mean', mfc='magenta', markersize=8)
        plt.plot(np.median(particles[0, :]), np.median(
            particles[1, :]), 'k^', label='median', mfc='green', markersize=8)
        plt.xlim(-bounding, grid_size_x+bounding)
        plt.ylim(-bounding, grid_size_y+bounding)
        plt.xlabel('asse X')
        plt.ylabel('asse Y')

        plt.title(
            f'Set di particelle posterior all\'iterazione {iteration} di {iterations}.')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

        print(f'Iterazione numero {iteration}:')
        start = timer()  # misura performance

        for i in range(n_particles):

            particles[:, i] = sample_motion_model(
                particles[:, i], U[:, iteration], var_mm)

            weights[i] = measurement_model(
                Z[:, iteration], particles[:, i], M, var_z, iteration)

        particles = resample(particles, weights)

        # Report sulle performance
        end = timer()
        print(
            f'Tempo di esecuzione ciclo {iteration}: {(end-start)*1e3:.3f} millisecondi.')
        times.append((end-start)*1e3)
        print(f'Media dei tempi: {np.mean(times):.3f} millisecondi per ciclo.')
