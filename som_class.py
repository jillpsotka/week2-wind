import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import pandas as pd
from scipy.interpolate import interp2d

#set random seed for reproducability
s = 0
seed = np.random.seed(s)


class SOM:
    "This is the self-organizing map class."

    def __init__(self, Nx, Ny, obs, N_epochs, colour=True, sigma_max=1.0, sigma_min=0.1, linewidth=2,
                 colours_list='default'):

        """
        Initialize attributes

        obs: array of all obserations; rows = observations, columns = dimensions
        Nx: number of map nodes in x-direction (number of columns)
        Ny: number of map nodes in y-direction (number of rows)
        sigma_max: maximum standard deviation for gaussian neighbourhood
        sigma_min: minimum standard deviation for gaussian neighbourhood
        N_epochs: the number of epochs to train the map for

        """

        self.Nx = Nx
        self.Ny = Ny
        self.N_nodes = self.Nx * self.Ny
        self.obs = obs
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.N_epochs = N_epochs
        self.sigmas = np.zeros(self.N_epochs)
        self.colour = True
        self.linewidth = linewidth
        self.colours_list = colours_list

    def initialize_map(self, node_shape='hex'):

        """
        Use principal component analysis to initialize the SOM as a grid in the space of PC1 and PC2

        node_shape: 'hex' or 'rect'
        """

        # PCA initialize
        pca = PCA(n_components=2)
        PCs = pca.fit_transform(self.obs)
        frac_var = pca.explained_variance_ratio_
        var = pca.explained_variance_
        std = var ** 0.5
        eigvecs = pca.components_

        # evenly space out Nx points along first eigenvector, Ny points along second eigenvector
        mid = np.mean(self.obs, axis=0)
        x_PC_space = np.linspace(-std[0], std[0], self.Nx)
        y_PC_space = np.linspace(-std[1], std[1], self.Ny)

        node = 0
        z_init = np.zeros((self.Nx * self.Ny, len(
            self.obs[0, :])))  # numer of dimensions of z is the same as the number of dimensions in the observations
        i_init = np.zeros((self.Nx * self.Ny, 2))  # i is 2-dimensional (since map-space is 2D)
        for kk in range(self.Nx):
            for jj in range(self.Ny):

                z_init[node, :] = mid + x_PC_space[kk] * eigvecs[0, :] + y_PC_space[jj] * eigvecs[1,
                                                                                          :]  # row, column == x, y position in data-space

                if node_shape == 'rect':
                    i_init[node, :] = np.array([kk, jj])  # row, column == x, y position in map-space
                elif node_shape == 'hex':
                    ix = np.mod(jj, 2) * 0.5 + kk  # odd rows (jj-->Ny-->rows) have shift from stacking
                    iy = np.sqrt(3) / 2 * jj
                    i_init[node, :] = np.array([ix, iy])

                node += 1

        self.z_init = z_init
        self.i_init = i_init

        self.z = z_init
        self.i = i_init

    def neighbourhood(self, x, sigma=1):

        """
        Calculates the gaussian neighbourhood distance.
        x: distance from gaussian mean
        sigma: standard deviation of gaussian

        """

        # gaussian
        d = np.exp(-0.5 * (x / sigma) ** 2)
        return d

    def sigma_linear(self, epoch):

        """
        This function returns sigma (standard deviation of neighbourhood function), which undergoes a linear decrease over
        all epochs, starting at sigma_max and ending at sigma_min

        sigma_max: initial sigma at epoch
        sigma_min: final minimum sigma at last epoch
        epoch: current training epoch (int)

        """

        sigmas = np.linspace(self.sigma_max, self.sigma_min, self.N_epochs + 1)
        sigma = sigmas[epoch]

        return sigma

    def train_step(self, obs, sigma):

        """
        This function performs one training step (epoch).  Each observation is passed through the map and the map nodes are updated.

        obs: array of all obserations; rows = observations, columns = dimensions
        sigma: the standard deviation of gaussian neighbourhood function

        """

        for kk, ob in enumerate(obs):
            BMU = np.argmin(np.linalg.norm(ob - self.z, axis=1))  # current BMU
            zk = self.z[BMU, :]  # current node (in data space) of the BMU node
            ik = self.i[BMU, :]  # current node (in map space) of the BMU node
            i2 = np.linalg.norm(self.i - ik, axis=-1) ** 2  # (i_j - i_k)^2, for all j

            self.z = self.z + learning_rate * self.neighbourhood(i2, sigma)[:, None] * (
                        ob - self.z)  # update nodes in data space

    def train_map(self, learning_rate):

        """
        This function iteratively trains the map.

        N_epochs: integer, the number of epochs to train the map
        learning_rate: float, the learning rate to use in the map update calculation

        """

        sigma_max = self.sigma_max
        sigma_min = self.sigma_min

        N_nodes = self.Nx * self.Ny
        N_dims = np.shape(self.obs)[-1]
        z_epochs = np.zeros((N_nodes, N_dims, self.N_epochs + 1))  # nodes in data-space at end of each epoch
        z_epochs[:, :, 0] = self.z_init

        for epoch in range(N_epochs):  # for each epoch

            # shuffle data -- present data to map in a different order
            obs_shuffle = np.copy(self.obs)
            np.random.shuffle(obs_shuffle)

            # calculate neighbourhood radius
            sigma = self.sigma_linear(epoch)
            self.sigmas[epoch] = sigma

            # do one training step
            self.train_step(obs=obs_shuffle, sigma=sigma)
            z_epochs[:, :, epoch + 1] = self.z

        self.z_epochs = z_epochs

    def plot(self):

        """
        This function plots the nodes of the map in subplots.

        """

        border = 0.1 * (np.max(self.z) - np.min(self.z))
        indices = np.arange(self.Nx * self.Ny).reshape(self.Nx, self.Ny).T.flatten()
        bmus = BMUs(self)
        dummy, bmus_colours = colourmap_2D(colours_list=self.colours_list, Nx=self.Nx, Ny=self.Ny)

        fig, axes = plt.subplots(nrows=self.Ny, ncols=self.Nx, figsize=(3 * self.Nx, 3 * self.Ny))

        for kk, ax in enumerate(axes.flatten()):  # for each axis (subplot)
            var = self.z[indices[kk], :]
            if self.colour == True:
                ax.plot(var, color=bmus_colours[indices[kk], :], linewidth=self.linewidth)
            else:
                ax.plot(var, linewidth=self.linewidth)
            ax.set_ylim(bottom=np.min(self.z) - border, top=np.max(self.z) + border)
            ax.set_title(
                'Node ' + str(indices[kk]) + '\nFreq = ' + str(np.round(BMU_frequency(self)[indices[kk]], decimals=2)))

        plt.tight_layout()

        return fig, axes

    def QE(self):

        """
        This function calculates the quantization error of the SOM

        """

        bmus = BMUs(self)
        d = np.zeros(len(self.obs))
        for kk, ob in enumerate(self.obs):
            d[kk] = np.sum(np.abs(ob - self.z[bmus[kk], :]))
        QE = np.mean(d)

        return QE

    def TE(self):

        """
        This function calculates the topographic error of the SOM

        """

        close_second_bmu = np.zeros(len(self.obs))  # is the second-best-matching-unit a neighbouring node?
        bmus = BMUs(self)

        second_bmus = np.zeros(len(self.obs), dtype='int')

        for jj, ob in enumerate(self.obs):
            second_bmu = np.argsort(np.linalg.norm(ob - self.z, axis=1))[1]
            second_bmus[jj] = second_bmu

        for kk in range(len(self.obs)):

            bmu_grid = np.zeros(self.N_nodes)
            second_bmu_grid = np.zeros(self.N_nodes)

            bmu_grid[bmus[kk]] = 1
            second_bmu_grid[second_bmus[kk]] = 1

            bmu_grid = np.reshape(bmu_grid, (self.Nx, self.Ny)).T
            second_bmu_grid = np.reshape(second_bmu_grid, (self.Nx, self.Ny)).T

            inds = np.argwhere(bmu_grid == 1)
            second_inds = np.argwhere(second_bmu_grid == 1)

            d = np.abs(inds - second_inds)

            if np.max(d) <= 1:
                close_second_bmu[kk] = 1

        TE = 1 - np.sum(close_second_bmu) / len(close_second_bmu)

        return TE


def BMUs(SOM):
    """
    Calculates the best matching unit (BMU) for each observation.

    """

    BMUs = np.zeros(len(SOM.obs), dtype='int')

    for kk, ob in enumerate(SOM.obs):
        BMU = np.argmin(np.linalg.norm(ob - SOM.z, axis=1))
        BMUs[kk] = BMU

    return BMUs


def BMU_frequency(SOM):
    bmus = BMUs(SOM)
    frequency = np.zeros(SOM.Nx * SOM.Ny)
    for node in range(SOM.Nx * SOM.Ny):
        n_node = len(np.argwhere(bmus == node))
        frequency[node] = n_node / len(SOM.obs)

    return frequency


def colourmap_2D(Nx, Ny, colours_list='default1'):
    # to choose your own beautiful colourmap colours, check out https://coolors.co/ and copy the RGB values of 4 colours

    if colours_list == 'default1':

        colours_list = np.array([[229, 99, 153],  # colour at corner (0,0)
                                 [35, 31, 32],  # colour at corner (0,1)
                                 [222, 110, 75],  # colour at corner (1,0)
                                 [240, 223, 173]]) / 256  # colour at corner (1,1)

    elif colours_list == 'default2':  # 'PiBuRdPu'

        colours_list = np.array([[164, 3, 111],  # colour at corner (0,0)
                                 [4, 139, 168],  # colour at corner (0,1)
                                 [22, 219, 147],  # colour at corner (1,0)
                                 [239, 234, 90]]) / 256  # colour at corner (1,1)

    elif colours_list == 'pink_blue_red_purple':  # 'PiBuRdPu'

        colours_list = np.array([[229, 99, 153],  # colour at corner (0,0)
                                 [109, 169, 217],  # colour at corner (0,1)
                                 [251, 35, 75],  # colour at corner (1,0)
                                 [64, 68, 99]]) / 256  # colour at corner (1,1)

    elif colours_list == 'pinks':  # 'Pinks'

        colours_list = np.array([[210, 204, 161],  # colour at corner (0,0)
                                 [255, 168, 169],  # colour at corner (0,1)
                                 [247, 134, 170],  # colour at corner (1,0)
                                 [161, 74, 118]]) / 256  # colour at corner (1,1)

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]

    zR = [colours_list[kk][0] for kk in range(4)]
    zG = [colours_list[kk][1] for kk in range(4)]
    zB = [colours_list[kk][2] for kk in range(4)]

    fR = interp2d(x, y, zR)
    fG = interp2d(x, y, zG)
    fB = interp2d(x, y, zB)

    xnew = np.linspace(0, 1, Nx)
    ynew = np.linspace(0, 1, Ny)

    zRnew = fR(xnew, ynew)
    zGnew = fG(xnew, ynew)
    zBnew = fB(xnew, ynew)

    colours = np.zeros((Ny, Nx, 3))
    colours[:, :, 0] = zRnew
    colours[:, :, 1] = zGnew
    colours[:, :, 2] = zBnew

    colours_flat = colours.transpose(1, 0, 2).reshape(Nx * Ny, 3)

    return colours, colours_flat