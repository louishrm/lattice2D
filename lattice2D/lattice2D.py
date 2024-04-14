import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Lattice2D:
    """A class to represent a 2D lattice.

    The 2D lattice is specified by a primitive unit cell and the positions of the atoms in the cell.
    The class generates a finite supercell of the lattice, specified by the supercell coefficients.
    """

    def __init__(self, primitive_lattice_vectors, atom_coords, 
                 supercell_coefficients):
        """Initialize a 2D lattice.
        
        Args: 
            primitive_lattice_vectors: numpy ndarray of shape (2,2) containing the coordinates of the primitive lattice vectors a1,a2.

            atom_coords: numpy ndarray of shape (nac,2) where nac is the number of atoms in the unit cell. The coordinates of the atoms in the unit cell in fractional coordinates.

            supercell_coefficients: numpy ndarray of shape (4,) containing the coefficients of the supercell expansion vectors A1 = m1*a1 + n1*a2 and A2 = m2*a1 + n2*a2.
        """
        self.a1, self.a2 = primitive_lattice_vectors[0],primitive_lattice_vectors[1] #primitive lattice vectors
        self.atom_coords = atom_coords #coordinates of the atoms in the unit cell in fractional coordinates
        self.m1,self.n1,self.m2,self.n2 = supercell_coefficients #supercell coefficients
        self.A1 = self.m1*self.a1 + self.n1*self.a2 #first supercell lattice vector
        self.A2 = self.m2*self.a1 + self.n2*self.a2 #second supercell lattice vector

        self.nuc = self.m1*self.n2 - self.m2*self.n1 #number of unit cells 
        self.nac = len(atom_coords) #number of atoms in the unit cell
        self.M = self.nuc*self.nac #number of sites in the supercell
        self.sites = self.construct_supercell()

            

    def supercell_expansion(self, r):
        """Compute the expansion coefficients of a point.
        
        Args:
            r: numpy ndarray of shape (2,) containing fractional coordinates of the point.
            
        Returns:
            a: expansion coefficient of x*a1 + y*a2 in the supercell basis.
            b: expansion coefficient of x*a1 + y*a2 in the supercell basis.
        """
        x,y = r[0], r[1]
        a,b = (x*self.n2-y*self.m2)/self.nuc, (y*self.m1-x*self.n1)/self.nuc
        return a,b


    def construct_supercell(self):
        """Construct the supercell of the lattice.
        
        Returns:
            sites: numpy ndarray of shape (M,2) containing the fractional coordinates of the sites in the supercell.
        """
        #find the largest 'diagonal' cell enclosing the supercell
        a1max = max(self.m1, self.m2, self.m1+self.m2) 
        a2max = max(self.n1, self.n2, self.n1+self.n2)
        sites= [] #list to store the sites in the supercell

        for i in range(-a1max, a1max+1):
            for j in range(-a2max, a2max+1):
                for coords in self.atom_coords:
                    r = coords + np.array([i,j])
                    a,b = self.supercell_expansion(r)
                    if a>=0 and a<1 and b>=0 and b<1: #finds the sites in the parallelogram spanned by A1 and A2
                        sites.append(r)

        return np.array(sites)


    def distance(self, r1, r2):

        """Compute the euclidean distance between two points r1 and r2.
        
        Args:
            r1: numpy ndarray of shape (2,) containing the fractional coordinates of the first point.
            r2: numpy ndarray of shape (2,) containing the fractional coordinates of the second point.
            
        Returns:
            distance: float, the euclidean distance between the two points.
        """
        r1, r2 = r1[0]*self.a1 + r1[1]*self.a2, r2[0]*self.a1 + r2[1]*self.a2
        diff = r2-r1
        distance = np.linalg.norm(diff)
        return distance

    def periodic_distance(self, r1,r2):
        """Compute the periodic distance between two points r1 and r2 using the minimum image convention.
        
        Args:
            r1: numpy ndarray of shape (2,) containing the fractional coordinates of the first point.
            r2: numpy ndarray of shape (2,) containing the fractional coordinates of the second point.
            
        Returns:
            distance: float, the periodic distance between the two points.
        """
        A1,A2 = np.array([self.m1, self.n1]), np.array([self.m2, self.n2])
        vectors_to_check = [np.zeros_like(A1), A1,A2, A1+A2, -A1, -A2, -A1-A2, A1-A2, -A1+A2]
        dist = min([self.distance(r1+v, r2) for v in vectors_to_check])  
        return dist

  
    def periodic_site(self, r):
        """Find the equivalent image of a point r in the supercell.
        
        Args:
            r: numpy ndarray of shape (2,) containing the fractional coordinates of the point.
        
        Returns:
            r: numpy ndarray of shape (2,) containing the fractional coordinates of the equivalent image of the point.
        """
        a,b = self.supercell_expansion(r)
        a,b = a%1, b%1
        r = np.array([a*self.m1+b*self.m2,a*self.n1+b*self.n2])
        return r
    

    def distance_matrix(self,sites=None):
        """Get the distance matrix of the supercell.
        
        Args:
            sites: numpy ndarray of shape (M,2) containing the fractional coordinates of the sites in the supercell. By default,
            it is set to self.sites.
            
        Returns:
            D: numpy ndarray of shape (M,M) containing the distance matrix of the supercell. The elements i,j of the
            distance matrix are the shortest path distance between the sites i and j.
        """
        if sites is None:
            sites = self.sites
        
        #first, get the unique distances supported by the supercell
        D = np.zeros((self.M, self.M))
        for i in range(self.M):
            for j in range(i+1, self.M):
                dist = self.periodic_distance(sites[i], sites[j])
                D[i,j] = D[j,i] = round(dist,5)

        unique_distances = np.sort(np.unique(D))
        
        #create a dictionary mapping the distances to their ranks
        distance_to_rank = {round(dist, 5): rank for rank, dist in enumerate(unique_distances, start=0)}

        #create the distance matrix
        D = np.zeros((self.M, self.M))
        for i in range(self.M):
            for j in range(i+1, self.M):
                dist = self.periodic_distance(sites[i], sites[j])
                D[i,j] = D[j,i] = distance_to_rank[round(dist, 5)]

        return D


    def sites_real(self,sites=None):
        """Get the real space coordinates of the sites in the supercell.
        
        Args:
            sites: numpy ndarray of shape (M,2) containing the fractional coordinates of the sites in the supercell. By default,
            it is set to self.sites.
            
        Returns:
            sites_real: numpy ndarray of shape (M,2) containing the real space coordinates of the sites in the supercell.
        """
        if sites is None:
            sites = self.sites

        sites_real = np.zeros((self.M,2))
        for i,(x,y) in enumerate(sites):
            r = x*self.a1 + y*self.a2
            sites_real[i]=r
        return sites_real
    

    def plot_supercell(self, sites = None, padding=0.2):
        """Plot the supercell of the lattice.
        
        Args:
            padding: float, the padding around the supercell in the plot. By default, it is set to 0.2.
            
        Returns:
            None, a matplotlib plot of the lattice with the supercell and the sites labelled in the supercell is displayed.
            The edges of the supercell are drawn as black lines and the points in the supercell are drawn as red balls.
        """
        if sites is None:
            sites = self.sites
        
        A1 = self.m1*self.a1 + self.n1*self.a2
        A2 = self.m2*self.a1 + self.n2*self.a2
        vertices = np.array([np.zeros(2), A1, A1+A2, A2])
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.add_patch(patches.Polygon(vertices, closed=True, fill=None, edgecolor='black'))

        copies = 2
        for n in range(-copies, copies+1):
            for m in range(-copies, copies+1):
                shift = n*A1 + m*A2
                for (x,y) in sites:
                    r = x*self.a1 + y*self.a2 + shift
                    ax.scatter(r[0], r[1], c='blue')

        
        for index, (x,y) in enumerate(sites):
            r = x*self.a1 + y*self.a2
            ax.scatter(r[0], r[1], c='red')
            ax.annotate(index, r)

        
        # Calculate the range of x and y coordinates
        x_range = np.ptp(vertices[:, 0])
        y_range = np.ptp(vertices[:, 1])

        # Calculate the center of the vertices
        center = np.mean(vertices, axis=0)

        # Set the padding (you can adjust this value)
        padding = max(x_range, y_range) * padding

        # Set the limits of the plot to center around the polygon
        ax.set_xlim(center[0] - x_range / 2 - padding, center[0] + x_range / 2 + padding)
        ax.set_ylim(center[1] - y_range / 2 - padding, center[1] + y_range / 2 + padding)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
