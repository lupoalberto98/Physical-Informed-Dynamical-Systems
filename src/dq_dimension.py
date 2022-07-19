import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.stats import linregress 
from scipy.linalg import qr
from odeintw import odeintw
from plot import plot_params_distr

class Box():
    """
    Basic class for a d-dimensional cube
    """
    def __init__(self, center, box_sizes, epsilon=1.):
        """
        Initialize a cell volume box, a n-dimensional box divieded in d_dim cubic boxes of size epsilon
        ----
        Args:
        center = the coordinates of the center of the box
        box_sizes = list of side dimension of the box
        epsilon = dimension of cubes sides
        """
        # check dimensions
        if len(center)!=len(box_sizes):
            raise ValueError("Dimension of center different from dimension of box sizes")
        
        self.dim = len(center) # dimension of box
        self.center = center
        self.box_sizes = box_sizes
        if epsilon <= 0:
            raise ValueError("Invalid epsilon value")
        
        self.epsilon = epsilon # side of small cubes 
        
        # ridefine box_sizes, possibily enlarging box to accomodate all points
        self.int_sizes = []
        for i in range(self.dim):
            n_i = math.ceil(self.box_sizes[i]/self.epsilon) # approximate to the next integer by excess
            self.box_sizes[i] = self.epsilon*n_i
            self.int_sizes.append(n_i)
            
        
        # zero points inside at the beginning
        self.num_points = 0
    
    def get_center_cube(self, indexes):
        """
        Copmute the center of a cube from int indexes
        ____
        Args:
        indexes = list of (int) indexes
        returns the coordinates of the center of the small cube of side epsilon
        """
        center_cube = []
        for i in range(self.dim):
            c_i = self.center[i] - self.box_sizes[i]/2. + self.epsilon/2. + indexes[i]*self.epsilon
            center_cube.append(c_i)
        
        return center_cube
    
    def contains(self, point):
        """
        Check if a point is inside the box
        ____
        Args:
        point = point to check (list of coordinates)
        return bool, if True is inside, else not
        """
        # check dimension of the point
        if len(point)!=self.dim:
            raise ValueError("Dimension of the point is different from dimension of the box")
        
        inside = True
        for i in range(self.dim):
            if abs(point[i]-self.center[i])>self.box_sizes[i]/2.:
                inside = False
        
        return inside
    
    def get_indexes(self, point):
        """
        Return the indexes of the cube in which the point is
        ____
        Args:
        point = point to check
        return indexes list
        """
        # check first if it is inside the box
        if not self.contains(point):
            raise ValueError("Point is not inside the box")
        
        indexes = []
        
        for i in range(self.dim):
            ind = int((point[i]+self.box_sizes[i]/2.-self.center[i])/self.epsilon)
            indexes.append(ind)
            
        return indexes
    
    def get_pdf(self, points, norm=False):
        """
        Compute the pdf of of an array of points of dim=self.dim
        ____
        Args:
        points = array of shape (num_points, dim)
        update self.pdf
        """
        # check dimensions
        if len(points.shape) != 2:
            raise ValueError("Input points have wrong number of dimensions")
            
        if self.dim != points.shape[1]:
            raise ValueError("Input points have incopatible dimension")
            
        # take number of points
        num_points = len(points)
       
        # reinitialize to zero pdf
        pdf = np.zeros(self.int_sizes)
        
        # count points in each box
        for i in range(num_points):
            indexes = self.get_indexes(points[i])
            pdf[tuple(indexes)] += 1.0
            
        # compute frquencies
        pdf = pdf/num_points*1.0
        
        # normalize
        if norm:
            pdf = pdf/(self.epsilon**self.dim)
        
        return pdf
        
class DS():
    """
    Compute statistics for dynamicsl systems attarctors
    - Dq dimension (information dimension and fractal dimension are particular cases)
    - Lyapunov exponents
    - Lyapunov dimension
    """
    def __init__(self, points, eta=1e-8):
        """
        Initialize with array of points to analysis
        ___
        Args:
        points = array of size (num_points, dim)
        """
        # check dimensions
        if len(points.shape) != 2:
            raise ValueError("Input points have wrong number of dimensions")
            
        self.points = points
        self.eta = eta
        self.num_points = points.shape[0]
        self.dim = points.shape[1]
        
        # define costume box sizes to accomodate all the points
        self.center = (np.amax(points, axis=0) + np.amin(points, axis=0))/2.
        self.box_sizes = np.amax(points, axis=0)- np.amin(points, axis=0)
        
        
        
    def d_q(self, q=1, min_eps=1, max_eps=2, num_eps=10, plot=False, filename=None):
        """
        Compute Dq of the system for q in [0,1]
        if q=1, returns information dimension
        if q=0 returns fractal dimension
        we set x=0 and x*ln(x)=0 if x<self.eta
        """
        # checks
        if min_eps >= max_eps:
            raise ValueError("Invalid interval given")
        if type(num_eps) is not int:
            raise TypeError("Number of epsilon points is not integer")
        
        if q<0 or q>1:
            raise ValueError("Q is out of range [0,1]")
        
            
        # define array of epsilon and information values 
        eps_array = np.logspace(math.log(min_eps), math.log(max_eps), num_eps, base=math.e)
        inf_array = np.zeros((num_eps))
        
        # loop over epislon values
        for i in range(num_eps):
            # define the box
            epsilon = eps_array[i]
            box = Box(self.center, self.box_sizes, epsilon)
            # compute frequencies
            pdf = box.get_pdf(self.points, norm=False)
            # flatten pdf
            pdf = pdf.flatten()
            # loop over all cubes in pdf
            for j in range(len(pdf)):
                # check if pdf[j] satisfies threshold
                if pdf[j]> self.eta:
                    if q==1:
                        inf_array[i] += pdf[j]*math.log(pdf[j])
                    else:
                        inf_array[i] += pdf[j]**q
                        
            # take logarithm
            if q!=1:
                inf_array[i] = math.log(inf_array[i])
            
        # interpolate
        self.dq_results = linregress(np.log(eps_array), inf_array)
        
        # plot
        if plot:
            fig, ax = plt.subplots(1,1, figsize=(10,5))
            ax.set_xlabel("$\\ln(\\epsilon)$")
            ax.set_ylabel("$I(\\epsilon)$")
            ax.scatter(np.log(eps_array), inf_array, c="orange")
            x = np.log(np.logspace(math.log(min_eps), math.log(max_eps), 100, base=math.e))    
            y = x*self.dq_results.slope + self.dq_results.intercept
            ax.plot(x, y)
            fig.show()
            if filename is not None:
                fig.savefig(filename)
            
        if q==1:
            return self.dq_results.slope, self.dq_results.stderr
        else:
            return self.dq_results.slope/(q-1), self.dq_results.stderr/(1-q)
    
   
    def compute_lle(self, true_system, n_div=40, dt=0.002, discard=100):
        """
        Compute local lyapunov exponents statistics and lyapunov exponents
        """
        # Initialize perturbation matrix and Local Lyapunov Exponents list
        Q = np.eye(3) # choose identity
        LLEs = []
        tau = dt*n_div

        # Define derivative function
        def df(M, t, J):
            return np.dot(J,M)

        # Loop over dataet, jump every n_time steps
        for j in range(int(self.num_points/n_div)):
            # Propagate perturbation
            M = np.eye(3)
            t = np.linspace(j*tau, (j+1)*tau, n_div)
            Jac = true_system.jacobian(self.points[j*n_div], j*tau) 
            M = odeintw(df, M, t, args=(Jac,))[-1]
            # Propagate perturbation
            V = np.matmul(M,Q)
            # QR decomposition
            Q, R= qr(V, check_finite=True)
            # Adjust sign
            sign = np.sign(np.diag(R))
            R = np.matmul(np.diag(sign), R)
            Q = np.matmul(Q, np.diag(sign))
            check_qr = np.allclose(V, np.dot(Q, R))
            # Compute local lyapunov exponents
            lles = np.log(np.diag(R))/tau
            # Append
            LLEs.append(lles)

        # convert to np.array
        LLEs = np.array(LLEs)[discard:]
        
        # compute (global) lyapunov exponents
        self.le = np.mean(LLEs, axis=0)
        self.le_std = np.std(LLEs, axis=0)
        
        return LLEs
    

    def lyap_d(self):
        """
        Compute lyapunov dimension of a system given lyapunov exponents 
        """
        # sort lyap_exp if not sorted yet
        sorted_le = np.sort(self.le)[::-1]

        # find k such that sum_k lyap_exp(k) >= 0
        sum_lyap = []
        for i in range(self.dim):
            sum_lyap.append(np.sum(sorted_le[:i+1]))

        k = np.where(np.array(sum_lyap) >= 0.)[0][-1]

        # Compute lyapunov dimension
        lyap_dim = 1+k + 1./abs(sorted_le[k+1])*sum_lyap[k]

        return lyap_dim

    
def KL_div(p_points, q_points, epsilon=1.):
    """
    compute KL divergence between data point distributions p and q
    p_points, q_points are array of size (num_points(i), dim) of which first a pdf must be computed
    """
    # check number of dimensions
    if len(p_points.shape) != 2:
        raise ValueError("Input points have wrong number of dimensions")
    if len(q_points.shape) != 2:
        raise ValueError("Input points have wrong number of dimensions")
        
    # check dimensions
    if p_points.shape[1] != q_points.shape[1]:
        raise ValueError("Distributions have different dimensions")
    
    dim = p_points.shape[1]
    
    # take maxima and minima
    max_p = np.amax(p_points, axis=0)
    max_q = np.amax(q_points, axis=0)
    min_p = np.amin(p_points, axis=0)
    min_q = np.amin(q_points, axis=0)
    max_tot = np.maximum(max_p, max_q)
    min_tot = np.minimum(min_p, min_q)
    
    # define then center and box sizes
    center = (max_tot + min_tot)/2.
    box_sizes = max_tot - min_tot
    
    # define box
    box = Box(center, box_sizes, epsilon)
    
    # compute pdf and flatten
    pdf_p = box.get_pdf(p_points, norm=True).flatten()
    pdf_q = box.get_pdf(q_points, norm=True).flatten()
    
    
    
    # check len pdfs
    if len(pdf_p)!= len(pdf_q):
        raise ValueError("Pdf distribution do not match in size")
    
    kl_div = 0.
    for i in range(len(pdf_p)):
        if pdf_p[i] > 0 and pdf_q[i]>0:
            kl_div += pdf_p[i]*np.log(pdf_p[i]/pdf_q[i])
    
    return kl_div, pdf_p, pdf_q