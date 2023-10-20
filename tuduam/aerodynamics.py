from math import sqrt, pi, cos, sin, tan
import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
from warnings import warn
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumulative_trapezoid
import pdb

def wing_planform(flight_perf, vtol, wing):
  """ The following functions sizes the wing planform based on the specified
  wing loading and maximum take off weight. All attributes are assigned to the wing data struct.

  :param flight_perf: FlightPerformance data structure
  :type flight_perf: FlightPerformance 
  :param vtol: VTOL data structure
  :type vtol: VTOL 
  :param wing: SingleWing data structure
  :type wing: SingleWing 
  """  

  wing.surface = vtol.mtom / flight_perf.wingloading_cruise*g
  wing.span  = np.sqrt( wing.aspect_ratio * wing.surface)
  wing.chord_root = 2 * wing.surface / ((1 + wing.taper) * wing.span)
  wing.chord_tip = wing.taper * wing.chord_root
  wing.chord_mac = (2 / 3) * wing.chord_root  * ((1 + wing.taper + wing.taper ** 2) / (1 + wing.taper))
  wing.y_mac = (wing.span / 6) * ((1 + 2 * wing.taper) / (1 + wing.taper))
  wing.sweep_le = 0.25 * (2 * wing.chord_root / wing.span) * (1 - wing.taper) + np.tan(np.radians(wing.quarterchord_sweep))
  wing.x_lemac = wing.y_mac * wing.sweep_le

def l_function(lam, spc, y, n, eps= 1e-10):
  """Weissinger-L function, formulation by De Young and Harper.

  :param lam: quarter chord sweep
  :type lam: float (radians)
  :param spc: local span/chord
  :type spc: float
  :param y: y/l = y*
  :type y: float
  :param n: eta/l = eta*
  :type n: float
  :param eps: machine error taking into account , defaults to 1e-10
  :type eps: float , optional
  :return: Weisinger L function output
  :rtype: float
  """  
 
  if abs(y-n) < eps:
    weissl = tan(lam)

  else:
    yp = abs(y)
    if n < 0.:
      weissl = sqrt((1.+spc*(yp+n)*tan(lam))**2. + spc**2.*(y-n)**2.) / \
                    (spc*(y-n) * (1.+spc*(yp+y)*tan(lam))) - 1./(spc*(y-n)) + \
                    2.*tan(lam) * sqrt((1.+spc*yp*tan(lam))**2. \
                                        + spc**2.*y**2.) / \
                    ((1.+spc*(yp-y)*tan(lam)) * (1.+spc*(yp+y)*tan(lam))) 
    else:
      weissl = -1./(spc*(y-n)) + sqrt((1.+spc*(yp-n)*tan(lam))**2. + \
                                       spc**2.*(y-n)**2.) / \
               (spc*(y-n) * (1.+spc*(yp-y)*tan(lam)))

  return weissl 

def weissinger_l(wing, alpha_root, m, plot= False): 
  """Weissinger-L method for a swept, tapered, twisted wing.


  :param wing: SingleWing datastructure, requires chord_root, chord_tip, quarterchord_sweep and washout
  :type wing: SingleWing
  :param alpha_root: Angle of attack at the root 
  :type alpha_root: float (radians)
  :param m: number of points along the span 
  :type m: integer (odd number)
  :param plot: request to plot or not, defaults to False
  :type plot: bool, optional
  :return:
      y: vector of points along span
      cl: local 2D lift coefficient cl
      ccl: cl * local chord (proportional to sectional lift)
      al_i: local induced angle of attack
      CL: lift coefficient for entire wing
      CDi: induced drag coefficient for entire wing 
  :rtype: tuple
  """  
  if (m % 2) == 0:
    m += 1
    warn(f"{m=} was not ann odd number, one spanwise point has been added to make it such")

  # Convert angles to radians
  lam = wing.quarterchord_sweep
  tw = -wing.washout

  # Initialize solution arrays
  O = m+2
  phi   = np.zeros((m))
  y     = np.zeros((m))
  c     = np.zeros((m))
  spc   = np.zeros((m))
  twist = np.zeros((m))
  theta = np.zeros((O))
  n     = np.zeros((O))
  rhs   = np.zeros((m,1))
  b     = np.zeros((m,m))
  g     = np.zeros((m,m))
  A     = np.zeros((m,m))

  # Compute phi, y, chord, span/chord, and twist on full span
  for i in range(m):
    phi[i]   = (i+1)*pi/float(m+1)                   #b[v,v] goes to infinity at phi=0
    y[i]     = cos(phi[i])                           #y* = y/l
    c[i]     = wing.chord_root + (wing.chord_tip-wing.chord_root)*y[i] #local chord
    spc[i]   = wing.span/c[i]                        #span/(local chord)
    twist[i] = abs(y[i])*tw                          #local twist

  # Compute theta and n
  for i in range(O):
    theta[i] = (i+1)*pi/float(O+1)
    n[i]     = cos(theta[i])
  n0 = 1.
  phi0 = 0.
  nO1 = -1.
  phiO1 = pi

  # Construct the A matrix, which is the analog to the 2D lift slope
  # print("Calculating aerodynamics ...")
  for j in range(m):
    # print("Point " + str(j+1) + " of " + str(m))
    rhs[j,0] = alpha_root + twist[j]

    for i in range(m):
      if i == j: b[j,i] = float(m+1)/(4.*sin(phi[j]))
      else: b[j,i] = sin(phi[i]) / (cos(phi[i])-cos(phi[j]))**2. * \
            (1. - (-1.)**float(i-j))/float(2*(m+1))

      g[j,i] = 0.
      Lj0 = l_function(lam, spc[j], y[j], n0)
      LjO1 = l_function(lam, spc[j], y[j], nO1)
      fi0 = 0.
      fiO1 = 0.
      for mu in range(m):
        fi0 += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*phi0)
        fiO1 += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*phiO1)

      for r in range(O):
        Ljr = l_function(lam, spc[j], y[j], n[r])
        fir = 0.
        for mu in range(m):
          fir += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*theta[r])
        g[j,i] += Ljr*fir
      g[j,i] = -1./float(2*(O+1)) * ((Lj0*fi0 + LjO1*fiO1)/2. + g[j,i])

      if i == j: A[j,i] = b[j,i] + wing.span/(2.*c[j])*g[j,i]
      else: A[j,i] = wing.span/(2.*c[j])*g[j,i] - b[j,i]

  # Scale the A matrix
  A *= 1./wing.span 

  # Calculate ccl
  ccl = np.linalg.solve(A, rhs)

  # Add a point at the tip where the solution is known
  y = np.hstack((np.array([1.]), y))
  ccl = np.hstack((np.array([0.]), ccl[:,0]))
  c = np.hstack((np.array([wing.chord_tip]), c))
  twist = np.hstack((np.array([tw]), twist))

  # Return only the right-hand side (symmetry)
  nrhs = int((m+1)/2)+1    # Explicit int conversion needed for Python3
  y = y[0:nrhs]
  ccl = ccl[0:nrhs]

  # Sectional cl and induced angle of attack
  cl = np.zeros(nrhs)
  al_i = np.zeros(nrhs)
  for i in range(nrhs):
    cl[i] = ccl[i]/c[i]
    al_e = cl[i]/(2*np.pi)
    al_i[i] = alpha_root + twist[i] - al_e

  # Integrate to get CL and CDi
  CL = 0.
  CDi = 0.
  area = 0.
  for i in range(1,nrhs):
    dA = 0.5*(c[i]+c[i-1]) * (y[i-1]-y[i])
    dCL = 0.5*(cl[i-1]+cl[i]) * dA
    dCDi = sin(0.5*(al_i[i-1]+al_i[i])) * dCL
    CL += dCL
    CDi += dCDi
    area += dA
  CL /= area 
  CDi /= area

  if plot:
    # Mirror to left side for plotting
    npt = y.shape[0]
    y = np.hstack((y, np.flipud(-y[0:npt-1])))
    cl = np.hstack((cl, np.flipud(cl[0:npt-1])))
    ccl = np.hstack((ccl, np.flipud(ccl[0:npt-1])))

    fig, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(y*wing.span/2, cl, 'r', y*wing.span/2, ccl/wing.chord_mac, 'b' )
    axarr[0].set_xlabel('y')
    axarr[0].set_ylabel('Sectional lift coefficient')
    axarr[0].legend(['Cl', 'cCl / MAC'], numpoints=1)
    axarr[0].grid()
    axarr[0].annotate("CL: {:.4f}\nCDi: {:.5f}".format(CL,CDi), xy=(0.02,0.95), 
                      xycoords='axes fraction', verticalalignment='top', 
                      bbox=dict(boxstyle='square', fc='w', ec='m'), color='m')

    yroot = 0.
    ytip =wing.span/2.
    xroot = [0.,wing.chord_root]
    xrootc4 =wing.chord_root/4.
    xtipc4 = xrootc4 +wing.span/2.*tan(wing.quarterchord_sweep*pi/180.)
    xtip = [xtipc4 - 0.25*wing.chord_tip, xtipc4 + 0.75*wing.chord_tip]

    ax = axarr[1]


    x = [xroot[0],xtip[0],xtip[1],xroot[1], \
         xtip[1],xtip[0],xroot[0]]
    y_plot = [yroot,ytip,ytip,yroot, \
        -ytip,   -ytip,yroot]
    xrng = max(x) - min(x)
    yrng =wing.span

    ax.plot(y_plot, x, 'k')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_xlim(-ytip-yrng/7.,ytip+yrng/7.)
    ax.set_ylim(min(x)-xrng/7., max(x)+xrng/7.)
    ax.set_aspect('equal', 'datalim')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.grid()
    ax.annotate("Area: {:.4f}\nAR: {:.4f}\nMAC: {:.4f}".format(wing.surface, 
                wing.aspect_ratio,wing.chord_mac), xy=(0.02,0.95),
                xycoords='axes fraction', verticalalignment='top',
                bbox=dict(boxstyle='square', fc='w', ec='m'), color='m')

    plt.show()


  return y*wing.span/2., cl, ccl, al_i*180./pi, CL, CDi



def lift_curve_slope( aero, wing, m, plot= False ):
  """ Returns 

  :param flight_perf: FlightPerfomance datat structurer
  :type flight_perf: FlightPerformance
  :param vtol: VTOL data struct
  :type vtol: VTOL
  :param wing: SingleWing data struct
  :type wing: SingleWing
  :param m:  number of points along the span (an odd number). 
  :type m: int

  """  


  alpha_lst = list()
  cL_lst =  list()
  induced_drag_lst = list()

  for alpha in np.arange(np.radians(-5), np.radians(15), np.pi/180):
    span_points, cl_vector, ccl_vector, local_aoa, cL, CDi = weissinger_l( wing, alpha, m)
    alpha_lst.append(alpha)
    cL_lst.append(cL)
    induced_drag_lst.append(CDi)

  aero.cL_alpha = (cL_lst[-1] - cL_lst[0])/(alpha_lst[-1] - alpha_lst[0]) 

  if plot:
    # Create a figure with 3 subplots arranged in 3 rows
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))  # 3 rows, 1 column

    # You can now customize each subplot as needed, for example:
    axs[0].plot(np.degrees(alpha_lst), cL_lst,  "k->")
    axs[0].set_xlabel("Angle of Attack")
    axs[0].set_ylabel("Lift coefficient")
    axs[0].grid()


    axs[1].plot(cL_lst, induced_drag_lst, "k->")
    axs[1].set_xlabel("Lift coefficient")
    axs[1].set_ylabel("Induced Drag coefficient")
    axs[1].grid()

    plt.tight_layout()
    plt.show()
    pass


  return aero.cL_alpha, cl_vector, local_aoa

def component_drag():
  pass

def drag_coefficient(aero, wing, alpha, m):
  """ Return the drag coefficient for a specific angle of attack

  :param wing: SingleWing dataclass
  :type wing: SingleWing
  :param alpha: Angle of attack
  :type alpha: float
  :param m: number of points along the span (an odd number). 
  :type m: _type_
  """  


  span_points, cl_vector, ccl_vector, local_aoa, cL, CDi = weissinger_l( wing, alpha, m)
  return CDi + aero.cd0_cruise



def lift_distribution(wing, alpha, m, rho, velocity ):
  """ returns a functions which outpouts the sectional lift for a given
  spanwise position

  :param wing: _description_
  :type wing: _type_
  :param alpha: _description_
  :type alpha: _type_
  :param m: _description_
  :type m: _type_
  :param rho: _description_
  :type rho: _type_
  :param velocity: _description_
  :type velocity: _type_
  :return:
      ccl: cl * local chord (proportional to sectional lift)
  :rtype: _type_
  """  
  span_points, cl_vector, ccl_vector, local_aoa, cL, CDi = weissinger_l( wing, alpha, m)

  if span_points[-1] < 1e-8:
    span_points[-1] = 0

  lift_vec = np.insert(cumulative_trapezoid(span_points, np.array(ccl_vector)*0.5*rho*velocity**2), 0,0)
  lift_func = interp1d(span_points, lift_vec)

  return lift_func
