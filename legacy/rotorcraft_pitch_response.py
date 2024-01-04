import numpy as np
import matplotlib.pyplot as plt

def wrapper_func():
    lok = 6  # Lock number (-)
    mass = 2200  # Rotorcraft mass (kg)
    vtip = 200  # Rotor tip speed (m/sec)
    rotor_radius = 7.32  # Rotor radius (m)
    iy = 10625  # Rotorcraft moment of inertia (kgm^2)
    h = 1  # Distance from vehicle CG to rotor hub
    kbeta = 46000  # Rotor hinge spring hingeless (Nm)
    kbeta0 = 0  # Rotor hinge spring teetering (Nm)
    omega = vtip / rotor_radius  # Rotor rpm
    N = 3  # Number of blades

    thiy = (mass * 9.81 * h + N / 2 * kbeta) / iy
    thiy0 = (mass * 9.81 * h + N / 2 * kbeta0) / iy

    # Starting integration for the pitch equation of motion
    t_end = 10  # Time (sec)
    step = 0.01  # Time step (sec)
    t_arr = int(t_end / step)  # Time step

    # INITIAL CONDITIONS
    q = np.zeros(t_arr + 1)  # Pitch rate (rad/sec)
    q0 = np.zeros(t_arr + 1)  # Initial pitch rate (rad/sec)
    t = np.zeros(t_arr + 1)  # Initial time t=0 sec
    input_cyc = np.radians(-1)  # Pilot cyclic control (1 deg/sec)

    # STARTING THE PROGRAM
    for i in range(t_arr):
        a1 = (-16 / lok) * (q[i] / omega)  # Calculation flapping angle (rad)
        a10 = (-16 / lok) * (q0[i] / omega)

        dq = -thiy * (-a1 + input_cyc)  # Pitch rate response q for hingeless rotorcraft
        dq0 = -thiy0 * (-a10 + input_cyc)  # Pitch rate response q for teetering rotorcraft

        # INTEGRATION
        q[i + 1] = q[i] + dq * step
        q0[i + 1] = q0[i] + dq0 * step
        t[i + 1] = t[i] + step

    # Convert q and q0 from radians/second to degrees/second for plotting
    q_deg = np.degrees(q)
    q0_deg = np.degrees(q0)

    plt.plot(t, q_deg, label='Hingeless Rotorcraft')
    plt.plot(t, q0_deg, label='Teetering Rotorcraft')
    plt.xlabel('Time (sec)')
    plt.ylabel('q (deg/sec)')
    plt.axis([0, 10, 0, 12])
    plt.legend()
    plt.show()


    import numpy as np
    import matplotlib.pyplot as plt

    # Constants
    lok = 6  # Lock number (-)
    mass = 2200  # Rotorcraft mass (kg)
    vtip = 200  # Rotor tip speed (m/s)
    rotor_radius = 7.32  # Rotor radius (m)
    iy = 10625  # Rotorcraft moment of inertia (kgm^2)
    h = 1  # Distance from vehicle CG to rotor hub
    kbeta = 46000  # Rotor hinged (Nm)
    kbeta0 = 0  # Rotor hinge  (Nm)
    omega = 50  # Rotor rpm
    N = 3  # Number of blades
    thiy = (mass * 9.81 * h + N/2 * kbeta) / iy
    thiy0 = (mass * 9.81 * h + N/2 * kbeta0) / iy
    input_cyc = -1 * np.pi / 180  # Pilot cyclic control (1 deg/sec)

    # Starting integration for the pitch equation of motion
    t_end = 10  # Time (sec)
    step = 0.001  # Time step (sec)
    t_arr = np.arange(0, t_end, step)

    # Initialize arrays

    q = -input_cyc * omega * lok / 16 * (1 - np.exp(-16 / (lok * omega) * thiy * t_arr))
    q0 = -input_cyc * omega * lok / 16 * (1 - np.exp(-16 / (lok * omega) * thiy0 * t_arr))

    # Plot the results for the first 100 time steps
    plt.plot(t_arr, np.degrees(q), label='Hingeless Rotorcraft')
    plt.plot(t_arr, np.degrees(q0), label='Teetering Rotorcraft')
    plt.xlabel('Time (sec)')
    plt.ylabel('q (deg/sec)')
    plt.legend()
    plt.show()
