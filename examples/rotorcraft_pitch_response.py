import numpy as np
import matplotlib.pyplot as plt

lok = 6  # Lock number (-)
mass = 2200  # Rotorcraft mass (kg)
vtip = 200  # Rotor tip speed (m/sec)
straal = 7.32  # Rotor radius (m)
iy = 10625  # Rotorcraft moment of inertia (kgm^2)
h = 1  # Distance from vehicle CG to rotor hub
kbeta = 46000  # Rotor hinge spring hingeless (Nm)
kbeta0 = 0  # Rotor hinge spring teetering (Nm)
om = vtip / straal  # Rotor rpm
N = 3  # Number of blades

thiy = (mass * 9.81 * h + N / 2 * kbeta) / iy
thiy0 = (mass * 9.81 * h + N / 2 * kbeta0) / iy

# Starting integration for the pitch equation of motion
teind = 10  # Time (sec)
stap = 0.01  # Time step (sec)
aant = int(teind / stap)  # Time step

# INITIAL CONDITIONS
q = np.zeros(aant + 1)  # Pitch rate (rad/sec)
q0 = np.zeros(aant + 1)  # Initial pitch rate (rad/sec)
t = np.zeros(aant + 1)  # Initial time t=0 sec
cyc = -1 * np.pi / 180  # Pilot cyclic control (1 deg/sec)

# STARTING THE PROGRAM
for i in range(aant):
    a1 = (-16 / lok) * (q[i] / om)  # Calculation flapping angle (rad)
    a10 = (-16 / lok) * (q0[i] / om)

    dq = -thiy * (-a1 + cyc)  # Pitch rate response q for hingeless rotorcraft
    dq0 = -thiy0 * (-a10 + cyc)  # Pitch rate response q for teetering rotorcraft

    # INTEGRATION
    q[i + 1] = q[i] + dq * stap
    q0[i + 1] = q0[i] + dq0 * stap
    t[i + 1] = t[i] + stap

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
straal = 7.32  # Rotor radius (m)
iy = 10625  # Rotorcraft moment of inertia (kgm^2)
h = 1  # Distance from vehicle CG to rotor hub
kbeta = 46000  # Rotor hinge spring hingeless (Nm)
kbeta0 = 0  # Rotor hinge spring teetering (Nm)
om = 50  # Rotor rpm
N = 3  # Number of blades
thiy = (mass * 9.81 * h + N/2 * kbeta) / iy
thiy0 = (mass * 9.81 * h + N/2 * kbeta0) / iy

# Starting integration for the pitch equation of motion
teind = 10  # Time (sec)
stap = 0.1  # Time step (sec)
aant = int(teind / stap)  # Time step

# Initialize arrays
q = np.zeros(aant+1)  # Pitch rate (rad/sec)
q0 = np.zeros(aant+1)  # Initial pitch rate (rad/sec)
t = np.zeros(aant+1)  # Initial time t=0 sec
cyc = -1 * np.pi / 180  # Pilot cyclic control (1 deg/sec)

# Starting the program
for i in range(aant+1):
    t[i] = i * stap
    q[i] = -cyc * om * lok / 16 * (1 - np.exp(-16 / (lok * om) * thiy * t[i]))
    q0[i] = -cyc * om * lok / 16 * (1 - np.exp(-16 / (lok * om) * thiy0 * t[i]))

# Plot the results for the first 100 time steps
plt.plot(t[:100], np.degrees(q[:100]), label='Hingeless Rotorcraft')
plt.plot(t[:100], np.degrees(q0[:100]), label='Teetering Rotorcraft')
plt.xlabel('Time (sec)')
plt.ylabel('q (deg/sec)')
plt.legend()
plt.show()
