import subprocess
import os
import numpy as np
import time


# Path to the XFOIL executable
xfoil_path = r'C:\Users\damie\Xfoil699src\xfoil.exe'
save_path = r'C:\Users\damie\Downloads\data1' # Path can not be too long as it wil fail at some point
naca_airfoil = "4412"
max_iter = "40"

# Reynolds range
start_reyn = 1e5
ending_reyn = 1e5
step_reyn = 1e6

#AoA range
start_alpha = "-5"
end_alpha = "16"
step_alpha = "0.1"

# XFOIL commands  start up command
startup_commands = [
    'naca ' + naca_airfoil,
    'oper',
    'iter' + max_iter,
    "visc 1000000"
]

# Run XFOIL
process = subprocess.Popen(xfoil_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

for command in startup_commands:
    process.stdin.write(command + '\n')  # Send each command
    process.stdin.flush()  # Flush the buffer

for  Reyn in  np.arange(start_reyn, ending_reyn + step_reyn, step_reyn):
    Reyn = int(Reyn)
    process.stdin.write("r"+ str(Reyn)  + '\n')  # Send each command
    process.stdin.write("pacc"+ '\n')  # Send each command
    process.stdin.write(os.path.join(save_path, "airfoil_Re" + str(Reyn)) +  ".txt" + '\n')  # Send each command
    process.stdin.write('\n')  # Send each command
    process.stdin.write(f'aseq {start_alpha} {end_alpha} {step_alpha}' + '\n')  # Send each command
    process.stdin.write('pacc' + '\n')  # Send each command
    process.stdin.flush()


output, error  = process.communicate()
