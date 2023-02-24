from wavefronts.verification import get_Spice_Arrays
from wavefronts.generation import generate_interface_data
from wavefronts.plotting import plot_time_interconnect
import matplotlib.pyplot as plt

interface = generate_interface_data(L_impedance = '500',C_impedance = '20')

# get spice kwarg array
spice_kwargs = interface.data_input.SPICE_input_values

# set step-size to be GCD/8 to be safe
step_size = interface.data_input.GCD/8

# get arrays
LTSpice_outputs = get_Spice_Arrays(**spice_kwargs,Step_size=str(step_size))

fig,ax = plt.subplots()

# plot lumped current
ax.plot(LTSpice_outputs['Time'],LTSpice_outputs['Capacitor_Current_Tx'])

# plot distributed from LTSpice
ax.plot(LTSpice_outputs['Time'],LTSpice_outputs['Capacitor_Current_Circuit'])

# plot distributed from simulator
plot_time_interconnect(interface.data_output_ordered,ax,'Current Capacitor',True)

ax.legend(['LT-lumped','LT-dist','Wavefronts'])

plt.show()