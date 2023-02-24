from wavefronts.generation import generate_interface_data
import wavefronts.plotting  as wp
import matplotlib.pyplot as plt

# simulate a resonator circuit :
# ==============================
#   inductor: impedance = 800 Ohm, timedelay = 3ns
#   capacitor: impedance = 0.65 Ohm, timedelay = 7ns
interface_1 = generate_interface_data(L_time = '3e-9', C_time = '7e-9')

# plot all wavefroms at terminals of capacitor and inductor
# for the plotting module 'make' indicates that the axes are made internally
wp.make_time_interconnect_all(interface_1)

# plot reflection diagram for interface
# have to make subplot axes for 'plot' modules requiring an 'ax'
fig_reflection_voltage, ax_reflection_voltage = plt.subplots()
wp.plot_refelction_diagram(interface_1,ax_reflection_voltage,True) # True for plot voltage

# plot all commutative and multiplicative interconnect fanout for cthe capacitor
wp.make_commutative_merged_lines(interface_1,'interconnect','voltage capacitor')

# plot the spatial distribution of voltage and current at 100.073ns
wp.make_spatial_voltage_and_current('100.073e-9',interface_1)


plt.show()
