"""A Module responsible getting LTSpice simulation data as for verification.
    Ensure that 'LTSpice_exe_Path' in module points to your LTspice installation.
    Requires the the LC interface spice file template 'LC_Spice_Input.txt' which must be in the `wavefronts` folder.
"""

import subprocess
import wavefronts.ltspy3

# Path to installed LTSpice exe
LTSpice_exe_Path = 'C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe'
# Name of provided parameterised LC interface template
Spice_File_Template = 'wavefronts/LC_Spice_Input.txt'
Spice_File_Altered = 'wavefronts/LC_Spice_Altered.txt'

# Default Parameters and values found in provided LC_Spice_Input.txt spice input file
default_Spice_parameters ={
    'L_impedance': '100',
    'L_time': '1',
    'C_impedance': '1',
    'C_time': '1',
    'number_periods': '1',
    'L_tot': 'L_impedance*L_time/2 ',
    'C_tot': 'C_time/(2*C_impedance)',
    'Simulation_stop_time': '2*number_periods*pi*sqrt(L_tot*C_tot)',
    'Step_size': '0.01',
    'V_source': '1'
}

def get_Spice_Arrays(**new_Spice_values):
    """Runs a LTSpice simulation on a Circuit theory LC interface as well as a distributed LC interface. Returns associated arrays.
    Useful for comparing distributed effects to lumped element effects, and the distributed soltuion made in using :py:func:`generation.generate_interface_data`.

    :param new_Spice_values: A set of key-value pairs used to configure the simulaiton. This dictonary is calculatedd on the creation of a :py:class:`storage.Input_Data`, 
    storete under the parameter of `SPICE_input_values`.
    Default simulation values are as follows and will be overwritten with provided key-values. 
    
    **new_Spice_values**:
    
        - **L_impedance** (*str*) - The inductor impedance. Default is '100'.
        - **L_time** (*str*) - The inductor time constant. Default is '1'.
        - **C_impedance** (*str*) - The capacitor impedance. Default is '1'.
        - **C_time** (*str*) - The capacitor time constant. Default is '1'.
        - **number_periods** (*str*) - The number of periods to simulate. Default is '1'.
        - **L_tot** (*str*) - The total inductance. Default is 'L_impedance*L_time/2 '.
        - **C_tot** (*str*) - The total capacitance. Default is 'C_time/(2*C_impedance)'.
        - **Simulation_stop_time** (*str*) - The simulation stop time. Default is '2*number_periods*pi*sqrt(L_tot*C_tot)'.
        - **Step_size** (*str*) - The step size for the simulation. Default is '0.01'.
        - **V_source** (*str*) - The voltage of the source. Default is '1'.

        
    Returns dictionary of output arrays [np.ndarrays]:
    
        dict{ 
            - "Time",  
            - "Inductor_Voltage_Circuit",  
            - "Inductor_Current_Circuit",  
            - "Capacitor_Voltage_Circuit",  
            - "Capacitor_Current_Circuit",  
            - "Inductor_Voltage_Tx",  
            - "Inductor_Current_Tx",  
            - "Capacitor_Voltage_Tx",  
            - "Capacitor_Current_Tx"  
        }
    
    .. code-block::
        :caption: manual SPICE simulaiton
        
        from wavefronts.verification import get_Spice_Arrays
        import matplotlib.pyplot as plt

        # Do manual simulation
                    
        # Change Impedances and simulaiton timestep
        LTSpice_Arrays = get_Spice_Arrays(L_impedance = '500',C_impedance = '20', Step_size='0.1')

        # Plot Inductor votlage using Lumped circuit elements
        plt.plot(LTSpice_Arrays['Time'],LTSpice_Arrays['Inductor_Voltage_Circuit'])
        plt.title('Lumped Element analysis of Inductor Voltage')
        plt.xlabel('time (s)')
        plt.ylabel('Voltage (V)')
        plt.show()


    .. code-block::
        :caption: Using SPICE to verify output
        
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
        
    .. warning::
        does not account for the lengths of the capacitor and inductor. 
        Also only wokrs for LC osscialtor configuration with no load.
    
    """

    # read input SPICE file, replace paramters in file with provided values
    with open(Spice_File_Template, 'rb') as file:
        Data_Template = file.read()
        
        for new_key,new_value in new_Spice_values.items():
            if(default_Spice_parameters.get(new_key) is None):
                # new value cannot be found
                raise ValueError(f"No setting found for {new_key}, here are the possible options: \n{default_Spice_parameters}")
            else:
                # search string to find default value in template
                Spice_search_string = new_key+'='+default_Spice_parameters[new_key]
                # new string to replace with matched string
                Spice_new_string = new_key+'='+new_value
                
                Data_Template = Data_Template.replace(Spice_search_string.encode('ascii'),Spice_new_string.encode('ascii'))

    # create new SPICE directive with replaced values
    with open(Spice_File_Altered,'wb') as file:
        file.write(Data_Template)
        
    # run LTSpice on new SPICE directive
    subprocess.call(LTSpice_exe_Path + ' -b '+Spice_File_Altered)

    # Extract data from outputted .raw file from LTSpice execution
    data_out = wavefronts.ltspy3.SimData('wavefronts/LC_Spice_Altered.raw')

    names = data_out.variables
    values = data_out.values

    # get arrays -> "name in SPICE data"
    # "time"
    time = values[names.index(b"time")]

    # "V(l_node_circuit)" - "V(c_node_circuit)"
    Inductor_Voltage_Circuit = values[names.index(b"V(l_node_circuit)")] - values[names.index(b"V(c_node_circuit)")] 
    # "I(Inductor_circuit)"
    Inductor_Current_Circuit = values[names.index(b"I(Inductor_circuit)")]

    # "V(c_node_circuit)"
    Capacitor_Voltage_Circuit = values[names.index(b"V(c_node_circuit)")]
    # "I(Capacitor_circuit)"
    Capacitor_Current_Circuit = values[names.index(b"I(Capacitor_circuit)")]

    # "V(l_node_tx)" - "V(c_node_tx)"
    Inductor_Voltage_Tx = values[names.index(b"V(l_node_tx)")] - values[names.index(b"V(c_node_tx)")] 
    # "Ia(Inductor_tx)"
    Inductor_Current_Tx = values[names.index(b"Ia(Inductor_tx)")]

    # "V(c_node_tx)"
    Capacitor_Voltage_Tx = values[names.index(b"V(c_node_tx)")]
    # "Ia(Capacitor_tx)"
    Capacitor_Current_Tx = values[names.index(b"Ia(Capacitor_tx)")]
    
    return({
        "Time":time,
        "Inductor_Voltage_Circuit":Inductor_Voltage_Circuit,
        "Inductor_Current_Circuit":Inductor_Current_Circuit,
        "Capacitor_Voltage_Circuit":Capacitor_Voltage_Circuit,
        "Capacitor_Current_Circuit":Capacitor_Current_Circuit,
        "Inductor_Voltage_Tx":Inductor_Voltage_Tx,
        "Inductor_Current_Tx":Inductor_Current_Tx,
        "Capacitor_Voltage_Tx":Capacitor_Voltage_Tx,
        "Capacitor_Current_Tx":Capacitor_Current_Tx,
    })