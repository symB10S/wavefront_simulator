Basic Usage
======================================

The functioning of the wavefront simulator is as follows:
----------------------------------------------------------
The :ref:`Generation module` populates the storage objects in the :ref:`Storage module`.
The :ref:`Plotting module` and :ref:`Interactive module` display the generated data.
The plots can then be compared to LTspice using the :ref:`Verification module`.

The most powerful tools are the interactive jupyter-notebook widgets, 
see the `examples.ipynb` to get an overview of the general functionality.
The specific capabilites of the simulator are best seen by overviewing the :ref:`Plotting module` functions

.. code-block::
    :caption: Simulating an interface and some basic plotting

    
        from wavefronts.generation import generate_interface_data
        import wavefronts.plotting as wp
        import matplotlib.pyplot as plt

        # simulate a resonator circuit 
        interface = generate_interface_data(L_time='8' , C_time='7',
                                             L_impedance = '500', C_impedance = '2')

        # plot interconnect time waveforms
        wp.make_time_interconnect_all(interface)

        # plot voltage reflection diagram
        fig_reflection_voltage, ax_reflection_voltage = plt.subplots()
        wp.plot_refelction_diagram(interface,ax_reflection_voltage,True) # True for plot voltage

        # display the commutative and multiplicative fanout of the capacitors voltage
        wp.make_commutative_merged_lines(interface,'interconnect','voltage capacitor')

        # plot the spatial distribution of voltage and current at 79.597s
        wp.make_spatial_voltage_and_current('79.597',interface)

        plt.show()


