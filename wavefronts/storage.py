from decimal import Decimal
import math
from collections import deque
import numpy as np
from dataclasses import dataclass
from wavefronts.misc import lcm_gcd_euclid, get_voltage_array, get_current_array, default_input_values ,handle_default_kwargs

class Input_Data :
    """The storage object for the input varibles of a interface simulation. Calculates all the associated variables required for the simulaitons. 
        Can be used to investigate network calcualted parameters based off input vairbles. 
        It also stores and calculates all the interaction functions of the interface.

        Is initialised using key-word arguments. All values with the provided keys are of type string. 
        This each input variable is converterted to a Decimal value to be used for precision calculations.
        The possible parameters to change and their defualt values are as follows, 
        
        :keyword L_impedance: Characteristic impedance of the inductor, assigned to self.Inductor_Impedance (default:'100')
        :type L_impedance: String
        :keyword L_time: The time delay of the inductor in seconds, assigned to self.Inductor_Time (default:'1')
        :type L_time: String
        :keyword L_length: The length of the inductor in meters, assigned to self.Inductor_Length (default:'1')
        :type L_length: String
        :keyword C_impedance: Characteristic impedance of the capacitor, assigned to self.Capacitor_Impedance (default:'1')
        :type C_impedance: String
        :keyword C_time: The time delay of the capacitor in seconds, assigned to self.Capacitor_Time (default:'1')
        :type C_time: String
        :keyword C_length: The length of the capacitor in meters, assigned to self.Capacitor_Length (default:'1')
        :type C_length: String
        :keyword V_source: The magnitude of the initial voltage excitation in volts, assigned to self.Voltage_Souce_Magnitude (default:'1')
        :type V_source: String
        :keyword number_periods: The number of periods as according to Lumped-Element LC-Osscilator solution. 
            Used to calculate the simulation stop time if provided. Overidden if 'Simulation_stop_time' is provided (default:'1')
        :type number_periods: String
        :keyword Load_impedance: The magnitude of the load resistance, if left inf the load is ignored and the interface takes form of an LC-Osscilator.
            If a value is provided the load is considered and the self.Is_Buck flag is set to True (default:'inf')
        :type Load_impedance: String
        :keyword Simulation_stop_time: The time to which the interface will be simulated. If provided it will overwrite the 'number_periods' simulation stop time calculation (default:'0')
        :type Simulation_stop_time: String
        :keyword show_about: Indicates information about the calcualted variabels must be printed (default:True)
        :type show_about: Boolean
        

        
        :**Stored and Calculated Parameters**:
            - **self.Number_Periods** (*Decimal*) - given or calcualted number of periods
            - **self.Simulation_Stop_Time** (*Decimal*) - given or calculated simulation stop time (s)
            - **self.Is_Buck** (*bool*) - if the load across the capacitor is considered or not 
            - **self.GCD** (*Decimal*) - The greatest common denomenator of the capacitor and inductor time delays.
            - **self.LCM** (*Decimal*) - The lowest common multiple of the capacitor and inductor time delays.
            - **self.Capacitor_LCM_Factor** (*int*) - The co-factor of the capacitor time delay required to make the LCM
            - **self.Inductor_LCM_Factor** (*int*) - The co-factor of the inductor time delay required to make the LCM
            - **self.is_Higher_Merging** (*bool*) - Indicates if multiplicative merging will occur for the given simulaiton stop time
            - **self.Number_of_Wavefronts** (*int*) - The total number of sending and returning wavefronts calculated
            - **self.Number_of_Layers** (*int*) - The total number of fanout-layers simulated
            - **self.Voltage_Souce_Magnitude** (*Decimal*) - the magnitude of the voltage excitation (V)
            - **self.Load_Resistance** (*Decimal*) - the magnitude of the load resistor (Ohm)
            
        :**Stored Inductor Parameters**:
            - **self.Inductor_Inductance_Per_Length** (*Decimal*) - the per length inductance of the inductor (H/m)
            - **self.Inductor_Capacitance_Per_Length** (*Decimal*) - the per length capacitance of the inductor (F/m)
            - **self.Inductor_Length** (*Decimal*) - the total length of the inductor (m)
            - **self.Inductor_Total_Inductance** (*Decimal*) - the total inductance of the inductor (H)
            - **self.Inductor_Total_Capacitance** (*Decimal*) - the total capacitance of the inductor (F)
            - **self.Inductor_Velocity** (*Decimal*) - the propagation velocity of the inductor (m/s)
            - **self.Inductor_Time** (*Decimal*) - the *one way* transit time of the inductor (s)
            - **self.Inductor_Impedance** (*Decimal*) - the characteristic impedance of the inductor (Ohm)
            
        :**Stored Capacitor Parameters**:
            - **self.Capacitor_Inductance_Per_Length** (*Decimal*) - the per length inductance of the capacitor (H/m)
            - **self.Capacitor_Capacitance_Per_Length** (*Decimal*) - the per length capacitance of the capacitor (F/m)
            - **self.Capacitor_Length** (*Decimal*) - the total length of the capacitor (m)
            - **self.Capacitor_Total_Inductance** (*Decimal*) - the total inductance of the capacitor (H)
            - **self.Capacitor_Total_Capacitance** (*Decimal*) - the total capacitance of the capacitor (F)
            - **self.Capacitor_Velocity** (*Decimal*) - the propagation velocity of the capacitor (m/s)
            - **self.Capacitor_Time** (*Decimal*) - the *one way* transit time of the capacitor (s)
            - **self.Capacitor_Impedance** (*Decimal*) - the characteristic impedance of the capacitor (Ohm)
            
        :**Dictionary for Spice simulation**:
            - **SPICE_input_values** (*dict*) - a list of altered input parameters compatible with :py:func:`verification.get_Spice_Arrays`.

        .. code-block:: python
            :caption: Example use of Input_Data
            
            from wavefronts.storage import Input_Data
            from wavefronts.generation import generate_commutative_data

            data_input = Input_Data(Simulation_stop_time = '100',L_impedance = '225')
            print(data_input.Simulation_Stop_Time) # prints '100'
            print(data_input.Capacitor_Impedance) # prints '1', assigned by default.
            
            # generate the output wavefronts from the created Input_Data object:
            data_output = generate_commutative_data(data_input)

        .. code-block:: python
            :caption: Advanced - change the termination function to make the circuit a RC charger
            
            from wavefronts.storage import Input_Data
            from wavefronts.generation import generate_interface_data
            from wavefronts.plotting import make_time_interconnect_all
            from copy import copy
            from decimal import Decimal 
            import builtins
            import matplotlib.pyplot as plt

            # generate a data input array
            data_input_1 = Input_Data(L_time = '1',C_time='7')
            # generate interface data storage object
            interface_1 = generate_interface_data(data_input_1)

            # create a copy of data_1
            data_input_2 = copy(data_input_1)

            # setup a new termination function
            R = Decimal('10')
            ZL = data_input_2.Inductor_Impedance
            def new_inductor_termination_func(V_arrive,I_arrive):
                V_out = I_arrive * (1/(1/R + 1/ZL)) - V_arrive*ZL/(R +ZL )
                I_out = -(V_out/ZL)
                
                return V_out, I_out

            # replace the old termination funciton
            builtins.setattr(data_input_2, 'termination_event_solver_inductor',new_inductor_termination_func)

            # generate interface data storage object unsing the altered data_input
            interface_2 = generate_interface_data(data_input_2)

            # plot data 
            fig_1, axes_1 = make_time_interconnect_all(interface_1)
            fig_2, axes_2 = make_time_interconnect_all(interface_2)

            fig_1.suptitle("LC - Osscilator")
            fig_2.suptitle("RC - Charger")
            axes_2['VL'].set_title("Resistor Votlage at Interconnect")
            axes_2['IL'].set_title("Resistor Current at Interconnect")

            plt.show()

    """
    def __init__(self,**provided_input_values):
        """Calcualtes the varibles to be stored about the interface based off provided input parameters
        """
        # an input variable dictionary with altered default values were relevant.
        self.input_values = default_input_values.copy()
        self.input_values = handle_default_kwargs(provided_input_values,self.input_values)
        
        # Make input dictionary compatible with SPICE simulation inputs
        self.SPICE_input_values = provided_input_values.copy()
        
        self.SPICE_input_values.pop('show_about',None)
        self.SPICE_input_values.pop('L_lenght',None)
        self.SPICE_input_values.pop('C_lenght',None)
        self.SPICE_input_values.pop('Load_impedance',None)
        
        
        # does the converter consider the load, or is it a LC osscilator.
        self.Is_Buck = True
        if self.input_values['Load_impedance'] == 'inf':
            self.Is_Buck = False
        
        # if simulation end time is specified, or is it calculated using "number_periods" input variable.
        self.Custom_stop_time = True
        if self.input_values['Simulation_stop_time'] == '0':
            self.Custom_stop_time = False
        
        # Extracting input variables provided for the inductor
        self.Inductor_Impedance = Decimal(self.input_values['L_impedance'])
        self.Inductor_Time = Decimal(self.input_values['L_time'])/2
        self.Inductor_Length = Decimal(self.input_values['L_length'])
        # Calcualte the variables around the inductor
        self.Inductor_Velocity = self.Inductor_Length/self.Inductor_Time
        self.Inductor_Inductance_Per_Length =  self.Inductor_Time*self.Inductor_Impedance
        self.Inductor_Capacitance_Per_Length =  self.Inductor_Time/self.Inductor_Impedance
        self.Inductor_Total_Inductance = self.Inductor_Inductance_Per_Length * self.Inductor_Length
        self.Inductor_Total_Capacitance = self.Inductor_Capacitance_Per_Length * self.Inductor_Length

        # Extracting input variables provided for the capacitor
        self.Capacitor_Impedance = Decimal(self.input_values['C_impedance'])
        self.Capacitor_Time = Decimal(self.input_values['C_time'])/2
        self.Capacitor_Length = Decimal(self.input_values['C_length'])
        # Calcualte the variables around the capacitor
        self.Capacitor_Velocity = self.Capacitor_Length/self.Capacitor_Time
        self.Capacitor_Inductance_Per_Length =  self.Capacitor_Time*self.Capacitor_Impedance
        self.Capacitor_Capacitance_Per_Length =  self.Capacitor_Time/self.Capacitor_Impedance
        self.Capacitor_Total_Inductance = self.Capacitor_Inductance_Per_Length * self.Capacitor_Length
        self.Capacitor_Total_Capacitance = self.Capacitor_Capacitance_Per_Length * self.Capacitor_Length

        # Extracting input variables provided about the circuit
        self.Voltage_Souce_Magnitude = Decimal(self.input_values['V_source'])
        self.Number_Periods = Decimal(self.input_values['number_periods'])
        self.Load_Impedance = Decimal(self.input_values['Load_impedance'])
        
        # Calculate simulation stop time
        self.Simulation_Stop_Time = Decimal()
        if(self.Custom_stop_time):
            self.Simulation_Stop_Time = Decimal(self.input_values['Simulation_stop_time'])
            self.Number_Periods = self.Simulation_Stop_Time/(Decimal('6.28318530718')*(Decimal.sqrt(self.Capacitor_Total_Capacitance*self.Inductor_Total_Inductance)))
        else:
            self.Simulation_Stop_Time = self.Number_Periods*Decimal('6.28318530718')*(Decimal.sqrt(self.Capacitor_Total_Capacitance*self.Inductor_Total_Inductance))
        
        # Determine the number of layers
        if (self.Capacitor_Time < self.Inductor_Time):
            self.Number_of_Layers = math.ceil(self.Simulation_Stop_Time/(self.Capacitor_Time*2))+1
        else:
            self.Number_of_Layers = math.ceil(self.Simulation_Stop_Time/(self.Inductor_Time*2))+1
        
        # Calculate the number of wavefronts that must be created
        self.Number_of_Wavefronts = 0
        for i in range(0,self.Number_of_Layers+1):
            self.Number_of_Wavefronts = self.Number_of_Wavefronts + 4*i
        
        # Calculate and store multiplicative realtionships between time delays
        Factor_Dict = lcm_gcd_euclid(self.Inductor_Time*2,self.Capacitor_Time*2)
        self.Inductor_LCM_Factor = int(Factor_Dict['KL'])
        self.Capacitor_LCM_Factor = int(Factor_Dict['KC'])
        self.GCD = Factor_Dict['GCD']
        self.LCM = Factor_Dict['LCM']
        
        # Determine if Multiplicative Merging will occur or not
        if(Factor_Dict['LCM'] > self.Simulation_Stop_Time):
            self.is_Higher_Merging = False
        else:
            self.is_Higher_Merging = True
        
        # Generate the associated response co-effcients for changes at the interface.
        if(self.Is_Buck):
            Load_Parallel_Inductor = 1/(1/self.Load_Impedance + 1/self.Inductor_Impedance)
            Load_Parallel_Capacitor = 1/(1/self.Load_Impedance + 1/self.Capacitor_Impedance)

            self.Inductor_Voltage_VL_coeff  = self.Inductor_Impedance/( self.Inductor_Impedance + Load_Parallel_Capacitor )
            self.Inductor_Voltage_VC_coeff  = Load_Parallel_Inductor/( self.Capacitor_Impedance + Load_Parallel_Inductor )
            self.Inductor_Voltage_IL_coeff  = self.Capacitor_Impedance * self.Inductor_Impedance * self.Load_Impedance /(self.Load_Impedance*self.Inductor_Impedance + self.Load_Impedance*self.Capacitor_Impedance + self.Inductor_Impedance * self.Capacitor_Impedance)
            self.Inductor_Voltage_IC_coeff  = self.Inductor_Voltage_IL_coeff
            self.Inductor_Voltage_VS_coeff  = self.Inductor_Impedance / ( self.Inductor_Impedance + Load_Parallel_Capacitor )

            self.Inductor_Current_VL_coeff  = self.Inductor_Voltage_VL_coeff / self.Inductor_Impedance
            self.Inductor_Current_VC_coeff  = self.Inductor_Voltage_VC_coeff / self.Inductor_Impedance
            self.Inductor_Current_IL_coeff  = self.Inductor_Voltage_IL_coeff / self.Inductor_Impedance
            self.Inductor_Current_IC_coeff  = self.Inductor_Voltage_IC_coeff / self.Inductor_Impedance
            self.Inductor_Current_VS_coeff  = self.Inductor_Voltage_VS_coeff / self.Inductor_Impedance

            self.Capacitor_Voltage_VC_coeff  = self.Capacitor_Impedance/( self.Capacitor_Impedance + Load_Parallel_Inductor )
            self.Capacitor_Voltage_VL_coeff  = Load_Parallel_Capacitor/( self.Inductor_Impedance + Load_Parallel_Capacitor )
            self.Capacitor_Voltage_IC_coeff  = self.Capacitor_Impedance * self.Inductor_Impedance * self.Load_Impedance /(self.Load_Impedance*self.Inductor_Impedance + self.Load_Impedance*self.Capacitor_Impedance + self.Inductor_Impedance * self.Capacitor_Impedance)
            self.Capacitor_Voltage_IL_coeff  = self.Capacitor_Voltage_IC_coeff
            self.Capacitor_Voltage_VS_coeff  = Load_Parallel_Capacitor / ( self.Inductor_Impedance + Load_Parallel_Capacitor )

            self.Capacitor_Current_VC_coeff  = self.Capacitor_Voltage_VC_coeff / self.Capacitor_Impedance
            self.Capacitor_Current_VL_coeff  = self.Capacitor_Voltage_VL_coeff / self.Capacitor_Impedance
            self.Capacitor_Current_IC_coeff  = self.Capacitor_Voltage_IC_coeff / self.Capacitor_Impedance
            self.Capacitor_Current_IL_coeff  = self.Capacitor_Voltage_IL_coeff / self.Capacitor_Impedance
            self.Capacitor_Current_VS_coeff  = self.Capacitor_Voltage_VS_coeff / self.Capacitor_Impedance
            
            self.Initial_Inductor_Current = self.Voltage_Souce_Magnitude/(self.Inductor_Impedance + Load_Parallel_Capacitor)
            self.Initial_Inductor_Voltage = self.Initial_Inductor_Current * self.Inductor_Impedance
            
            self.Initial_Capacitor_Voltage = self.Initial_Inductor_Current * Load_Parallel_Capacitor
            self.Initial_Capacitor_Current = self.Initial_Capacitor_Voltage/self.Capacitor_Impedance
            
        else:
            Load_Parallel_Inductor = self.Inductor_Impedance
            Load_Parallel_Capacitor = self.Capacitor_Impedance

            self.Inductor_Voltage_VL_coeff  = self.Inductor_Impedance/( self.Inductor_Impedance + self.Capacitor_Impedance )
            self.Inductor_Voltage_VC_coeff  = self.Inductor_Impedance/( self.Inductor_Impedance + self.Capacitor_Impedance )
            self.Inductor_Voltage_IL_coeff  = self.Capacitor_Impedance * self.Inductor_Impedance /(self.Inductor_Impedance + self.Capacitor_Impedance )
            self.Inductor_Voltage_IC_coeff  = self.Inductor_Voltage_IL_coeff
            self.Inductor_Voltage_VS_coeff  = self.Inductor_Impedance / ( self.Inductor_Impedance + self.Capacitor_Impedance )

            self.Inductor_Current_VL_coeff  = self.Inductor_Voltage_VL_coeff / self.Inductor_Impedance
            self.Inductor_Current_VC_coeff  = self.Inductor_Voltage_VC_coeff / self.Inductor_Impedance
            self.Inductor_Current_IL_coeff  = self.Inductor_Voltage_IL_coeff / self.Inductor_Impedance
            self.Inductor_Current_IC_coeff  = self.Inductor_Voltage_IC_coeff / self.Inductor_Impedance
            self.Inductor_Current_VS_coeff  = self.Inductor_Voltage_VS_coeff / self.Inductor_Impedance

            self.Capacitor_Voltage_VC_coeff  = self.Capacitor_Impedance/( self.Capacitor_Impedance + self.Inductor_Impedance )
            self.Capacitor_Voltage_VL_coeff  = self.Capacitor_Impedance/( self.Inductor_Impedance + self.Capacitor_Impedance )
            self.Capacitor_Voltage_IC_coeff  = self.Capacitor_Impedance * self.Inductor_Impedance  /(self.Inductor_Impedance + self.Capacitor_Impedance )
            self.Capacitor_Voltage_IL_coeff  = self.Capacitor_Voltage_IC_coeff
            self.Capacitor_Voltage_VS_coeff  = self.Capacitor_Impedance / ( self.Inductor_Impedance + self.Capacitor_Impedance )

            self.Capacitor_Current_VC_coeff  = self.Capacitor_Voltage_VC_coeff / self.Capacitor_Impedance
            self.Capacitor_Current_VL_coeff  = self.Capacitor_Voltage_VL_coeff / self.Capacitor_Impedance
            self.Capacitor_Current_IC_coeff  = self.Capacitor_Voltage_IC_coeff / self.Capacitor_Impedance
            self.Capacitor_Current_IL_coeff  = self.Capacitor_Voltage_IL_coeff / self.Capacitor_Impedance
            self.Capacitor_Current_VS_coeff  = self.Capacitor_Voltage_VS_coeff / self.Capacitor_Impedance

            self.Initial_Inductor_Current = self.Voltage_Souce_Magnitude/(self.Inductor_Impedance + self.Capacitor_Impedance)
            self.Initial_Inductor_Voltage = self.Initial_Inductor_Current * self.Inductor_Impedance
            
            self.Initial_Capacitor_Current = self.Initial_Inductor_Current
            self.Initial_Capacitor_Voltage = self.Initial_Capacitor_Current* self.Capacitor_Impedance
        
        # Show information about network
        if(self.input_values['show_about']):
            self.about()
    
    def circuit_solver_inductor_voltage(self,VL: Decimal,IL: Decimal,VC: Decimal,IC: Decimal):
        """Generates the voltage response of the inductor to wavefront distrubances. Solves by means of the wavefront equivalent circuit.

        :param VL: the magnitude of the voltage disturbance from the inductor
        :type VL: Decimal
        :param IL: the magnitude of the current disturbance from the inductor
        :type IL: Decimal
        :param VC: the magnitude of the voltage disturbance from the capacitor
        :type VC: Decimal
        :param IC: the magnitude of the current disturbance from the capacitor
        :type IC: Decimal
        :return: the the magnitude of the voltage response of the inductor to the disturbance
        :rtype: Decimal
        """
        return -VL * self.Inductor_Voltage_VL_coeff - VC * self.Inductor_Voltage_VC_coeff - IL * self.Inductor_Voltage_IL_coeff + IC * self.Inductor_Voltage_IC_coeff 

    def circuit_solver_inductor_current(self,VL: Decimal,IL: Decimal,VC: Decimal,IC: Decimal):
        """Generates the current response of the inductor to wavefront distrubances. Solves by means of the wavefront equivalent circuit.

        :param VL: the magnitude of the voltage disturbance from the inductor
        :type VL: Decimal
        :param IL: the magnitude of the current disturbance from the inductor
        :type IL: Decimal
        :param VC: the magnitude of the voltage disturbance from the capacitor
        :type VC: Decimal
        :param IC: the magnitude of the current disturbance from the capacitor
        :type IC: Decimal
        :return: the the magnitude of the current response of the inductor to the disturbance
        :rtype: Decimal
        """
        return -VL * self.Inductor_Current_VL_coeff - VC * self.Inductor_Current_VC_coeff - IL * self.Inductor_Current_IL_coeff + IC * self.Inductor_Current_IC_coeff 

    def circuit_solver_inductor_source_voltage(self,VS: Decimal):
        """The magnitude of the voltage response of the inductor to a voltage source excitation.

        :param VS: magnitude of soure voltage excitation. 
        :type VS: Decimal
        :return: the the magnitude of the voltage response of the inductor to the disturbance
        :rtype: Decimal
        """
        return VS * self.Inductor_Voltage_VS_coeff

    def circuit_solver_inductor_source_current(self,VS: Decimal):
        """The magnitude of the current response of the inductor to a voltage source excitation.

        :param VS: magnitude of soure voltage excitation. 
        :type VS: Decimal
        :return: the the magnitude of the current response of the inductor to the disturbance
        :rtype: Decimal
        """
        return VS * self.Inductor_Current_VS_coeff

    def circuit_solver_capacitor_voltage(self,VL: Decimal,IL: Decimal,VC: Decimal,IC: Decimal):
        """Generates the voltage response of the capacitor to wavefront distrubances. Solves by means of the wavefront equivalent circuit.

        :param VL: the magnitude of the voltage disturbance from the inductor
        :type VL: Decimal
        :param IL: the magnitude of the current disturbance from the inductor
        :type IL: Decimal
        :param VC: the magnitude of the voltage disturbance from the capacitor
        :type VC: Decimal
        :param IC: the magnitude of the current disturbance from the capacitor
        :type IC: Decimal
        :return: the the magnitude of the voltage response of the capacitor to the disturbance
        :rtype: Decimal
        """
        return -VC * self.Capacitor_Voltage_VC_coeff - VL * self.Capacitor_Voltage_VL_coeff - IC * self.Capacitor_Voltage_IC_coeff + IL * self.Capacitor_Voltage_IL_coeff 

    def circuit_solver_capacitor_current(self,VL: Decimal,IL: Decimal,VC: Decimal,IC: Decimal):
        """Generates the current response of the capacitor to wavefront distrubances. Solves by means of the wavefront equivalent circuit.

        :param VL: the magnitude of the voltage disturbance from the inductor
        :type VL: Decimal
        :param IL: the magnitude of the current disturbance from the inductor
        :type IL: Decimal
        :param VC: the magnitude of the voltage disturbance from the capacitor
        :type VC: Decimal
        :param IC: the magnitude of the current disturbance from the capacitor
        :type IC: Decimal
        :return: the the magnitude of the current response of the capacitor to the disturbance
        :rtype: Decimal
        """
        return -VC * self.Capacitor_Current_VC_coeff - VL * self.Capacitor_Current_VL_coeff - IC * self.Capacitor_Current_IC_coeff + IL * self.Capacitor_Current_IL_coeff 

    def circuit_solver_capacitor_source_voltage(self,VS: Decimal):
        """The magnitude of the voltage response of the capacitor to a voltage source excitation.

        :param VS: magnitude of soure voltage excitation. 
        :type VS: Decimal
        :return: the the magnitude of the voltage response of the capacitor to the disturbance
        :rtype: Decimal
        """
        return VS * self.Capacitor_Voltage_VS_coeff

    def circuit_solver_capacitor_source_current(self,VS: Decimal):
        """The magnitude of the current response of the capacitor to a voltage source excitation.

        :param VS: magnitude of soure voltage excitation. 
        :type VS: Decimal
        :return: the the magnitude of the current response of the capacitor to the disturbance
        :rtype: Decimal
        """
        return VS * self.Capacitor_Current_VS_coeff
    
    def self_reflection_event_solver_inductor(self,Wavefront_Parent_voltage: Decimal,Wavefront_Parent_current: Decimal):
        """Calculates the voltage and current magnitude of a produced inductive wavefront due to an inductive self-reflection event. 
        A self-reflection event is when a wavefron form a transmission line arrives at the interface and is reflected back into itself.
        The Parent wavefront's parameters are passed to this function, the self reflected child wavefront magnitudes are calculated.
        
        :param Wavefront_Parent_voltage: voltage of the parent wavefront arriving at the interface
        :type Wavefront_Parent_voltage: Decimal
        :param Wavefront_Parent_current: current of the parent wavefront arriving at the interface
        :type Wavefront_Parent_current: Decimal
        :return: (voltage, current ) of child wavefront
        :rtype: Tuple
        """
        magnitude_voltage = self.circuit_solver_inductor_voltage( Wavefront_Parent_voltage, Wavefront_Parent_current, 0, 0)
        magnitude_current = self.circuit_solver_inductor_current( Wavefront_Parent_voltage, Wavefront_Parent_current, 0, 0)
        
        return magnitude_voltage, magnitude_current
    
    def exitation_event_solver_inductor(self,Wavefront_Parent_voltage : Decimal,Wavefront_Parent_current : Decimal):
        """The voltage and current calcualtion of the inducitve wavefront produced due to a source excitation event.

        :param Wavefront_Parent_voltage: voltage of the source excitation wavefront 
        :type Wavefront_Parent_voltage: Decimal
        :param Wavefront_Parent_current: current of the source excitation wavefront 
        :type Wavefront_Parent_current: Decimal
        :return: (voltage, current) of the produced inductive wavefront 
        :rtype: Tuple (Decimal, Decimal)
        """
        magnitude_voltage = self.circuit_solver_inductor_source_voltage(Wavefront_Parent_voltage)
        magnitude_current = self.circuit_solver_inductor_source_current(Wavefront_Parent_voltage)
        
        return magnitude_voltage, magnitude_current

    def transmission_event_solver_inductor(self,Wavefront_Parent_voltage : Decimal,Wavefront_Parent_current : Decimal):
        """The voltage and current calculation of the inductive wavefront produced due to a capacitve wavefront arriving at the interface.

        :param Wavefront_Parent_voltage: voltage of the incident capacitve wavefront
        :type Wavefront_Parent_voltage: Decimal
        :param Wavefront_Parent_current: current of the incident capacitve wavefront
        :type Wavefront_Parent_current: Decimal
        :return: (voltage, current) of the produced inductive wavefront 
        :rtype: Tuple (Decimal, Decimal)
        """
        magnitude_voltage = self.circuit_solver_inductor_voltage(0, 0, Wavefront_Parent_voltage, Wavefront_Parent_current)
        magnitude_current = self.circuit_solver_inductor_current(0, 0, Wavefront_Parent_voltage, Wavefront_Parent_current)
        
        return magnitude_voltage, magnitude_current

    def termination_event_solver_inductor(self,Arriving_Voltage: Decimal,Arriving_Current: Decimal):
        """The voltage and current calcutation of the re-reflected wavefront produced when an inductive wavefront reaches its termination.

        :param Arriving_Voltage: voltage of the wavefront arriving at the inductor termination
        :type Arriving_Voltage: Decimal
        :param Arriving_Current: current of the wavefront arriving at the inductor termination
        :type Arriving_Current: Decimal
        :return: (voltage, current) of the re-reflected inductive wavefront
        :rtype: Tuple (Decimal, Decimal)
        """
        return -Arriving_Voltage, Arriving_Current

    def self_reflection_event_solver_capacitor(self,Wavefront_Parent_voltage: Decimal,Wavefront_Parent_current: Decimal):
        """Calculates the voltage and current magnitude of a produced capcaitve wavefront due to a capacitve self-reflection event. 
        A self-reflection event is when a wavefront form a transmission line arrives at the interface and is reflected back into itself.
        The Parent wavefront's parameters are passed to this function, the self reflected child wavefront magnitudes are calculated.

        :param Wavefront_Parent_voltage: voltage of the parent wavefront arriving at the interface
        :type Wavefront_Parent_voltage: Decimal
        :param Wavefront_Parent_current: current of the parent wavefront arriving at the interface
        :type Wavefront_Parent_current: Decimal
        :return: (voltage, current ) of child wavefront
        :rtype: Tuple
        """
        magnitude_voltage = self.circuit_solver_capacitor_voltage( 0,0,Wavefront_Parent_voltage, Wavefront_Parent_current)
        magnitude_current = self.circuit_solver_capacitor_current( 0,0,Wavefront_Parent_voltage, Wavefront_Parent_current)
        
        return magnitude_voltage, magnitude_current
    
    def exitation_event_solver_capacitor(self,Wavefront_Parent_voltage : Decimal,Wavefront_Parent_current : Decimal):
        """The voltage and current calcualtion of the capacitive wavefront produced due to a source excitation event.

        :param Wavefront_Parent_voltage: voltage of the source excitation wavefront 
        :type Wavefront_Parent_voltage: Decimal
        :param Wavefront_Parent_current: current of the source excitation wavefront 
        :type Wavefront_Parent_current: Decimal
        :return: (voltage, current) of the produced capacitive wavefront 
        :rtype: Tuple (Decimal, Decimal)
        """     
        magnitude_voltage = self.circuit_solver_capacitor_source_voltage(Wavefront_Parent_voltage)
        magnitude_current = self.circuit_solver_capacitor_source_current(Wavefront_Parent_voltage)
        
        return magnitude_voltage, magnitude_current

    def transmission_event_solver_capacitor(self,Wavefront_Parent_voltage : Decimal,Wavefront_Parent_current : Decimal):
        """The voltage and current calculation of the capacitve wavefront produced due to a inductive wavefront arriving at the interface.

        :param Wavefront_Parent_voltage: voltage of the incident inductive wavefront
        :type Wavefront_Parent_voltage: Decimal
        :param Wavefront_Parent_current: current of the incident inductive wavefront
        :type Wavefront_Parent_current: Decimal
        :return: (voltage, current) of the produced capacitve wavefront 
        :rtype: Tuple (Decimal, Decimal)
        """    
        magnitude_voltage = self.circuit_solver_capacitor_voltage(Wavefront_Parent_voltage, Wavefront_Parent_current,0,0)
        magnitude_current = self.circuit_solver_capacitor_current(Wavefront_Parent_voltage, Wavefront_Parent_current,0,0)
        
        return magnitude_voltage, magnitude_current

    def termination_event_solver_capacitor(self,Arriving_Voltage: Decimal,Arriving_Current: Decimal):
        """The voltage and current calcutation of the re-reflected wavefront produced when a capacitve wavefront reaches its termination.

        :param Arriving_Voltage: voltage of the wavefront arriving at the capcitor termination
        :type Arriving_Voltage: Decimal
        :param Arriving_Current: current of the wavefront arriving at the capcitor termination
        :type Arriving_Current: Decimal
        :return: (voltage, current) of the re-reflected capacitve wavefront
        :rtype: Tuple (Decimal, Decimal)
        """
        return Arriving_Voltage, -Arriving_Current
    
    def about(self):
        """Prints out information input varibles and associated calculated variables.
        """
        print(f"\nInformation about this network : \n")

        print(f"\n- The Inductor -")
        print(f"{'Inductor Inductance Per Length :':<40}{self.Inductor_Inductance_Per_Length}")
        print(f"{'Inductor Capacitance Per Length :':<40}{self.Inductor_Capacitance_Per_Length}")
        print(f"{'Inductor Length :':<40}{self.Inductor_Length}")
        print(f"{'Inductor Total Inductance :':<40}{self.Inductor_Total_Inductance}")
        print(f"{'Inductor Total Capacitance :':<40}{self.Inductor_Total_Capacitance}")
        print(f"{'Inductor Velocity :':<40}{self.Inductor_Velocity}")
        print(f"{'Inductor One Way Time Delay :':<40}{self.Inductor_Time}")
        print(f"{'Inductor Impedance :':<40}{self.Inductor_Impedance}")
        
        print(f"\n- The Capacitor -")
        print(f"{'Capacitor Inductance Per Length :':<40}{self.Capacitor_Inductance_Per_Length}")
        print(f"{'Capacitor Capacitance Per Length :':<40}{self.Capacitor_Capacitance_Per_Length}")
        print(f"{'Capacitor Length :':<40}{self.Capacitor_Length}")
        print(f"{'Capacitor Total Inductance :':<40}{self.Capacitor_Total_Inductance}")
        print(f"{'Capacitor Total Capacitance :':<40}{self.Capacitor_Total_Capacitance}")
        print(f"{'Capacitor Velocity :':<40}{self.Capacitor_Velocity}")
        print(f"{'Capacitor One Way Time Delay :':<40}{self.Capacitor_Time}")
        print(f"{'Capacitor Impedance :':<40}{self.Capacitor_Impedance}")
        
        print(f"\n- The Time -")
        print(f"{'Number Periods :':<40}{self.Number_Periods}")
        print(f"{'Simulation Stop Time :':<40}{self.Simulation_Stop_Time}")
        print(f"{'Number of Wavefronts :':<40}{self.Number_of_Wavefronts}")
        print(f"{'Number of Layers :':<40}{self.Number_of_Layers}")
        print(f"{'Inductor Return Time Delay :':<40}{2*self.Inductor_Time}")
        print(f"{'Inductor LCM Factor :':<40}{self.Inductor_LCM_Factor}")
        print(f"{'Capacitor Return Time Delay :':<40}{2*self.Capacitor_Time}")
        print(f"{'Capacitor LCM Factor :':<40}{self.Capacitor_LCM_Factor}")
        print(f"{'LCM :':<40}{self.LCM}")
        print(f"{'GCD :':<40}{self.GCD}")
        print(f"{'Higher Merging? :':<40}{self.is_Higher_Merging}")
        

        print(f"\n- The Circuit -")
        print(f"{'Votage Source Magnitude :':<40}{self.Voltage_Souce_Magnitude}")
        print(f"{'Buck Converter :':<40}{self.Is_Buck}")
        print(f"{'Load Resistance :':<40}{self.Load_Impedance}")

class Wavefront:
    """Base wavefront class. Assings basic wavefront parameters, all of Decimal type initialised to zero.
    """
    def __init__(self):

        self.position_start = Decimal('0')
        self.position_end = Decimal('0')

        self.time_start = Decimal('0')
        self.time_end = Decimal('0')

        self.magnitude_voltage = Decimal('0')
        self.magnitude_current = Decimal('0')
        
    def __add__(self, Wavefront_add ):
        """superimposes two wavefronts and adds their magnitudes.

        :param Wavefront_add: wavefront to be added with self
        :type Wavefront_add: Wavefront
        :return: self wavefront with magnitudes added
        :rtype: Wavefront (same as self)
        """
        
        if(Wavefront_add == 0):
            pass
        elif(self.time_start == 0 and self.time_end == 0 ):
            pass
        elif(Wavefront_add.time_start == 0 and Wavefront_add.time_end == 0 ):
            pass
        elif(Wavefront_add.time_start == self.time_start and Wavefront_add.time_end == self.time_end ):
            self.magnitude_voltage += Wavefront_add.magnitude_voltage
            self.magnitude_current += Wavefront_add.magnitude_current
        else:
            raise Exception("Wavefronts cannot be added")
        
        return self
        
    def __radd__(self, Wavefront_add ):
        """add operation but for RHS. same as __add__
        """
        return self.__add__(Wavefront_add)
        
    def about(self) :
        """Displays information anout the wavefront
        """
        print("\nSome Information about a wavefront")
        print(f"{'Type :':<35}{type(self)}")
        print(f"{'Poisiton Start :':<35}{self.position_start}")
        print(f"{'Poisiton End :':<35}{self.position_end}")
        print(f"{'Time Start :':<35}{self.time_start}")
        print(f"{'Time End :':<35}{self.time_end}")
        print(f"{'Voltage Magnitude :':<35}{self.magnitude_voltage}")
        print(f"{'Current Magnitude :':<35}{self.magnitude_current}")
    
    def position_at_time(self,time):
        """Generates the position of wavefront at time-equirey. Returns False if no intercept.

        :param time: time enquirey
        :type time: Decimal or str
        :return: position attime enquiret
        :rtype: Decimal
        """
        t = Decimal(time)

        if self.time_start <= t <= self.time_end:
            if self.position_start == 0 :
                return (t-self.time_start)*self.velocity
            else:
                return self.length - (t-self.time_start)*self.velocity
        else:
            return False

class Wavefront_Source(Wavefront):
    """Class representing switiching wavefronts of the voltage source. 
    In this release wavefonts switching is not supported, 
    this class used to initiated wavefronts at t=0 only. 
    """

    def __init__(self, Data_input: Input_Data,time_start, magnitude = 1  ):
        """intialised the wavefront.

        :param magnitude: Source magnitude
        :type magnitude: str or Decimal
        :param time_start: start of pulse
        :type time_start: str or Decimal
        """
        super().__init__()
        self.Data_input = Data_input

        self.time_start = Decimal(time_start)
        self.time_end = self.time_start
        self.magnitude_voltage = Decimal(magnitude)

    def generate_and_store(self, storage_Away : deque):
        """triggers the creation of the inital wavefronts in the inductor and capacitor. 
        Stores them in the storage. Order is important, inductive first followed by capacitive. 

        :param storage: Storage array for away waves
        :type storage: deque
        """
        storage_Away.append(Wavefront_Inductive(self.Data_input,self,False))
        storage_Away.append(Wavefront_Capacitive(self.Data_input,self,False))
   
class Wavefront_Kintetic( Wavefront ):
    """An parent class for Inductive and Capacitve wavefronts. Contains the logic for determining how wavefronts respond to particualr events.
    """
    def setup_wavefront(self,Wavefront_Parent : Wavefront, is_self_reflection : bool):
        """handles how the voltage and current of a newly produced wavefront must be assigned. 

        :param Wavefront_Parent: The Parent wavefront that is producing this wavefronts
        :type Wavefront_Parent: Wavefront
        :param is_self_reflection: if the parent wavefront is in the same wavefront as the child. Limits the need for an isinstance check.
        :type is_self_reflection: bool
        """
        # key:
        # | = interface,  X = termination, --> = this wavefront ,(Vs) = Source excitation
        
        #               waves travelling to termination : | --> X
        #    | --> X  = this wavefront travelling to termination, parent from same tx - self-reflection
        # (v)| --> X  = this wavefront travelling to termination, parent is voltage source - source excitation
        #  ->| --> X  = this wavefront travelling to termination, parent from other tx - transmission
        
        #               waves returning to interface : | <-- X
        #    | <-- X  = this wavefront returning to inerface, parent from same - re-reflection
        
        # waves travelling to termination : | --> X
        if self.position_start == 0:

            # was the parent wavefront in the same transmission line?
            if is_self_reflection: # Yes, | --> X , self-reflection

                self.magnitude_voltage,self.magnitude_current  = self.self_reflection_event_solver(Wavefront_Parent.magnitude_voltage,Wavefront_Parent.magnitude_current)

            # was the parent wavefront an excitation event ? 
            elif isinstance(Wavefront_Parent, Wavefront_Source) : # (v)| --> X , source excitation

                self.magnitude_voltage,self.magnitude_current = self.exitation_event_solver(Wavefront_Parent.magnitude_voltage,Wavefront_Parent.magnitude_current)

            else: # A transmitted wave at source side  ->| --> X, transmission

                self.magnitude_voltage,self.magnitude_current = self.transmission_event_solver(Wavefront_Parent.magnitude_voltage,Wavefront_Parent.magnitude_current)
        
        # waves returning to interface : | <-- X , re-reflection
        else: 
            self.magnitude_voltage,self.magnitude_current = self.termination_event_solver(Wavefront_Parent.magnitude_voltage,Wavefront_Parent.magnitude_current)
    
    def merge(self, Wavefront_Other : Wavefront):
        """superimposes two wavefronts by altering the voltage and current magnitudes of this wavefront.

        :param Wavefront_Other: Partner Wavefront to be merging 
        :type Wavefront_Other: Wavefront
        """
        self.magnitude_voltage = self.magnitude_voltage + Wavefront_Other.magnitude_voltage
        self.magnitude_current = self.magnitude_current + Wavefront_Other.magnitude_current

    def self_reflection_event_solver(self,Wavefront_Parent_voltage,Wavefront_Parent_current):
        pass
    
    def exitation_event_solver(self,Wavefront_Parent_voltage,Wavefront_Parent_current):
        pass
    
    def transmission_event_solver(self,Wavefront_Parent_voltage,Wavefront_Parent_current):
        pass
    
    def termination_event_solver(self,Wavefront_Parent_voltage,Wavefront_Parent_current):
        pass

class Wavefront_Capacitive( Wavefront_Kintetic ):
    """
    A wavefront travelling in the capacitor. Follows the "wavefronts create wavefronts" paradigm. 
    """

    def __init__(self, input_data : Input_Data, Wavefront_Parent : Wavefront, is_self_reflection : bool):
        """
        Generates a capacitive wavefront based off the information of the parent wavefront. 

        :param input_data: the input paramaters of the interface being investigated.
        :type input_data: Input_Data
        :param Wavefront_Parent: the parent wavefront producing this wavefront
        :type Wavefront_Parent: Wavefront
        :param is_self_reflection: if the parent wavefront is in the same wavefront as the child. Limits the need for an isinstance check.
        :type is_self_reflection: bool
        """
        self.input_data = input_data
        
        self.position_start = Wavefront_Parent.position_end
        
        if self.position_start == 0:
            self.position_end = self.input_data.Capacitor_Length
        else:
            self.position_end = 0
        
        self.time_start = Wavefront_Parent.time_end
        self.time_end = self.time_start + self.input_data.Capacitor_Time
        
        self.velocity = self.input_data.Capacitor_Velocity
        self.length = self.input_data.Capacitor_Length
        
        self.magnitude_voltage = 0
        self.magnitude_current = 0
        
        self.self_reflection_event_solver = self.input_data.self_reflection_event_solver_capacitor
        self.exitation_event_solver = self.input_data.exitation_event_solver_capacitor
        self.transmission_event_solver = self.input_data.transmission_event_solver_capacitor
        self.termination_event_solver = self.input_data.termination_event_solver_capacitor

        self.setup_wavefront(Wavefront_Parent,is_self_reflection)

    def generate_and_store(self, storage : deque):
        """Generates and stores wavefronts the childern wavefront in a que to be processed

        :param storage: The deque of wavefronts that are actively being processed
        :type storage: deque
        """
        if self.position_end == 0:
            storage.append(Wavefront_Inductive(self.input_data,self,False))
            storage.append(Wavefront_Capacitive(self.input_data,self,True))
        else:
            storage.append(Wavefront_Capacitive(self.input_data,self,True))
    
    def generate_and_return(self):
        """Generates the children wavefront/s of this wavefront without directly storing them. 

        :return: children wavefront/s
        :rtype: Tuple (Wavefront_Inductive, Wavefront_Capacitive) or Wavefront_Capacitive
        """
        if self.position_end == 0:
            return Wavefront_Inductive(self.input_data,self,False), Wavefront_Capacitive(self.input_data,self,True)
        else :
            return Wavefront_Capacitive(self.input_data,self,self.input_data,True)

class Wavefront_Inductive( Wavefront_Kintetic ):
    """
    A wavefront travelling in the inductor. Follows the "wavefronts create wavefronts" paradigm. 
    """
    def __init__(self, input_data : Input_Data, Wavefront_Parent : Wavefront, is_self_reflection : bool):
        """
        Generates a inductive wavefront based off the information of the parent wavefront. 

        :param input_data: the input paramaters of the interface being investigated.
        :type input_data: Input_Data
        :param Wavefront_Parent: the parent wavefront producing this wavefront
        :type Wavefront_Parent: Wavefront
        :param is_self_reflection: if the parent wavefront is in the same wavefront as the child. Limits the need for an isinstance check.
        :type is_self_reflection: bool
        """
        
        self.input_data = input_data
        
        self.position_start = Wavefront_Parent.position_end
        
        if self.position_start == 0:
            self.position_end = self.input_data.Inductor_Length
        else:
            self.position_end = 0
        
        self.time_start = Wavefront_Parent.time_end
        self.time_end = self.time_start + self.input_data.Inductor_Time
        
        self.velocity = self.input_data.Inductor_Velocity
        self.length = self.input_data.Inductor_Length
        
        self.magnitude_voltage = 0
        self.magnitude_current = 0
        
        self.self_reflection_event_solver = self.input_data.self_reflection_event_solver_inductor
        self.exitation_event_solver = self.input_data.exitation_event_solver_inductor
        self.transmission_event_solver = self.input_data.transmission_event_solver_inductor
        self.termination_event_solver = self.input_data.termination_event_solver_inductor

        self.setup_wavefront(Wavefront_Parent,is_self_reflection)

    def generate_and_store(self, storage):
        """Generates and stores wavefronts the childern wavefront in a que to be processed

        :param storage: The deque of wavefronts that are actively being processed
        :type storage: deque
        """
        if self.position_end == 0:
            storage.append(Wavefront_Inductive(self.input_data,self,True))
            storage.append(Wavefront_Capacitive(self.input_data,self,False))
        else :
            storage.append(Wavefront_Inductive(self.input_data,self,True))

    def generate_and_return(self):
        """Generates the children wavefront/s of this wavefront without directly storing them. 

        :return: children wavefront/s
        :rtype: Tuple (Wavefront_Inductive, Wavefront_Capacitive) or Wavefront_Inductive
        """
        if self.position_end == 0:
            return Wavefront_Inductive(self.input_data,self,True), Wavefront_Capacitive(self.input_data,self,False)
        else :
            return Wavefront_Inductive(self.input_data,self,True)

@dataclass
class Output_Data:
    """Stores data of various types of fanout diagrams after simulation. 
        Stores information for commutatively merged fanouts, as well as, multipicatively merged fanouts.
        
        Fanout diagrams take form of 2D numpy arrays of format Array[L,C] where L is the inductive event number and C the capacitve event number.
        There are a total of 9 arrays stored, one for the arrival time of each grid node, 
        four for the current and voltage at the interconncet for the capacitor and inductor, 
        and another four for the sending and returning wavefronts of the capacitor and inductor. 
        
        :param Time: 2D numpy array of the return times of grid nodes
        :type Time: np.ndarray[Decimal]
        :param Voltage_Interconnect_Inductor: 2D numpy array of the interconnect voltage change of the inductor at a grid node
        :type Voltage_Interconnect_Inductor: np.ndarray[Decimal]
        :param Current_Interconnect_Inductor: 2D numpy array of the interconnect current change of the inductor at a grid node
        :type Current_Interconnect_Inductor: np.ndarray[Decimal]
        :param Voltage_Interconnect_Capacitor: 2D numpy array of the interconnect voltage change of the capacitor at a grid node
        :type Voltage_Interconnect_Capacitor: np.ndarray[Decimal]
        :param Current_Interconnect_Capacitor: 2D numpy array of the interconnect current change of the capacitor at a grid node
        :type Current_Interconnect_Capacitor: np.ndarray[Decimal]
        :param Wavefronts_Sending_Inductor: 2D numpy array of the wavefronts sent into the inductor at grid nodes
        :type Wavefronts_Sending_Inductor: np.ndarray[Wavefront_Inductive]
        :param Wavefronts_Sending_Capacitor: 2D numpy array of the wavefronts sent into the capacitor at grid nodes
        :type Wavefronts_Sending_Capacitor: np.ndarray[Wavefront_Capacitive]
        :param Wavefronts_Returning_Inductor: 2D numpy array of the wavefronts returning from the inductor at grid nodes
        :type Wavefronts_Returning_Inductor: np.ndarray[Wavefront_Inductive]
        :param Wavefronts_Returning_Capacitor: 2D numpy array of the wavefronts returning from the capacitor at grid nodes
        :type Wavefronts_Returning_Capacitor: np.ndarray[Wavefront_Capacitive]
        :param has_merged: indicates if the data stored has been multiplicatively merged or not.
        :type has_merged: bool
        
        .. code-block:: python
            :caption: Example use of Output_Data
            
            from wavefronts.storage import Input_Data
            from wavefronts.generation import generate_commutative_data, generate_multiplicative_data
            
            
            # Generate input data object from input paramters
            data_input = Input_Data(Simulation_stop_time = '100',L_impedance = '225')
            
            # Generate the commutative merging output data from the created Input_Data object:
            data_output_commutative : Output_Data = generate_commutative_data(data_input)
            # Get sending wavefronts of the capacitor after only commutative merging:
            data_output_commutative.Wavefronts_Sending_Capacitor
            
            # Generate the merged data after multiplicative merging:
            data_output_merged  : Output_Data = generate_multiplicative_data(data_input,data_output_commutative)
            # Get sending wavefronts of the capacitor after multiplicative merging:
            data_output_merged.Wavefronts_Sending_Capacitor
        
    """
    
    Time : np.ndarray
    
    Voltage_Interconnect_Inductor : np.ndarray 
    Current_Interconnect_Inductor : np.ndarray

    Voltage_Interconnect_Capacitor : np.ndarray
    Current_Interconnect_Capacitor : np.ndarray

    Wavefronts_Sending_Inductor : np.ndarray
    Wavefronts_Sending_Capacitor : np.ndarray

    Wavefronts_Returning_Inductor : np.ndarray
    Wavefronts_Returning_Capacitor : np.ndarray
    
    has_merged : bool 
    
    def get_interconnect_array(self,which_string):
        """A method for getting interconnect arrays with a sting enquirey.

        :param which_string: possible options: ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        :type which_string: str
        :raises ValueError: errors if incorrect string is given. 
        :return: the matching interconnect array
        :rtype: np.ndarray[Decimal]
        """
        allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        if(which_string.lower() == allowed_strings[0] ):
            return  self.Voltage_Interconnect_Inductor
        
        elif(which_string.lower() == allowed_strings[1] ):
            return  self.Current_Interconnect_Inductor
        
        elif(which_string.lower() == allowed_strings[2] ):
            return  self.Voltage_Interconnect_Capacitor
        
        elif(which_string.lower() == allowed_strings[3] ):
            return  self.Current_Interconnect_Capacitor
        
        else:
            raise ValueError("Incorrect plotting choice,\'"+which_string+"\' is not an option. Options are : "+ str(allowed_strings))
        
    
    def get_sending_wavefronts_magnitudes(self,which_string):
        """A method for extracting voltage or current from *sending* wavefronts.

        :param which_string: possible options: ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        :type which_string: str
        :raises ValueError: errors if incorrect string is given. 
        :return: Sending wavefront's Current or Voltage magnitudes 
        :rtype: np.ndarray[Decimal]
        """
        allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        if(which_string.lower() == allowed_strings[0] ):
            return  get_voltage_array(self.Wavefronts_Sending_Inductor)
        
        elif(which_string.lower() == allowed_strings[1] ):
            return  get_current_array(self.Wavefronts_Sending_Inductor)
        
        elif(which_string.lower() == allowed_strings[2] ):
            return  get_voltage_array(self.Wavefronts_Sending_Capacitor)
        
        elif(which_string.lower() == allowed_strings[3] ):
            return  get_current_array(self.Wavefronts_Sending_Capacitor)
        
        else:
            raise ValueError("Incorrect plotting choice,\'"+which_string+"\' is not an option. Options are : "+ str(allowed_strings))
        
    def get_returning_wavefronts_magnitudes(self,which_string):
        """A method for extracting voltage or current from *returning* wavefronts.

        :param which_string: possible options: ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        :type which_string: str
        :raises ValueError: errors if incorrect string is given. 
        :return: Returning wavefront's Current or Voltage magnitudes 
        :rtype: np.ndarray[Decimal]
        """
        allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        if(which_string.lower() == allowed_strings[0] ):
            return  get_voltage_array(self.Wavefronts_Returning_Inductor)
        
        elif(which_string.lower() == allowed_strings[1] ):
            return  get_current_array(self.Wavefronts_Returning_Inductor)
        
        elif(which_string.lower() == allowed_strings[2] ):
            return  get_voltage_array(self.Wavefronts_Returning_Capacitor)
        
        elif(which_string.lower() == allowed_strings[3] ):
            return  get_current_array(self.Wavefronts_Returning_Capacitor)
        
        else:
            raise ValueError("Incorrect plotting choice,\'"+which_string+"\' is not an option. Options are : "+ str(allowed_strings))

@dataclass
class Ordered_Output_Data(Output_Data):
    """A dataclass that stores ordered inteface output data in form of single dimenstional arrays. 
    All the core arrays that are present in the Output_Data class are present here but in a one-dimensional chronological form.
    
    :param Indexes: An additonal array, indicating the grid co-ordiantes on the merged fanout structure in the order events occured. 
        Is a single dimesional list of (L,C) coordiante lists. The inner lists take form of [L,C].
    :type Indexes: List[Lists]
    """
    Indexes : np.ndarray

    def get_sending_wavefronts_magnitudes(self,which_string):
        """A method for extracting voltage or current from *sending* wavefronts.

        :param which_string: possible options: ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        :type which_string: str
        :raises ValueError: errors if incorrect string is given. 
        :return: Sending wavefront's Current or Voltage magnitudes 
        :rtype: np.ndarray[Decimal]
        """
        allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        if(which_string.lower() == allowed_strings[0] ):
            return  get_voltage_array(self.Wavefronts_Sending_Inductor)
        
        elif(which_string.lower() == allowed_strings[1] ):
            return  get_current_array(self.Wavefronts_Sending_Inductor)
        
        elif(which_string.lower() == allowed_strings[2] ):
            return  get_voltage_array(self.Wavefronts_Sending_Capacitor)
        
        elif(which_string.lower() == allowed_strings[3] ):
            return  get_current_array(self.Wavefronts_Sending_Capacitor)
        
        else:
            raise ValueError("Incorrect plotting choice,\'"+which_string+"\' is not an option. Options are : "+ str(allowed_strings))
        
    def get_returning_wavefronts_magnitudes(self,which_string):
        """A method for extracting voltage or current from *returning* wavefronts.

        :param which_string: possible options: ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        :type which_string: str
        :raises ValueError: errors if incorrect string is given. 
        :return: Returning wavefront's Current or Voltage magnitudes 
        :rtype: np.ndarray[Decimal]
        """
        allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
        if(which_string.lower() == allowed_strings[0] ):
            return  get_voltage_array(self.Wavefronts_Returning_Inductor)
        
        elif(which_string.lower() == allowed_strings[1] ):
            return  get_current_array(self.Wavefronts_Returning_Inductor)
        
        elif(which_string.lower() == allowed_strings[2] ):
            return  get_voltage_array(self.Wavefronts_Returning_Capacitor)
        
        elif(which_string.lower() == allowed_strings[3] ):
            return  get_current_array(self.Wavefronts_Returning_Capacitor)
        
        else:
            raise ValueError("Incorrect plotting choice,\'"+which_string+"\' is not an option. Options are : "+ str(allowed_strings))

@dataclass
class Interface_Data:
    """A Dataclass that holds all simulation data for a praticular interface. Contains four data storage components that are also the initialization parameters.
    The best way to create this Data storage object is through :py:func:`generation.generate_interface_data` 
    
    :param data_input: input data and calcualted parameters of the interface
    :type data_input: Input_Data
    :param data_output_commutative: Output_Data object for commutative fanouts
    :type data_output_commutative: Output_Data
    :param data_output_multiplicative: Output_Data object for multiplicatively merged fanouts
    :type data_output_multiplicative: Output_Data
    :param data_output_ordered: Chronologically ordered merged data in a linear format 
    :type data_output_ordered: Ordered_Output_Data
    
    .. code-block::
        :caption: make a `Interface_Data` object and plot it's refelction diagrm
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_refelction_diagram
        import matplotlib.pyplot as plt

        # simulate an interface by providing key-values altered from the defaults
        interface_data = generate_interface_data(L_time = '3.6',C_time = '3.2',L_impedance = '300')

        # The interface object created stores all level of data from the simulation
        data_input = interface_data.data_input
        data_output_commutative = interface_data.data_output_commutative
        data_output_multiplicative = interface_data.data_output_multiplicative
        data_output_ordered = interface_data.data_output_ordered

        # plot the current reflection diagram  of the interface
        fig, ax = plt.subplots()
        plot_refelction_diagram(interface_data,ax,False,stop_time='40')

        plt.show()
    """
    data_input : Input_Data
    data_output_commutative : Output_Data
    data_output_multiplicative : Output_Data
    data_output_ordered : Ordered_Output_Data
