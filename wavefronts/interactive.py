from wavefronts.storage import Interface_Data
import wavefronts.plotting as wp
from wavefronts.misc import handle_default_kwargs

from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import ipywidgets as widgets
from IPython.display import display


def interact_spatial(Interface : Interface_Data, number_of_steps:int = 1000):
    """Creates an interactive spatial plot using ipywidgets. The slider bar ranges from 0 to number_of_steps (default is 1000).

    :param Interface: interface data storage object
    :type Interface: Interface_Data
    :param number_of_steps: the number of steps in the slider, defaults to 1000
    :type number_of_steps: int, optional
    """
    # define widgets
    increment_button = widgets.Button(description = "step forward", layout=widgets.Layout(width='auto'))
    decrement_button = widgets.Button(description = "step backward", layout=widgets.Layout(width='auto'))
    increment_text = widgets.FloatText(description = 'increment', value=0.1)
    auto_zoom_toggle = widgets.Checkbox(value=False,description='Auto-Zoom',disabled=False,tooltip='if spatial plots axes must zoom to features or be constant')
    time_slider = widgets.FloatSlider(value=0, min =0, max = number_of_steps, step = 1, layout=widgets.Layout(width='auto'),tooltip='enter number from 0 to 1000')
    
    fig_s,ax_s = plt.subplot_mosaic([['V','inter-V'],
                                     ['I','inter-I']],figsize=(14, 8))
    
    def handle_input(t:Decimal):
        t= Interface.data_input.Simulation_Stop_Time* t / Decimal(str(number_of_steps))
        wp.clear_subplot(ax_s.values())
        VC,VL,IC,IL = wp.make_spatial_voltage_and_current(t,Interface,ax=ax_s,fig_size=(14, 8),return_data=True)
        wp.plot_timewaveforms_and_intercepts(t,Interface.data_output_ordered,ax_voltage=ax_s['inter-V'],ax_current=ax_s['inter-I'])

        x_v_edge = ax_s['V'].get_xlim()[1]
        x_i_edge = ax_s['I'].get_xlim()[1]
        
        ax_s['V'].plot([0,x_v_edge],[VL,VL],linestyle='--')
        ax_s['V'].plot([0,x_v_edge],[VC,VC],linestyle='--')
        
        ax_s['I'].plot([0,x_i_edge],[IL,IL],linestyle='--')
        ax_s['I'].plot([0,x_i_edge],[IC,IC],linestyle='--')
        
        if(auto_zoom_toggle.value == False):
            ax_s['V'].set_ylim(ax_s['inter-V'].get_ylim())
            ax_s['I'].set_ylim(ax_s['inter-I'].get_ylim())
    
    handle_input(Decimal('0'))


    def on_increment_click(b):
        time_slider.value += increment_text.value
        time = Decimal(str(time_slider.value))
        handle_input(time)
        
    def on_decrement_click(b):
        time_slider.value -= increment_text.value
        time = Decimal(str(time_slider.value))
        handle_input(time)
        
    def handle_slider_change(change):
        if(isinstance(change.new,dict)):
            if(len(change.new) > 0):
                time = Decimal(str(change.new['value']))
                handle_input(time)
                
    def update(b):
        time = Decimal(str(time_slider.value))
        handle_input(time)
                
                
    increment_button.on_click(on_increment_click)
    decrement_button.on_click(on_decrement_click)
    time_slider.observe(handle_slider_change)
    auto_zoom_toggle.observe(update)

    increment_grid = widgets.GridspecLayout(1,4)
    increment_grid[0,0] = decrement_button
    increment_grid[0,1] = increment_button
    increment_grid[0,2] = increment_text
    increment_grid[0,3] = auto_zoom_toggle

    display(increment_grid,time_slider)
    
def interact_fanout_path(Interface : Interface_Data, is_Voltage:bool =True,padding:int =0):
    """Creates an interactive plot that draws the 'path of occurence' ontop merged interconnect fanouts.

    :param Interface: interface data storage object
    :type Interface: Interface_Data
    :param is_Voltage: if to show voltage or current fanouts, defaults to True.
    :type is_Voltage: bool
    :param padding: the padding around the fanouts, default = 0.
    :type padding: int
    
    """

    fig_path, ax_path = plt.subplot_mosaic([['INTER','INTER'],
                                            ['LF','CF']])
    
    formatter =  EngFormatter('s')
    
    if(is_Voltage):
        which_str_prefix = 'voltage '
        ax_voltage = ax_path['INTER']
        ax_current =False
    else:
        which_str_prefix = 'current '
        ax_voltage =False
        ax_current = ax_path['INTER']
        

    wp.plot_timewaveforms_and_intercepts(0,Interface,ax_voltage=ax_voltage,ax_current=ax_current)
    wp.plot_fanout_interconnect(Interface.data_output_multiplicative,ax_path['LF'],which_str_prefix+'inductor',padding=padding)
    wp.plot_fanout_interconnect(Interface.data_output_multiplicative,ax_path['CF'],which_str_prefix+'capacitor',padding=padding)
    
    def remember_lims(axes):
        lims = []
        for ax in axes:
            lims.append((ax.get_xlim(),ax.get_ylim()))
            
        return lims
    
    def set_lims(axes,lims):
        for ax, lim in zip(axes,lims):
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])

    def plot_path(t):
        
        lims = remember_lims(ax_path.values())
        wp.clear_subplot(ax_path.values())
        set_lims(ax_path.values(),lims)
        t = Decimal(str(t))
        t = Interface.data_input.Simulation_Stop_Time * t/Decimal('1000')
        
        wp.plot_timewaveforms_and_intercepts(t,Interface,ax_voltage=ax_voltage,ax_current=ax_current)
        wp.plot_fanout_interconnect(Interface.data_output_multiplicative,ax_path['LF'],which_str_prefix+'inductor',show_colour_bar=False,padding=padding)
        wp.plot_fanout_interconnect(Interface.data_output_multiplicative,ax_path['CF'],which_str_prefix+'capacitor',show_colour_bar=False,padding=padding)
        wp.plot_trace_on_merged_fanout_axis(Interface,ax_path['LF'],t,show_cross=True,padding=padding)
        wp.plot_trace_on_merged_fanout_axis(Interface,ax_path['CF'],t,show_cross=True,padding=padding)
        fig_path.suptitle(f"t = {formatter.format_eng(float(t))}s")
        
    inter = widgets.interact(plot_path,t=widgets.FloatSlider(min=0, max=1000, step=1, value=0, layout=widgets.Layout(width='auto')))

def interact_3D_spatial(Interface : Interface_Data,**kwargs):
    """Creates an interactive 3D plot of voltage and current distributed in space with a slider bar to scrub time.
    (Not recomended for ver small time delays as the interactive )

    :param Interface: interface data storage object
    :type Interface: Interface_Data
    :param is_Voltage: if to show voltage or current fanouts, defaults to True.
    :type is_Voltage: bool
    :param padding: the padding around the fanouts, default = 0.
    :type padding: int
    
    """
    default_kwargs ={
        'number_of_steps': 1000,
        'z_lim': False,
        'y_lim': False
    }
    
    formatter =  EngFormatter('s')
    
    kwargs = handle_default_kwargs(kwargs,default_kwargs)
    
    fig_3d = plt.figure()

    ax_3d = fig_3d.add_subplot(111,projection='3d')

    x_pos_lim_0 = -float(Interface.data_input.Capacitor_Length)
    x_pos_lim_1 = float(Interface.data_input.Inductor_Length)

    if(isinstance(kwargs['y_lim'],bool)):
        
        y_current_lim_0 = float(min([min(np.cumsum(Interface.data_output_ordered.Current_Interconnect_Inductor)),min(np.cumsum(Interface.data_output_ordered.Current_Interconnect_Capacitor))]))
        y_current_lim_1 = float(max([max(np.cumsum(Interface.data_output_ordered.Current_Interconnect_Inductor)),max(np.cumsum(Interface.data_output_ordered.Current_Interconnect_Capacitor))]))
    else:
        y_current_lim_0 = kwargs['y_lim'][0]
        y_current_lim_1 = kwargs['y_lim'][1]
        
    if(isinstance(kwargs['z_lim'],bool)):
        z_voltage_lim_0 = float(min([min(np.cumsum(Interface.data_output_ordered.Voltage_Interconnect_Inductor)),min(np.cumsum(Interface.data_output_ordered.Voltage_Interconnect_Capacitor))]))
        z_voltage_lim_1 = float(max([max(np.cumsum(Interface.data_output_ordered.Voltage_Interconnect_Inductor)),max(np.cumsum(Interface.data_output_ordered.Voltage_Interconnect_Capacitor))]))
    else:
        z_voltage_lim_0 = kwargs['z_lim'][0]
        z_voltage_lim_1 = kwargs['z_lim'][1]

    def interact_3D_func(t):
        t = t* float(Interface.data_input.Simulation_Stop_Time)/kwargs['number_of_steps']
        ax_3d.clear()
        ax_3d.set_xlim(x_pos_lim_0,x_pos_lim_1)
        ax_3d.set_ylim(y_current_lim_1,y_current_lim_0)
        ax_3d.set_zlim(z_voltage_lim_0,z_voltage_lim_1)
        wp.make_3d_spatial(str(t),Interface,ax_3d)
        ax_3d.set_title(f"spatial plot at t = {formatter.format_eng(t)}s")
        
    inter = widgets.interact(interact_3D_func,t=widgets.FloatSlider(min=0, max=kwargs['number_of_steps'], step=1, value=0, layout=widgets.Layout(width='auto')))