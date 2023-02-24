"""The module responsible for visualisation of distributed behaviours. 
In general a function will either be a 'make' or a 'plot' type.
'plot' functions require the creation of plotting  axes to be provided to the function to be 'plotted on'.
These type of functions typically are plotted on a single axis, were the format of the axis is irrelavant and flexible.
'make' functions on the other hand generate axes internally and can have axes passed to them, however they must be of a particualr format.
Make functions oftens setup axes in a particular way and is why they handle the generation of the the axes.
Internal creation of axes can potentially be problematic when doing multiple loops on a make function, in this case be sure to pass an in axes of the correct format as described per function.
"""

from wavefronts.generation import get_spatial_voltage_current_at_time, transform_merged_array_to_capacitor_axis
from wavefronts.storage import *
from wavefronts.misc import *

from decimal import Decimal, ROUND_HALF_DOWN
import copy
from warnings import warn
import numpy as np
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.animation import FFMpegWriter
plt.rcParams['animation.ffmpeg_path'] = 'wavefronts\\ffmpeg\\ffmpeg.exe'



def clear_subplot(axs):
    """a little loop that clears all axes of an axes object

    :param axs: axes object array to be cleared
    :type axs: matplotlib Axes 
    """
    for ax in axs:
        ax.cla()

# Fanout Diagrams

def handle_interface_to_ordered(data) -> Ordered_Output_Data:
    """ensures data is ordered, extracts it if it can, else raises an error.

    :param data: input data to be checked
    :type data: any
    :raises TypeError: if ordered data cannot be extracted
    :return: ordered data 
    :rtype: Ordered_Output_Data
    """
    if isinstance(data, Interface_Data ):
        data = data.data_output_ordered
    elif isinstance(data, Ordered_Output_Data ):
        pass
    else:
        raise TypeError(f"input data is of wrong type, must be Ordered_Output_Data. inputted {type(data)} instead.")
    
    return data

default_fanout_kwargs = {
    'title': "Magnitude Fanout",
    'show_colour_bar': True,
    'contrast' : True,
    'padding' : 0,
    'units' : 'A',
    'origin' : 'lower',
    'transpose' : True,
    'show_ticks' : False,
    'custom_colour_bar_limits': False
}

def plot_fanout_magnitude(array_to_plot : np.ndarray , ax, **input_kwargs):
    """the core function for plotting the fanout diagram of a 2D numpy array.
    Points are coloured using the 'seismic' colour map with red being positive and blue negative.
    See :py:func:`plot_fanout_interconnect` and :py:func:`plot_fanout_wavefronts` for prettier plots with more automation

    :param array_to_plot: The array to be plotted, can also accept lists of numerical data
    :type array_to_plot: np.ndarray or List
    :param ax: a matplotlib Axe object to plot using 'imshow'
    :type ax: matplotlib.Axe
    
    :**input_kwargs**:
        - **title** (*str*) - The title of the fanout (default = "Magnitude Fanout")
        - **show_colour_bar** (*bool*) - if colour bar must be shown (default = True)
        - **contrast** (*bool*) - if the orign node must be ignored for the colour mapping maximum value calculation (default = False)
        - **padding** (*int*) - the amount of padding around the array, thinner arrays are easier to navigate with padding (default = 0)
        - **units** (*str*) - the units of the colour bar (default = 'A')
        - **origin** (*str*) - either 'lower' or 'upper', sets the postion of the origin (default = 'lower')
        - **transpose** (*bool*) - makes x-axis the L-axis if true (default = True)
        - **show_ticks** (*bool*) - if axis ticks are shown (default = False)
        - **custom_colour_bar_limits** (*tuple or bool*) - pass a (max_value, min_value) tuple to customize colouring extent of the fanout(default = False)
        
    .. warning::
        a **wavefront storage array** must be in their magnitude forms, these arrays can be fetched using :py:meth:`storage.Output_Data.get_sending_wavefronts_magnitudes` 
        or :py:meth:`storage.Output_Data.get_returning_wavefronts_magnitudes`. 
        Alternatively magnitdues from a **wavefront array** can be manually extracted by passing as an input parameter to 
        :py:func:`misc.get_voltage_array` or :py:func:`misc.get_current_array`
        
    .. code-block::
        :caption: simple use
        
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_fanout_magnitude
        import matplotlib.pyplot as plt

        # simulate interface
        interface_data = generate_interface_data(L_time='0.7' , C_time='3.2')

        # plot the commutatiive capacitor interconnect voltage 
        fig, ax = plt.subplots()
        arr = interface_data.data_output_commutative.Voltage_Interconnect_Capacitor
        # set units to 'V'
        plot_fanout_magnitude(arr,ax, units = 'V')
        plt.show()
        
    .. code-block::
        :caption: manual wavefront fanout, see :py:func:`plot_fanout_wavefronts`
        
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_fanout_magnitude
        import matplotlib.pyplot as plt

        # simulate interface
        interface_data = generate_interface_data(L_time='23' , C_time='11')

        # plot the multiplicative sending current capacitor wavefronts
        fig, ax = plt.subplots()
        arr = interface_data.data_output_multiplicative.get_sending_wavefronts_magnitudes('current capacitor')
        # set units to 'V'
        plot_fanout_magnitude(arr,ax, units = 'A')
        plt.show()
        
    :return: plots a magnitude fanout on the provided axis
    """
    
    default_kwargs = handle_default_kwargs(input_kwargs,default_fanout_kwargs)
    # convert Lists to image array if necessary 
    array_to_plot = convert_to_image_array(array_to_plot)
    
    if(isinstance(default_kwargs['custom_colour_bar_limits'] ,tuple)):
        max_boundary = default_kwargs['custom_colour_bar_limits'][0]
        min_boundary = default_kwargs['custom_colour_bar_limits'][1]
    elif (default_kwargs['contrast']): 
        Contrast = copy.copy(array_to_plot.astype(float))
        max_index = np.unravel_index(np.argmax(Contrast, axis=None), Contrast.shape)
        Contrast[max_index] = 0
        max_boundary = get_array_absolute_maximum(Contrast)
        min_boundary = -max_boundary
    else:
        max_boundary = get_array_absolute_maximum(array_to_plot.astype(float))
        min_boundary = -max_boundary
    
    if default_kwargs['transpose'] :
        array_plot = np.pad(array_to_plot.astype(float),(default_kwargs['padding'],default_kwargs['padding'])).transpose()
        ax.set_xlabel('L - axis ')
        ax.set_ylabel('C - axis ')
    else:
        array_plot = np.pad(array_to_plot.astype(float),(default_kwargs['padding'],default_kwargs['padding']))
        ax.set_ylabel('L - axis ')
        ax.set_xlabel('C - axis ')
    
    def offset_formatter(x, pos):
        if (x - default_kwargs['padding'] >= 0):
            return int(x - default_kwargs['padding'])
        else:
            return ""

    if(default_kwargs['show_ticks']):
        ax.xaxis.set_major_formatter(plt.FuncFormatter(offset_formatter))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(offset_formatter))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
      
    ax.set_title(default_kwargs['title'])
    
    c = ax.imshow(array_plot,cmap= mpl.cm.seismic,vmax =max_boundary, vmin = min_boundary,origin=default_kwargs['origin'])
    
    if(default_kwargs['show_colour_bar']):
        cb = ax.get_figure().colorbar(c,ax=ax)
        cb.ax.yaxis.set_major_formatter(EngFormatter(default_kwargs['units']))
        
def plot_fanout_time(input_array : np.ndarray ,ax , **input_kwargs):
    """Plot a time fanout of a provided input array. 
    Coloured in a rainbow pattern from the minimum array value to the maximum array value.

    :param input_array: The array to be plotted, can also accept lists of numerical data
    :type input_array: np.ndarray or List
    :param ax: a matplotlib Axe object to plot using 'imshow'
    :type ax: matplotlib.Axe
    
    :**input_kwargs**:
        - same input kwargs as :py:func:`plot_fanout_magnitude`
        - **mask_zero** (*bool*) - if zeros values must be masked (default = True)
        
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_fanout_time
        import matplotlib.pyplot as plt

        # simulate interface
        interface_data = generate_interface_data(L_time='2' , C_time='7')

        # plot the time fanout
        fig, ax = plt.subplots()
        plot_fanout_time(interface_data.data_output_commutative.Time, ax)

        plt.show()
        
    """
    default_kwargs = default_fanout_kwargs.copy()
    default_kwargs['mask_zero']  = True
    default_kwargs['units'] = 'S'
    default_kwargs['title'] = 'Time Fanout'
    
    default_kwargs = handle_default_kwargs(input_kwargs,default_kwargs)
    # convert Lists to image array if necessary 
    input_array = convert_to_image_array(input_array)
    
    if (default_kwargs['custom_colour_bar_limits']==False):
        max_boundary = np.max(input_array.astype(float))  
        min_boundary = np.min(input_array.astype(float))  
        
    else:
        max_boundary, min_boundary = default_kwargs['custom_colour_bar_limits']
        
    if default_kwargs['transpose'] :
        array_plot = np.pad(input_array.astype(float),(default_kwargs['padding'],default_kwargs['padding'])).transpose()
        ax.set_xlabel('L - axis ')
        ax.set_ylabel('C - axis ')
    else:
        array_plot = np.pad(input_array.astype(float),(default_kwargs['padding'],default_kwargs['padding']))
        ax.set_ylabel('L - axis ')
        ax.set_xlabel('C - axis ')
    
    def offset_formatter(x, pos):
        return int(x - default_kwargs['padding'])

    if(default_kwargs['show_ticks']):
        ax.xaxis.set_major_formatter(plt.FuncFormatter(offset_formatter))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(offset_formatter))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        
    if(default_kwargs['mask_zero']):
        array_plot = np.ma.masked_where(array_plot == 0, array_plot)
        array_plot[0,0] = 0
      
    ax.set_title(default_kwargs['title'])
    
    c = ax.imshow(array_plot,cmap= mpl.cm.jet,vmax =max_boundary, vmin = min_boundary,origin=default_kwargs['origin'])
    
    if(default_kwargs['show_colour_bar']):
        cb = ax.get_figure().colorbar(c,ax=ax)
        cb.ax.yaxis.set_major_formatter(EngFormatter(default_kwargs['units']))
            
def plot_fanout_interconnect(data_output: Output_Data,ax, which_string :str, contrast_voltage = True,**kwargs):
    """A wrapper function for :py:func:`plot_fanout_magnitude` for plotting interconnect fanouts.
    Takes in a Output_Data object and a string to plot and auto format the fanout.
    It will pass provided **kwargs to the underlying plot_fanout_magnitude function.
    To plot all interface interconnect fanouts at once see :py:func:`make_fanout_interconnect_all`
    
    :param data_output: The data output object that contians the interconnect arrays. Could be commutative or multiplicative data. 
    :type data_output: Output_Data
    :param ax: the matplotlib axis to plot on
    :type ax: matplotlib Axes object
    :param which_string: determine which interconnect value to plot. Options are "voltage inductor", "current inductor", "voltage capacitor", "current capacitor"
    :type which_string: str
    :param contrast_voltage: determine if voltage arrays must exclude the orign point for better contrast, default is True
    :type contrast_voltage: bool
    :raises ValueError: if incorrect 'which_string' is not provided.
    :raises warning: if 'title=', 'units=' or 'contrast=' keyword are included as they are auto assigned by this function
    
    .. code-block ::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_fanout_interconnect
        import matplotlib.pyplot as plt

        # simulate interface
        interface_data = generate_interface_data(L_time='12' , C_time='13')

        # compare commutative and multiplicative capacitor interconnect voltage
        # (on the same subplot)
        fig, ax = plt.subplots(1,2, figsize= (10,6))

        # pass commutative data output
        plot_fanout_interconnect(interface_data.data_output_commutative,
                                ax[0], 'voltage capacitor')

        # pass multiplicative data output 
        plot_fanout_interconnect(interface_data.data_output_multiplicative,
                                ax[1], 'voltage capacitor')

        plt.show()
    
    .. warning::
        When providing the ****kwargs**, you cannot specify 'title=', 'units=' or ''contrast='  as these are auto assinged. Providing these values will result in an error. 
    """
    
    if ('title' in kwargs):
        warn("you cannot specifiy the title of these fanouts as they are automatically assigned. Use plot_fanout_magnitude() instead")
        del kwargs['title']
    elif('units' in kwargs):
        warn("you cannot specifiy the units of these fanouts as they are automatically assigned. Use plot_fanout_magnitude() instead")
        del kwargs['units']
    elif('contrast' in kwargs):
        warn("you cannot specifiy the contrast of these fanouts as they are automatically assigned. Use the third input parameter of this function to control contrast")
        del kwargs['contrast']
    
    allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
    if(which_string.lower() == allowed_strings[0] ):
        plot_fanout_magnitude( data_output.Voltage_Interconnect_Inductor,ax,title = "Inductor Voltage at Interconnect",units='V',contrast=contrast_voltage,**kwargs)
    elif(which_string.lower() == allowed_strings[1] ):
        plot_fanout_magnitude(data_output.Current_Interconnect_Inductor,ax,title = "Inductor Current at Interconnect",**kwargs)
    elif(which_string.lower() == allowed_strings[2] ):
        plot_fanout_magnitude(data_output.Voltage_Interconnect_Capacitor,ax,title = "Capacitor Voltage at Interconnect",units='V',contrast=contrast_voltage,**kwargs)
    elif(which_string.lower() == allowed_strings[3] ):
        plot_fanout_magnitude(data_output.Current_Interconnect_Capacitor,ax,title = "Capacitor Current at Interconnect",**kwargs)
    else:
            raise ValueError(f"Incorrect plotting choice /, {which_string} is not a valid option. Optiond are: \n {allowed_strings}")
        
def plot_fanout_wavefronts(data_output: Output_Data,ax, which_string :str, is_sending : bool = True, **kwargs):
    """A wrapper function for :py:func:`plot_fanout_magnitude` for plotting wavefront fanouts.
    Takes in a Output_Data object, a string and a bool are passed to plot and auto format the fanout.
    It will pass provided **kwargs to the underlying plot_fanout_magnitude function.
    To plot all wavefront fanouts at once see :py:func:`make_fanout_wavefronts_all`
    

    :param data_output: The data output object that contians the interconnect arrays. Could be commutative or multiplicative data. 
    :type data_output: Output_Data
    :param ax: the matplotlib axis to plot on
    :type ax: matplotlib Axes object
    :param which_string: determine which interconnect value to plot. Options are "voltage inductor", "current inductor", "voltage capacitor", "current capacitor"
    :type which_string: str
    :raises ValueError: if incorrect 'which_string' is not provided.
    :param is_sending: determines if sending or returning wavefronts must be plotted, defaults to True
    :type is_sending: bool, optional
    
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_fanout_wavefronts
        import matplotlib.pyplot as plt

        # simulate interface
        interface_data = generate_interface_data(L_time='12' , C_time='13')

        # compare sending and returning capacitor current wavefronts
        # (on the same subplot)
        fig, ax = plt.subplots(1,2, figsize= (10,6))

        # sending current wavefronts
        plot_fanout_wavefronts(interface_data.data_output_commutative,
                                ax[0], 'current capacitor',True)

        # returning current wavefront 
        plot_fanout_wavefronts(interface_data.data_output_commutative,
                                ax[1], 'current capacitor',False)

        plt.show()
    
    .. warning::
        When providing the ****kwargs**, you cannot specify 'title=' or 'units=' as these are auto assinged. Providing these values will result in an error. 
    """
    if ('title' in kwargs):
        warn("you cannot specifiy the title of these fanouts as they are automatically assigned. Use plot_fanout_magnitude() instead")
        del kwargs['title']
    elif('units' in kwargs):
        warn("you cannot specifiy the units of these fanouts as they are automatically assigned. Use plot_fanout_magnitude() instead")
        del kwargs['units']
    
    allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
    
    if(is_sending):
        title_prefix = "Sending "
        get_func = data_output.get_sending_wavefronts_magnitudes
    else:
        title_prefix = "Returning "
        get_func = data_output.get_returning_wavefronts_magnitudes
    
    if(which_string.lower() == allowed_strings[0] ):
        plot_fanout_magnitude(get_func(which_string),ax,title = title_prefix + "Voltage Wavefronts\n in Inductor",units='V',**kwargs)
        
    elif(which_string.lower() == allowed_strings[1] ):
        plot_fanout_magnitude(get_func(which_string),ax,title = title_prefix + "Current Wavefronts\n in Inductor",**kwargs)
        
    elif(which_string.lower() == allowed_strings[2] ):
        plot_fanout_magnitude(get_func(which_string),ax,title = title_prefix + "Voltage Wavefronts\n in Capacitor",units='V',**kwargs)
        
    elif(which_string.lower() == allowed_strings[3] ):
        plot_fanout_magnitude(get_func(which_string),ax,title = title_prefix + "Current Wavefronts\n in Capacitor",**kwargs)
    else:
            raise ValueError(f"Incorrect plotting choice /, {which_string} is not a valid option. Optiond are: \n {allowed_strings}")

def make_fanout_crossection(input_array : np.ndarray, L_intercept : int, C_intercept : int, **kwargs):
    """Plots a magnitude fanout and corssection at a L and C intercept for a given input data array.
    The kwargs supplied are passed down to :py:func:`plot_fanout_magnitude`. 
    Additonal key-value customiztion is included for the crossection plot below.

    :param input_array: The fanout data to be investigated
    :type input_array: np.ndarray
    :param L_intercept: The value on the L-axis to intercept
    :type L_intercept: int
    :param C_intercept: The value on the C-axis to intercept
    :type C_intercept: int
    :return: the matplotlib Figure and Axes objects created in this function
    :rtype: tuple( fig , ax )
    :**kwargs for crossection**:
        - **ax** (*Dict(Axes)*) - Whether to create a subpot or use exsiting subplot axes.If left blank default is 'False' and subplot is created internally.If axes are provided, the must be of a matplotlib.pyplot.subplot_mosaic() form.The labels for these axes must inculde: 
            - 'C' for C-plot/ L interception
            - 'L' for L-plot/ C interception
            - 'D' for the Diagonal plot
            - 'F' for Fanout magnitude plot
        - **fig_size** (*tuple of ints*) - The size of the figure. Default is (10, 8).
        - **Transpose_C_Plot** (*bool*) - Whether to transpose the C plot. Default is True.
        - **Transpose_L_Plot** (*bool*) - Whether to transpose the L plot. Default is False
        
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import make_fanout_crossection
        import matplotlib.pyplot as plt
        
        # simulate interface
        interface = generate_interface_data(L_time='6.5' , C_time='3' , L_impedance='700')
        
        # make axes internally, intercept at L=25, C= 10
        data = interface.data_output_commutative.Voltage_Interconnect_Capacitor
        make_fanout_crossection(data, 25, 10, units='V')
        
        # make axes externally, intercept at L=25, C= 10
        
        fig, ax = plt.subplot_mosaic([['C','F'],
                                      ['D','L']])
                                      
        make_fanout_crossection(data, 25, 10, units='V', ax=ax)
        
        plt.show()
        
        
    .. warning::
        if ax keyword is not provided, function will make new subplot objects each time it is called.
        These plots will not be closed by default, so if multiple calls are needed it is suggested you provide
        the appropriate subplot_mosaic Axes object.
    """
    
    default_crossection_kwargs : dict = {'ax':False,
                                         'fig_size':(10,8),
                                         'Transpose_C_Plot':True,
                                         'Transpose_L_Plot':False}
    
    internal_fanout_kwargs = default_fanout_kwargs.copy()
    internal_fanout_kwargs['show_ticks'] = True
    
    crossection_kwargs, fanout_kwargs = split_outer_inner_default_kwargs(kwargs,default_crossection_kwargs,internal_fanout_kwargs)
    input_array = convert_to_image_array(input_array)# converts list to 2D numpy object
    
    if (crossection_kwargs['ax'] == False):
        # create fig and ax
        fig, ax = plt.subplot_mosaic([['C','F'],
                                      ['D','L']])
    else :
        ax = crossection_kwargs['ax']
        fig = ax['C'].get_figure()

    fig.set_size_inches(crossection_kwargs['fig_size'])
    fig.suptitle(f"Crossection of Fanout at index L = {L_intercept}, C = {C_intercept}")
    
    # handle out of bounds
    input_array_shape = input_array.shape
    if (L_intercept < 0):
        L_intercept = 0
    elif(L_intercept>input_array_shape[0]-1):
        L_intercept = input_array_shape[0]-1
        
    if (C_intercept < 0):
        C_intercept = 0
    elif(C_intercept>input_array_shape[1]-1):
        C_intercept = input_array_shape[1]-1
        
    L_y = input_array[:,C_intercept]
    C_y = input_array[L_intercept,:]
    D_y = np.diag(input_array)
    
    L_x = np.arange(0,len(L_y))
    C_x = np.arange(0,len(C_y))
    
    if (crossection_kwargs['Transpose_L_Plot']):
        ax['L'].set_ylabel('L-axis')
        ax['L'].set_ylim(0,input_array_shape[0])
        ax['L'].xaxis.set_major_formatter(EngFormatter(kwargs['units']))
        ax['L'].plot(L_y,L_x)
        ax['L'].axvline(x=0,color='gray',linestyle= '--')
    else:
        ax['L'].set_xlabel('L-axis')
        ax['L'].set_xlim(0,input_array_shape[0])
        ax['L'].yaxis.set_major_formatter(EngFormatter(kwargs['units']))
        ax['L'].plot(L_x,L_y)
        ax['L'].axhline(y=0,color='gray',linestyle= '--')
    
    if(crossection_kwargs['Transpose_C_Plot']):
        ax['C'].set_ylabel('C-axis')
        ax['C'].set_ylim(0,input_array_shape[1])
        ax['C'].xaxis.set_major_formatter(EngFormatter(kwargs['units']))
        ax['C'].plot(C_y,C_x)
        ax['C'].axvline(x=0,color='gray',linestyle= '--')
    else:
        ax['C'].set_xlabel('L-axis')
        ax['C'].set_xlim(0,input_array_shape[1])
        ax['C'].yaxis.set_major_formatter(EngFormatter(kwargs['units']))
        ax['C'].plot(C_x,C_y)
        ax['C'].axhline(y=0,color='gray',linestyle= '--')
        
    ax['D'].set_xlabel('Diagonal-axis')
    ax['D'].yaxis.set_major_formatter(EngFormatter(kwargs['units']))
    ax['D'].plot(D_y)
    ax['D'].set_xlim(0)
    
    plot_fanout_magnitude(input_array,ax['F'],**fanout_kwargs)
    ax['F'].plot([0,input_array_shape[0]],[C_intercept,C_intercept],'k-')
    ax['F'].plot([L_intercept,L_intercept],[0,input_array_shape[1]],'k-')
    ax['F'].plot([0,input_array_shape[0]],[0,input_array_shape[1]],'k-')
    
    # if figure created internally then return created handles 
    if (crossection_kwargs['ax'] == False):
        return fig,ax

def make_fanout_interconnect_all(data_output: Output_Data,contrast_voltage = True,**kwargs):
    """plots all the interconnect magnitude fanouts for a particular Output_Data object. 
    A wrapper fucniton for :py:func:`plot_fanout_interconnect`.
    This is a 'make' type function which means that by default the function will internally create the plotting axes unless specified otherwise. 
    The kwargs supplied are passed down to :py:func:`plot_fanout_interconnect`. 
    Additonal key-value customiztion is included for the crossection plot below.

    :param data_output: The data object to be plotted
    :type data_output: Output_Data
    :param contrast_voltage: if the voltage arrays must ignore the intial excitation point for better contrast, defaults to True
    :type contrast_voltage: bool, optional
    :return: the matplotlib Figure and Axes objects created in this function (if created)
    :rtype: tuple( fig , ax )
    :**kwargs for figure creation**:
        - **ax** (*Dict(Axes)*) - Whether to create a subpot or use exsiting subplot axes.If left blank default is 'False' and subplot is created internally.If axes are provided, the must be of a matplotlib.pyplot.subplot_mosaic() form.The labels for these axes must inculde: 
            - 'VL' axis for inductor voltage
            - 'VC' axis for capcitor voltage
            - 'IL' axis for inductor current
            - 'IC' axis for capacitor current
        - **fig_size** (*tuple of ints*) - The size of the figure. Default is (10, 8).

    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import make_fanout_interconnect_all
        import matplotlib.pyplot as plt

        # simulate interface
        interface = generate_interface_data(L_time='12' , C_time='8')

        # make figure internally, plot commutative data
        fig1,ax1 = make_fanout_interconnect_all(interface.data_output_commutative)
        fig1.suptitle(f"commutative Fanouts") # customize title

        # make figure externally, put currents left and voltages right
        fig2, ax2 = plt.subplot_mosaic([['IL','VL'],
                                        ['IC','VC']])

        # pass ax2 to fucniton, also, show multiplicative data this time
        make_fanout_interconnect_all(interface.data_output_multiplicative, ax=ax2)
        fig2.suptitle(f"multiplicative Fanouts") # customize title

        plt.show()
        
    .. warning::
        if ax keyword is not provided, function will make new subplot objects each time it is called.
        These plots will not be closed by default, so if multiple calls are needed it is suggested you provide
        the appropriate subplot_mosaic Axes object.
    """

    default_make_kwargs : dict = {'ax':False,
                                  'fig_size':(10,8)}
    
    make_kwargs, fanout_kwargs = split_outer_inner_default_kwargs(kwargs,default_make_kwargs,default_fanout_kwargs)
    
    del fanout_kwargs['title']
    del fanout_kwargs['units']
    del fanout_kwargs['contrast']
    
    if (make_kwargs['ax'] == False):
        fig, ax = plt.subplot_mosaic([['VL','VC'],
                                    ['IL','IC']])
    else:
        ax = make_kwargs['ax']
        fig = ax['VL'].get_figure()
        
    
    fig.set_size_inches(make_kwargs['fig_size'])
    fig.suptitle(f"Interconnect Fanouts")
    
    plot_fanout_interconnect(data_output,ax['VL'],"Voltage Inductor",contrast_voltage,**fanout_kwargs)
    plot_fanout_interconnect(data_output,ax['IL'],"Current Inductor",**fanout_kwargs)
    plot_fanout_interconnect(data_output,ax['VC'],"Voltage Capacitor",**fanout_kwargs)
    plot_fanout_interconnect(data_output,ax['IC'],"Current Capacitor",**fanout_kwargs)
    
    # figure internally created, return figure and axes
    if (make_kwargs['ax'] == False):
        return fig,ax

def make_fanout_wavefronts_all(data_output: Output_Data,is_Inductor: bool,**kwargs):
    """plots all the sending and returning magnitude fanouts for a transmission line of a Output_Data object. 
    A wrapper fucniton for :py:func:`plot_fanout_wavefronts`.
    This is a 'make' type function which means that by default the function will internally create the plotting axes unless specified otherwise. 
    The kwargs supplied are passed down to :py:func:`plot_fanout_wavefronts`. 
    Additonal key-value customiztion is included for the crossection plot below.

    :param data_output: The data object to be plotted
    :type data_output: Output_Data
    :param is_Inductor: if the wavefronts shown are form the inductor or the capacitor.
    :type is_Inductor: bool
    :return: the matplotlib Figure and Axes objects created in this function (if created)
    :rtype: tuple( fig , ax )
    :**kwargs for figure creation**:
        - **ax** (*Dict(Axes)*) - Whether to create a subpot or use exsiting subplot axes.If left blank default is 'False' and subplot is created internally.If axes are provided, the must be of a matplotlib.pyplot.subplot_mosaic() form.The labels for these axes must inculde: 
            - 'VS' axis for sending voltage
            - 'VR' axis for returning voltage
            - 'IS' axis for sending current
            - 'IR' axis for returning current
        - **fig_size** (*tuple of ints*) - The size of the figure. Default is (10, 8).

    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import make_fanout_wavefronts_all
        import matplotlib.pyplot as plt

        # simulate interface
        interface = generate_interface_data(L_time='0.34' , C_time='0.12', L_impedance = '700', C_impedance = '7')

        # make figure internally, 
        # plot commutative inductive wavefronts
        fig_ind,ax_ind = make_fanout_wavefronts_all(interface.data_output_commutative,True)

        # plot commutative capacitive wavefronts
        fig_cap,ax_cap = make_fanout_wavefronts_all(interface.data_output_commutative,False)

        # make figure externally,
        # put sending wavefronts left and returning wavefronts right
        # show merged data

        fig2_ind, ax2_ind = plt.subplot_mosaic([['IS','IR'],
                                                ['VS','VR']])
        make_fanout_wavefronts_all(interface.data_output_multiplicative,True, ax=ax2_ind)

        # put voltages in opposite corners (for fun)
        fig2_cap, ax2_cap = plt.subplot_mosaic([['IS','VR'],
                                                ['VS','IR']])
        make_fanout_wavefronts_all(interface.data_output_multiplicative,False, ax=ax2_cap)

        plt.show()
        
    .. warning::
        if ax keyword is not provided, function will make new subplot objects each time it is called.
        These plots will not be closed by default, so if multiple calls are needed it is suggested you provide
        the appropriate subplot_mosaic Axes object.
    """   
    default_make_kwargs : dict = {'ax':False,
                                  'fig_size':(12,10)}
    
    make_kwargs, fanout_kwargs = split_outer_inner_default_kwargs(kwargs,default_make_kwargs,default_fanout_kwargs)
    
    del fanout_kwargs['title']
    del fanout_kwargs['units']
    
    if (make_kwargs['ax'] == False):
        fig, ax = plt.subplot_mosaic([['VS','IS' ],
                                      ['VR','IR']])
    else:
        ax = make_kwargs['ax']
        fig = ax['VS'].get_figure()
    
    fig.set_size_inches(make_kwargs['fig_size'])
    
    if (is_Inductor):
    
        fig.suptitle("Inductor Wavefront Fanouts")
        plot_fanout_wavefronts(data_output,ax['VS'],"voltage inductor",True,**fanout_kwargs)
        plot_fanout_wavefronts(data_output,ax['IS'],"current inductor",True,**fanout_kwargs)
        plot_fanout_wavefronts(data_output,ax['VR'],"voltage inductor",False,**fanout_kwargs)
        plot_fanout_wavefronts(data_output,ax['IR'],"current inductor",False,**fanout_kwargs)
    else:
        
        fig.suptitle("Capacitor Wavefront Fanouts")
        plot_fanout_wavefronts(data_output,ax['VS'],"voltage capacitor",True,**fanout_kwargs)
        plot_fanout_wavefronts(data_output,ax['IS'],"current capacitor",True,**fanout_kwargs)
        plot_fanout_wavefronts(data_output,ax['VR'],"voltage capacitor",False,**fanout_kwargs)
        plot_fanout_wavefronts(data_output,ax['IR'],"current capacitor",False,**fanout_kwargs)
    
    if (make_kwargs['ax'] == False):
        return fig, ax

def plot_trace_on_merged_fanout_axis(data_output_ordered : Ordered_Output_Data, ax, upto_time :Decimal = False,**kwargs):
    """Plots a path of arrows on a merged fanout diagram.

    :param data_output_ordered: the ordered data array, can also be an interface object
    :type data_output_ordered: Ordered_Output_Data or Interface_Data
    :param ax: the axis with a fanout diagram plotted on it
    :type ax: Matplotlib Axes
    :param upto_time: the time to which the path must be plotted, defaults to False
    :type upto_time: Decimal, optional
    :**kwargs**:
        - **show_cross** (*bool*) - If a cross must be plotted to show current arrow at 'upto_time'. Default is False
        - **padding** (*int*) - The padding around the arrow. Default is 0.
        - **length_includes_head** (*bool*) - Whether the head is included in the arrow length. Default is True.
        - **head_width** (*float*) - The width of the arrow head. Default is 0.3.
        - **head_length** (*float*) - The length of the arrow head. Default is 0.3.
        - **width** (*float*) - The width of the arrow shaft. Default is 0.0005.
        - **facecolor** (*str*) - The face color of the arrow. Default is 'gray'.
        - **edgecolor** (*str*) - The edge color of the arrow. Default is 'black'.
        
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_trace_on_merged_fanout_axis, plot_fanout_interconnect
        import matplotlib.pyplot as plt

        # simulate an interface
        interface_data = generate_interface_data(L_time = '3.6',C_time = '3.2')

        fig, ax = plt.subplots()
        plot_fanout_interconnect(interface_data.data_output_multiplicative,ax,'voltage capacitor')
        plot_trace_on_merged_fanout_axis(interface_data,ax)
        plt.show()
        
    .. warning::

        the trace plotted is compatible with *merged* fanouts. (fanout plots of data_output_multiplicative) 
        
    """
    
    trace_default_kwargs : dict= {
        'show_cross' : False,
        'padding' : 0,
        'length_includes_head' : True,
        'head_width' : 0.3,
        'head_length': 0.3,
        'width': 0.0005,
        'facecolor':'gray',
        'edgecolor':'black'
        }
    
    kwargs = handle_default_kwargs(kwargs,trace_default_kwargs)
    
    # check it is an interface or data_ordered object
    data_output_ordered = handle_interface_to_ordered(data_output_ordered)
    
    # set limit of indexes based on upto_time if supplied
    if (isinstance(upto_time,bool)):
        Indexes = data_output_ordered.Indexes[1:]
    else:
        i,_ = closest_event_to_time(data_output_ordered.Time,upto_time,True)
        Indexes = data_output_ordered.Indexes[1:i]
    
    get_xy = lambda index : (index[0] + kwargs['padding'], index[1] + kwargs['padding'])
    
    last_x = 0
    last_y = 0
    
    # plot arrow from last to next
    for i,index in enumerate(Indexes):
        
        x1,y1 = get_xy(data_output_ordered.Indexes[i])
        x2,y2 = get_xy(index)

        dx = x2 - x1
        dy = y2 - y1

        # Draw the arrow using quiver
        ax.arrow(x1,y1,dx, dy,
                 length_includes_head=kwargs['length_includes_head'], 
                 head_width=kwargs['head_width'], 
                 head_length=kwargs['head_length'],
                 width=kwargs['width'],
                 facecolor = kwargs['facecolor'],
                 edgecolor =kwargs['edgecolor'])
        
        last_x = x2
        last_y = y2
        
    if(kwargs['show_cross']):
        ax.axhline(y=last_y,c='k')
        ax.axvline(x=last_x,c='k')

def plot_merging_lines_on_fanout(array_to_plot: np.ndarray , KL :int, KC: int, ax,**kwargs):
    """Plots the borders of merging regions for a given array_to_plot onto an axis that is plotting a fanout.

    :param array_to_plot: The data array contianing fanout magnitude information
    :type array_to_plot: np.ndarray
    :param KL: Inductor LCM factor
    :type KL: int
    :param KC: Capacitor LCM factor
    :type KC: int
    :param ax: Axis with a fanout plot on it
    :type ax: matplotlib Axes object
    :**kwargs**:
        - **transpose** (*bool*) - Whether the plot is transposed (L-axis is horizontal axis). Default is True
        - **padding** (*int*) - The padding of the plot. Default is 0
        - **line_colour** (*str*) - The color of the lines. Default is 'k'
        - **line_width** (*float*) - The width of the lines. Default is 0.5
    """
    
    default_kwargs = {
        'transpose' : True,
        'padding' : 0,
        'line_colour' : 'k',
        'line_width' : 0.5
    }
    
    kwargs = handle_default_kwargs(kwargs,default_kwargs)
    
    # get index limits of L and C axes 
    L_max, C_max = array_to_plot.shape
    
    number_KL = math.floor(L_max/KL) +1
    number_KC = math.floor(C_max/KC) +1
    
    for i in range (number_KL):
        if(kwargs['transpose']): # L-axis is horizontal 
            line_position = i*KL - 0.5 + kwargs['padding']
            ax.axvline(x=line_position, linewidth=kwargs['line_width'],c=kwargs['line_colour'])
        else: 
            ax.axhline(y=line_position, linewidth=kwargs['line_width'],c=kwargs['line_colour'])
            
    for i in range (number_KC):
        if(kwargs['transpose']): # C-axis is vertical
            line_position = i*KC - 0.5 + kwargs['padding']
            ax.axhline(y=line_position, linewidth=kwargs['line_width'],c=kwargs['line_colour'])
        else: 
            ax.axvline(x=line_position, linewidth=kwargs['line_width'],c=kwargs['line_colour'])

def make_commutative_merged_lines(interface_data : Interface_Data ,which_operation : str ,which_string : str):
    """Make 3 - magnitude fanouts with their merging regions shown.
    Fanout 1 is the commutative fanout before merging.
    Fanout 2 is the merged fanout along the L-axis.
    Fanout 3 is the merged fanout along the C-axis.

    :param interface_data: the interface data to be plotted
    :type interface_data: Interface_Data
    :param which_operation: the operation for fetching fanout data, options are 'interconnect','sending' or 'returning'.
    :type which_operation: str
    :param which_string: which specific magnitude to extract in form '{voltage or current} {inductor or capacitor}'
    :type which_string: str
    :raises ValueError: if incorrect 'which_operation' or 'which_string' information is provided.
    """
    
    fig, ax = plt.subplots(1,3)
    allowed_operations = ['interconnect','sending','returning']
    
    if (which_operation.lower() == allowed_operations[0]):
        commutative_array = interface_data.data_output_commutative.get_interconnect_array(which_string)
        merged_array = interface_data.data_output_multiplicative.get_interconnect_array(which_string)
    elif(which_operation.lower() == allowed_operations[1]):
        commutative_array = interface_data.data_output_commutative.get_sending_wavefronts_magnitudes(which_string)
        merged_array = interface_data.data_output_multiplicative.get_sending_wavefronts_magnitudes(which_string)
    elif(which_operation.lower() == allowed_operations[3]):
        commutative_array = interface_data.data_output_commutative.get_returning_wavefronts_magnitudes(which_string)
        merged_array = interface_data.data_output_multiplicative.get_returning_wavefronts_magnitudes(which_string)
    else:
        raise ValueError(f'the provided which_operation paramters was incorrect, possible options are : {allowed_operations}')
        
    C_axis_merged_array = transform_merged_array_to_capacitor_axis(interface_data.data_input, merged_array)
    
    if (which_string.find('voltage') > 0 ):
        units = 'V'
    else:
        units = 'A'

    padding =10

    plot_fanout_magnitude(commutative_array,ax[0], show_colour_bar = False, show_ticks = True)
    plot_fanout_magnitude(merged_array,ax[1], padding=padding, show_colour_bar = False, show_ticks = True)
    plot_fanout_magnitude(C_axis_merged_array,ax[2], units = units,padding=padding, show_ticks = True)

    ax[0].set_title(which_operation+' '+which_string)
    ax[1].set_title('Merged on L-axis')
    ax[2].set_title('Merged on C-axis')

    KL = interface_data.data_input.Inductor_LCM_Factor
    KC = interface_data.data_input.Capacitor_LCM_Factor

    plot_merging_lines_on_fanout(commutative_array,KL,KC,ax[0])
    plot_merging_lines_on_fanout(merged_array,KL,KC,ax[1],padding=padding)
    plot_merging_lines_on_fanout(C_axis_merged_array,KL,KC,ax[2],padding=padding)
    

def plot_time_interconnect(data_output_ordered : Ordered_Output_Data,ax, which_string :str, is_integrated: bool = True,**kwarg): 
    """Plots the time waveform of one of the interconncet metrics. 
    It must be noted that interconnect values stored in the :Ordered_Output_Data: object signify the 'change' in interface values due to wavefronts.
    To see the full time wavefrom, the changes must be accumulated. This function shows both change and accumulated quantities. 

    :param data_output_ordered: The data object containing 1D ordered simulation data
    :type data_output_ordered: Ordered_Output_Data or (Interface_Data)
    :param ax: The axis on which the interconncet wavefrom will be plotted.
    :type ax: Matplotlib Axes object
    :param which_string: The interconnect value to be plotted, options are "voltage inductor", "current inductor", "voltage capacitor" and "current capacitor"
    :type which_string: str
    :param is_integrated: If the wavefrom must represent the 'change' or 'accumulation of changes' of the data selected to be plotted, default is True
    :type is_integrated: bool
    :raises ValueError: if an incorrect which_string is provided.
    :return: (optional) if key word 'return_data = True' is passed, will return the plotted array, default is False
    :rtype: np.ndarray[Decimal]
    
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_time_interconnect
        import matplotlib.pyplot as plt

        # simulate interface
        interface = generate_interface_data(L_time='0.34' , C_time='0.12', L_impedance = '700', C_impedance = '7')

        # Make axes 
        fig,ax = plt.subplots(2,1,figsize=(8,8))

        # make a handle for ordered data (very optional)
        data = interface.data_output_ordered

        # plot accumulated data on ax[0]
        plot_time_interconnect(data,ax[0],'current capacitor',True)

        # plot change data on ax[1], use 'interface' instead of 'data' (for fun)
        plot_time_interconnect(interface,ax[1],'current capacitor',False)

        plt.show()

    .. warning::
    
        This function accepts only :py:class:`storage.Ordered_Output_Data` as an input. The data is required to be 1D and ordered.
    
    """
    default_kwarg = {'return_data' : False}
    kwarg = handle_default_kwargs(kwarg, default_kwarg)
    
    # make interface_data -> ordered_data
    data_output_ordered = handle_interface_to_ordered(data_output_ordered)
    
    # get data
    data_to_plot = data_output_ordered.get_interconnect_array(which_string)
    
    # Format axes
    allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
    if(which_string.lower() == allowed_strings[0] or which_string.lower() == allowed_strings[1]):
        transmisson_line_str = 'Inductor '
    else:
        transmisson_line_str = 'Capacitor '
        
    if(which_string.lower() == allowed_strings[0] or which_string.lower() == allowed_strings[2]):
        magnitude_str = 'voltage '
        ax.yaxis.set_major_formatter(EngFormatter('V'))
    else:
        magnitude_str = 'current '
        ax.yaxis.set_major_formatter(EngFormatter('A'))
        
    if(is_integrated):
        data_to_plot = np.cumsum(data_to_plot)
        operation_str = ''
    else:
        operation_str = 'change'
    
    ax.set_title(transmisson_line_str + magnitude_str + operation_str + ' at Interconnect')
    ax.xaxis.set_major_formatter(EngFormatter('s'))
    
    ax.step(data_output_ordered.Time,data_to_plot,where='post')
    
    if(kwarg['return_data']):
        return data_to_plot

def plot_time_wavefronts(data_output_ordered : Ordered_Output_Data,ax, which_string :str, is_sending :bool,is_integrated: bool = True ): 
    """Plots the time waveform of one of the wavefront metrics. 
    It must be noted that interconnect values stored in the `Ordered_Output_Data` object signify the 'change' in interface values due to wavefronts.
    To see the full time wavefrom, the changes must be accumulated. This function shows both change and accumulated quantities. 

    :param data_output_ordered: The data object containing 1D ordered simulation data, also accepts full interface data
    :type data_output_ordered: Ordered_Output_Data or Interface_Data
    :param ax: The axis on which the interconncet wavefrom will be plotted.
    :type ax: Matplotlib Axes object
    :param which_string: The wavefront value to be plotted, options are "voltage inductor", "current inductor", "voltage capacitor" and "current capacitor"
    :type which_string: str
    :param is_sending: If the the wavefront data shown must be for sending or returning wavefronts.
    :type is_sending: bool
    :param is_integrated: If the wavefrom must represent the 'change' or 'accumulation of changes' of the data selected to be plotted, default is True
    :type is_integrated: bool
    :raises ValueError: if an incorrect which_string is provided.
    
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_time_wavefronts
        import matplotlib.pyplot as plt

        # Example, comparing the sending and returning current wavefronts in the capacitor:
        # =================================================================================

        # simulate interface
        interface = generate_interface_data(L_time='3' , C_time='7', L_impedance = '700', C_impedance = '7')
        data = interface.data_output_ordered

        # Make axes 
        fig,ax = plt.subplots()

        # plot sending wavefronts (not accumulated)
        plot_time_wavefronts(data,ax,'current capacitor',True,False)

        # plot returning wavefronts (not accumulated)
        plot_time_wavefronts(data,ax,'current capacitor',False,False)

        plt.show()
        

    .. warning::
    
        This function accepts only :py:class:`storage.Ordered_Output_Data` as an input. The data is required to be 1D and ordered.
    
    """
    data_output_ordered = handle_interface_to_ordered(data_output_ordered)
    
    allowed_strings = ["voltage inductor", "current inductor", "voltage capacitor", "current capacitor"]
    
    if (is_sending):
        title_prefix = "Sending "
        get_func = data_output_ordered.get_sending_wavefronts_magnitudes
    else:
        title_prefix = "Returning "
        get_func = data_output_ordered.get_returning_wavefronts_magnitudes
    
    ax.xaxis.set_major_formatter(EngFormatter('s'))
    
    if(is_integrated):
        if(which_string.lower() == allowed_strings[0] ):
            ax.set_title(title_prefix + "voltage wavefronts\n accumulated in Inductor")
            ax.yaxis.set_major_formatter(EngFormatter('V'))
        elif(which_string.lower() == allowed_strings[1] ):
            ax.set_title(title_prefix + "current wavefronts\n accumulated in Inductor")
            ax.yaxis.set_major_formatter(EngFormatter('A'))
        elif(which_string.lower() == allowed_strings[2] ):
            ax.set_title(title_prefix + "voltage wavefronts\n accumulated in Capacitor")
            ax.yaxis.set_major_formatter(EngFormatter('V'))
        elif(which_string.lower() == allowed_strings[3] ):
            ax.set_title(title_prefix + "current wavefronts\n accumulated in Capacitor")
            ax.yaxis.set_major_formatter(EngFormatter('A'))
        else:
            raise ValueError(f"Incorrect plotting choice /, {which_string} is not a valid option. Optiond are: \n {allowed_strings}")
        
        ax.step(data_output_ordered.Time,np.cumsum(get_func(which_string)),where='post')
    else:
        if(which_string.lower() == allowed_strings[0] ):
            ax.set_title(title_prefix + "voltage wavefronts\n in Inductor")
            ax.yaxis.set_major_formatter(EngFormatter('V'))
        elif(which_string.lower() == allowed_strings[1] ):
            ax.set_title(title_prefix + "current wavefronts\n in Inductor")
            ax.yaxis.set_major_formatter(EngFormatter('A'))
        elif(which_string.lower() == allowed_strings[2] ):
            ax.set_title(title_prefix + "voltage wavefronts\n in Capacitor")
            ax.yaxis.set_major_formatter(EngFormatter('V'))
        elif(which_string.lower() == allowed_strings[3] ):
            ax.set_title(title_prefix + "current wavefronts\n in Capacitor")
            ax.yaxis.set_major_formatter(EngFormatter('A'))
        else:
            raise ValueError(f"Incorrect plotting choice /, {which_string} is not a valid option. Optiond are: \n {allowed_strings}")
        
        ax.step(data_output_ordered.Time,get_func(which_string),where='post')

def make_time_interconnect_all(data_output_ordered: Ordered_Output_Data,is_integrated :bool = True,**kwargs):
    """Plots all interconnect time waveforms of an interface/ orderd data.

    :param data_output_ordered: data to be plotted. Can be interface or ordered data.
    :type data_output_ordered: Ordered_Output_Data
    :param is_integrated: If the wavefrom must represent the 'change' or 'accumulation of changes' of the data selected to be plotted, default is True
    :type is_integrated: bool, optional
    :return: the matplotlib Figure and Axes objects created in this function (if created)
    :rtype: tuple( fig , ax )
    :**kwargs for figure creation**:
        - **ax** (*Dict(Axes)*) - Whether to create a subpot or use exsiting subplot axes.If left blank default is 'False' and subplot is created internally.If axes are provided, the must be of a matplotlib.pyplot.subplot_mosaic() form.The labels for these axes must inculde: 
            - 'VL' axis for inductor voltage
            - 'VC' axis for capcitor voltage
            - 'IL' axis for inductor current
            - 'IC' axis for capacitor current
        - **fig_size** (*tuple of ints*) - The size of the figure. Default is (10, 8).
        
    .. code-block::

        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import make_time_interconnect_all
        import matplotlib.pyplot as plt

        # simulate interface
        interface = generate_interface_data(L_time='8' , C_time='7', L_impedance = '500', C_impedance = '2')

        # plot all interconnect time waveforms
        fig,ax = make_time_interconnect_all(interface)

        # plot the 'change' in those waveforms
        fig2,ax2 = make_time_interconnect_all(interface,False)

        plt.show()
        
    """
    
    data_output_ordered = handle_interface_to_ordered(data_output_ordered)
    
    default_make_kwargs : dict = {'ax':False,
                                  'fig_size':(10,8)}
    
    make_kwargs = handle_default_kwargs(kwargs,default_make_kwargs)
    
    if (make_kwargs['ax'] == False):
        fig, ax = plt.subplot_mosaic([['VL','VC'],
                                      ['IL','IC']])
    else:
        ax = make_kwargs['ax']
        fig = ax['VL'].get_figure()
        
    fig.set_size_inches(make_kwargs['fig_size'])
    
    if (is_integrated):
        fig.suptitle(f"Interconnect Time Waveforms")
    else:
        fig.suptitle(f"Interconnect Change Time Waveforms")

    plot_time_interconnect(data_output_ordered,ax['VL'],"Voltage Inductor", is_integrated)
    plot_time_interconnect(data_output_ordered,ax['IL'],"Current Inductor", is_integrated)
    plot_time_interconnect(data_output_ordered,ax['VC'],"Voltage Capacitor", is_integrated)
    plot_time_interconnect(data_output_ordered,ax['IC'],"Current Capacitor", is_integrated)
    
    if (make_kwargs['ax'] == False):
        return fig,ax

def make_time_wavefronts_all(data_output_ordered : Ordered_Output_Data, is_Inductor :bool,is_integrated: bool = True, **kwargs ):
    """Plots the wavefront time beahviour of a particualr transmission line. 
    Both sending and returning, current and voltage wavefront time behaviour is shown

    :param data_output_ordered:  data to be plotted. Can be interface or ordered data.
    :type data_output_ordered: Ordered_Output_Data or Interface_Data
    :param is_Inductor: if the inductor or capacitor wavefronts must be plot.
    :type is_Inductor: bool
    :param is_integrated: if the individual wavefront value or an accumulation of these values msut be shown, defaults to True
    :type is_integrated: bool, optional
    :return: the matplotlib Figure and Axes objects created in this function (if created)
    :rtype: tuple( fig , ax )
    :**kwargs for figure creation**:
        - **ax** (*Dict(Axes)*) - Whether to create a subpot or use exsiting subplot axes.If left blank default is 'False' and subplot is created internally.If axes are provided, the must be of a matplotlib.pyplot.subplot_mosaic() form.The labels for these axes must inculde: 
            - 'VS' axis for sending voltage
            - 'VR' axis for returning voltage
            - 'IS' axis for sending current
            - 'IR' axis for returning current
        - **fig_size** (*tuple of ints*) - The size of the figure. Default is (10, 8).
        
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import make_time_wavefronts_all
        import matplotlib.pyplot as plt

        # Example: the accumulated wavefront behaviour over time between the capacitor and inductor
        # ==========================================================================================

        # simulate interface
        interface = generate_interface_data(L_time='7' , C_time='3.4', L_impedance = '654', C_impedance = '2.5')

        # plot accumulation wavefront activity for inductor
        fig,ax = make_time_wavefronts_all(interface,True,True)

        # plot the accumulation wavefront activity for capacitor
        # here we just pass the ax object as a kwarg so that it is plotted on the same axes
        make_time_wavefronts_all(interface,False,True,ax=ax)

        # rename the auto generated suptitle
        fig.suptitle('Comparison between accumulated wavefronts over time in each transmission line')

        # use the key word to set titles of each axis independantly 
        ax['VS'].set_title('Sending Voltage Wavefronts')
        ax['VR'].set_title('Returning Voltage Wavefronts')
        ax['IS'].set_title('Sending Current Wavefronts')
        ax['IR'].set_title('Returning Current Wavefronts')

        # plot a legend for all axes
        for ax_i in ax.values(): 
            ax_i.legend(['Inductor', 'Capacitor'])

        plt.show()
    """
    
    
    data_output_ordered = handle_interface_to_ordered(data_output_ordered)

    default_make_kwargs : dict = {'ax':False,
                                  'fig_size':(12,10)}
    
    make_kwargs = handle_default_kwargs(kwargs, default_make_kwargs)
    
    if (make_kwargs['ax'] == False):
        fig, ax = plt.subplot_mosaic([['VS','IS' ],
                                      ['VR','IR']])
    else:
        ax = make_kwargs['ax']
        fig = ax['VS'].get_figure()
    
    fig.set_size_inches(make_kwargs['fig_size'])
    
    if (is_Inductor):
        title_prefix = 'Inductor'
        plot_time_wavefronts(data_output_ordered,ax['VS'],"voltage inductor",True, is_integrated)
        plot_time_wavefronts(data_output_ordered,ax['IS'],"current inductor",True, is_integrated)
        plot_time_wavefronts(data_output_ordered,ax['VR'],"voltage inductor",False, is_integrated)
        plot_time_wavefronts(data_output_ordered,ax['IR'],"current inductor",False, is_integrated)
    else:
        title_prefix = 'Capacitor'
        plot_time_wavefronts(data_output_ordered,ax['VS'],"voltage capacitor",True, is_integrated)
        plot_time_wavefronts(data_output_ordered,ax['IS'],"current capacitor",True, is_integrated)
        plot_time_wavefronts(data_output_ordered,ax['VR'],"voltage capacitor",False, is_integrated)
        plot_time_wavefronts(data_output_ordered,ax['IR'],"current capacitor",False, is_integrated)
    
    fig.suptitle(title_prefix + " Wavefronts Time Behaviour")
    
    if (make_kwargs['ax'] == False):
        return (fig, ax)

def plot_refelction_diagram(interface_data : Interface_Data, ax, is_voltage : bool, **kwargs):
    """plots a coloured current or voltage reflection for the inductor and capacitor of a simulated interface.

    :param interface_data: The interface to be plotted.
    :type interface_data: Interface_Data
    :param ax: axis to plot on
    :type ax: Matplotlib Axes 
    :param is_voltage: if the plot
    :type is_voltage: bool
    :raises TypeError: if supplied custom_colour_bar are not a tuple 
    :**kwargs for figure creation**:
        - **stop_time** (*float*) - The simulation stop time. Default is the value of `interface_data.data_input.Simulation_Stop_Time`.
        - **custom_colour_bar_limits** (*tuple*) - supply a tuple in form of (Vmax,Vmin) for colourbar limits. Default is False, meaning it is calcuated of absolute maximum of wavefronts.
        - **face_colour** (*str*) - The face color of the plot. Default is 'xkcd:grey'.
        - **LS_colour** (*bool or str*) - if supplied overides colour map colouring, The color of the sending inductor wavefronts, matplotlib colour. Default is False.
        - **LR_colour** (*bool or str*) - if supplied overides colour map colouring, The color of the returning inductor wavefronts, matplotlib colour. Default is False.
        - **CS_colour** (*bool or str*) - if supplied overides colour map colouring, The color of the sending capacitor wavefronts, matplotlib colour. Default is False.
        - **CR_colour** (*bool or str*) - if supplied overides colour map colouring, The color of the returning capacitor wavefronts, matplotlib colour. Default is False.
        - **LS_style** (*str*) - The style of the sending inductor wavefronts, matplotlib linestyle. Default is '-'.
        - **LR_style** (*str*) - The style of the returning inductor wavefronts, matplotlib linestyle. Default is '-'.
        - **CS_style** (*str*) - The style of the sending capacitor wavefronts, matplotlib linestyle. Default is '-'.
        - **CR_style** (*str*) - The style of the returning capacitor wavefronts, matplotlib linestyle. Default is '-'.
        - **info_title** (*bool*) - Whether to include a title with input information about the plot. Default is True.

    .. code-block::
        :caption: Compare Voltage and Current wavefronts of the interface
        
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_refelction_diagram
        import matplotlib.pyplot as plt

        # simulate interface
        interface = generate_interface_data(L_time = '12',C_time = '11',Simulation_stop_time=100)

        # create subplot
        fig,ax = plt.subplots(1,2,figsize=(18,8))

        # compare voltage and current 
        plot_refelction_diagram(interface,ax[0],True)
        plot_refelction_diagram(interface,ax[1],False)

        plt.show()
        
    .. code-block::
        :caption: Customizing plots to highlight sending wavefronts
        
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_refelction_diagram
        import matplotlib.pyplot as plt

        # simulate interface
        interface = generate_interface_data(L_time = '12',C_time = '11',Simulation_stop_time=100)

        # create subplot
        fig,ax = plt.subplots(1,2,figsize=(18,8))

        # highlight sending wavefronts and make returning gray
        c = 'dimgray'
        plot_refelction_diagram(interface,ax[0],True, CR_colour=c, CR_style = '--', LR_colour=c, LR_style = '--')
        plot_refelction_diagram(interface,ax[1],False, CR_colour=c, CR_style = '--', LR_colour=c, LR_style = '--')

        plt.show()
    """
    
    default_kwargs = {
        'stop_time' : interface_data.data_input.Simulation_Stop_Time,
        'custom_colour_bar_limits': False,
        'face_colour':'xkcd:grey',
        'LS_colour':False,
        'LR_colour':False,
        'CS_colour':False,
        'CR_colour':False,
        'LS_style':'-',
        'LR_style':'-',
        'CS_style':'-',
        'CR_style':'-',
        'info_title':True,
        }
    
    kwargs = handle_default_kwargs(kwargs,default_kwargs)
    
    inductor_sending = interface_data.data_output_ordered.Wavefronts_Sending_Inductor
    inductor_returning = interface_data.data_output_ordered.Wavefronts_Returning_Inductor
    capacitor_sending = interface_data.data_output_ordered.Wavefronts_Sending_Capacitor
    capacitor_returning = interface_data.data_output_ordered.Wavefronts_Returning_Capacitor
    
    if (is_voltage):
        title = 'Voltage Reflection Diagram'
        get_func = get_voltage_array
        get_val = get_voltage_from_wavefront
        units = 'V'
    else:
        title = 'Current Reflection Diagram'
        get_func = get_current_array
        units = 'A'
        get_val = get_current_from_wavefront
        
    inductor_sending_magnitude = get_func(inductor_sending)
    inductor_returning_magnitude = get_func(inductor_returning)
    capacitor_sending_magnitude = get_func(capacitor_sending)
    capacitor_returning_magnitude = get_func(capacitor_returning)
    
    if (kwargs['custom_colour_bar_limits'] == False):
        inductor_sending_max = get_array_absolute_maximum(inductor_sending_magnitude)
        inductor_returning_max = get_array_absolute_maximum(inductor_returning_magnitude)
        capacitor_sending_max = get_array_absolute_maximum(capacitor_sending_magnitude)
        capacitor_returning_max = get_array_absolute_maximum(capacitor_returning_magnitude)
        
        colour_bar_boundary_max = float(max(inductor_sending_max,inductor_returning_max,capacitor_sending_max,capacitor_returning_max))
        colour_bar_boundary_min = - colour_bar_boundary_max
        
    elif(isinstance(kwargs['custom_colour_bar_limits'],tuple)):
        colour_bar_boundary_max = kwargs['custom_colour_bar_limits'][0]
        colour_bar_boundary_min = kwargs['custom_colour_bar_limits'][1]
    else:
        raise TypeError(f"custom_colour_bar_limits must be of type tuple not {type(kwargs['custom_colour_bar_limits'])}")
        
    colour_map = mpl.cm.seismic
    norm = mpl.colors.Normalize(vmin=colour_bar_boundary_min, vmax=colour_bar_boundary_max)
    colour_mag_func = lambda x : colour_map(norm(float(get_val(x))))
    
    if(kwargs['LS_colour']==False):
        LS_colour = colour_mag_func
    else:
        LS_colour = lambda x : kwargs['LS_colour']
        
    if(kwargs['LR_colour']==False):
        LR_colour = colour_mag_func
    else:
        LR_colour = lambda x : kwargs['LR_colour']
        
    if(kwargs['CS_colour']==False):
        CS_colour = colour_mag_func
    else:
        CS_colour = lambda x : kwargs['CS_colour']
        
    if(kwargs['CR_colour']==False):
        CR_colour = colour_mag_func
    else:
        CR_colour = lambda x : kwargs['CR_colour']
    
    index_stop, stop_time_closest = closest_event_to_time(interface_data.data_output_ordered.Time,kwargs['stop_time'])
    
    for LS, LR, CS, CR in zip(inductor_sending[:index_stop],inductor_returning[:index_stop],capacitor_sending[:index_stop], capacitor_returning[:index_stop]):
        
        ax.plot([LS.position_start, LS.position_end],[LS.time_start,LS.time_end],c=LS_colour(LS), linestyle = kwargs['LS_style'])
        ax.plot([LR.position_start, LR.position_end],[LR.time_start,LR.time_end],c=LR_colour(LR), linestyle = kwargs['LR_style'])

        ax.plot([CS.position_start, -CS.position_end],[CS.time_start,CS.time_end],c=CS_colour(CS), linestyle = kwargs['CS_style'])
        ax.plot([-CR.position_start, CR.position_end],[CR.time_start,CR.time_end],c=CR_colour(CR), linestyle = kwargs['CR_style'])
        
    
    # Styling
    L_l = float(interface_data.data_input.Inductor_Length)
    C_l = float(interface_data.data_input.Capacitor_Length)
    
    ax.set_facecolor(kwargs['face_colour'])
    ax.set_xlim([-C_l,L_l])
    ax.set_xticks([-1*C_l,-0.5*C_l,0,0.5*L_l,1*L_l])
    ax.set_xticklabels(["$\mathregular{\ell_C}$","Capacitor","Interface","Inductor","$\mathregular{\ell_L}$"],fontsize='large')
    ax.axvline(0, color = 'k', linestyle = '--', linewidth=2)
    
    
    ax.set_ylim([0,stop_time_closest])
    ax.yaxis.set_major_formatter(EngFormatter('s'))
    
    fig = ax.get_figure()
    
    ZL = interface_data.data_input.Inductor_Impedance
    TL = interface_data.data_input.Inductor_Time*2
    
    ZC = interface_data.data_input.Capacitor_Impedance
    TC = interface_data.data_input.Capacitor_Time*2
    
    if(kwargs['info_title']):
        title += f"\n ZC = {ZC}Ω TC = {TC}s, ZL = {ZL}Ω TL = {TL}s, "
    ax.set_title(title)
    
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colour_map), ax=ax)
    cb.ax.yaxis.set_major_formatter(EngFormatter(units))

def make_spatial_voltage_and_current(Time_Enquriey : Decimal , Interface : Interface_Data, **kwargs):
    """ Plots the spatial distribution of voltage and current in both the inductor and capacitor.

    :param Time_Enquriey: the time at which spatial distrinution of energy is shown. 
    :type Time_Enquriey: Decimal
    :param Interface: the data storage object for the interface simulation
    :type Interface: Interface_Data
    :return: interconnect values of voltage for capacitor and inductor and current for capacitor and inductor in that order if 'return-data' keyword set to True, default is False
    :rtype: tuple ( Decimal[VC], Decimal[VL], Decimal[IC], Decimal[IL] )
    :**kwargs for figure creation**:
        - **ax** (*Dict(Axes)*) - Whether to create a subpot or use exsiting subplot axes.If left blank default is 'False' and subplot is created internally.If axes are provided, the must be of a matplotlib.pyplot.subplot_mosaic() form or a 1D np.ndarray of two items. The first will be assigned voltage and the other current.The labels for these axes must inculde: 
            - 'V' axis for voltage spatial plot
            - 'I' axis for current spatial plot
        - **fig_size** (*tuple of ints*) - The size of the figure. Default is (12,10).
        - **quantize** (*str or Decimal*) - the precision to round the input time shown in the title
        - **return_data** (*bool*) - if the interconnect values must be returned or not, default is False
    
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import make_spatial_voltage_and_current
        import matplotlib.pyplot as plt
        from decimal import Decimal

        # simulate an interface
        interface_data = generate_interface_data(L_time = '2.07',C_time = '3.2')

        # investgate the spatial waveforms a t = 30.573
        make_spatial_voltage_and_current(Decimal('30.573'),interface_data)

        # this time we will pass an axes dict, put current at the top:
        # notice the correct formatting of ['V'] and ['I']
        fig,ax = plt.subplot_mosaic([['I'],
                                    ['V']])

        # investgate the spatial waveforms a t = 30.7
        make_spatial_voltage_and_current(Decimal('30.7'),interface_data,ax=ax)

        plt.show()
        
    .. warning::

        if you do not pass an axes object the function will make a new suplot at each call. 
        This means that if you plan to run the function such that it called multiple timea, like a loop, 
        it is advised to pass axes object to avoid uneccassary creation of supblots each interation.
    """
    Time_Enquriey = Decimal(Time_Enquriey)
    
    default_make_kwargs : dict = {'ax':False,
                                  'fig_size':(12,10),
                                  'quantize':'0.001',
                                  'return_data':False}
    
    kwargs = handle_default_kwargs(kwargs,default_make_kwargs)
    
    if (isinstance(kwargs['ax'],bool)): # not provided make axes
        fig, ax = plt.subplot_mosaic([[  'V'  ],
                                      [  'I'  ]])
    elif isinstance(kwargs['ax'],dict): # if mosaic use
        ax = kwargs['ax']
        fig = ax[ 'V' ].get_figure()
    elif isinstance(kwargs['ax'],np.ndarray): # else make dict
        ax = {}
        ax['V'] = kwargs['ax'][0]
        ax['I'] = kwargs['ax'][1]
        fig = ax[ 'V' ].get_figure()
    else:
        raise TypeError(f"axes object provided is not of type dict or numpy.ndarray, was instead of type {type(kwargs['ax'])}")
    
    fig.set_size_inches(kwargs['fig_size'])
    
    # Set title
    formatter =  EngFormatter('s')
    ax['V'].get_figure().suptitle(f"Spatial Waveforms at {formatter.format_eng(float(Time_Enquriey))}s")
    
    # Set axes
    L_l = float(Interface.data_input.Inductor_Length)
    C_l = float(Interface.data_input.Capacitor_Length)
    
    ax['V'].set_title('Spatial Voltage')
    ax['V'].yaxis.set_major_formatter(EngFormatter('V'))
    ax['V'].set_xlim([-C_l,L_l])
    ax['V'].set_xticks([-1*C_l,-0.5*C_l,0,0.5*L_l,1*L_l])
    ax['V'].set_xticklabels(["$\mathregular{\ell_C}$","Capacitor","Interface","Inductor","$\mathregular{\ell_L}$"],fontsize='large')
    ax['V'].set_ylabel('voltage')
    
    ax['I'].set_title('Spatial Current')
    ax['I'].yaxis.set_major_formatter(EngFormatter('A'))
    ax['I'].set_xlim([-C_l,L_l])
    ax['I'].set_xticks([-1*C_l,-0.5*C_l,0,0.5*L_l,1*L_l])
    ax['I'].set_xticklabels(["$\mathregular{\ell_C}$","Capacitor","Interface","Inductor","$\mathregular{\ell_L}$"],fontsize='large')
    ax['I'].set_ylabel('current')

    # Inductor
    # Get spatial intercepts for inductor
    pos_all, value_lv, value_rv, value_lc, value_rc = get_spatial_voltage_current_at_time(Time_Enquriey, Interface, True)
    zip_out = zip(pos_all, value_lv, value_rv, value_lc, value_rc)

    # set interconnect to zero
    interconncet_voltage_inductor = 0
    interconncet_voltage_capacitor = 0
    interconnect_current_inductor = 0
    interconnect_current_capacitor = 0
    
    #arrays
    x_position =[]
    y_current =[]
    y_voltage =[]
    
    dx_position =[]
    dy_current =[]
    dy_voltage =[]

    #initiate variables
    x_old = 0
    y_voltage_old = 0
    y_current_old = 0

    is_first = True

    for (position, left_voltage, right_voltage, left_current, right_current) in zip_out:
            
            x = float(position)
            
            x_position.append(x)
            y_current.append(0)
            y_voltage.append(0)
            
            if(is_first):
                    x_old = position
                    y_voltage_old = left_voltage
                    y_current_old = left_current
                    interconncet_voltage_inductor = left_voltage
                    interconnect_current_inductor = left_current
                    is_first = False
            else:
                    dx_position.append(x - x_old)
                    dy_current.append(y_current_old)
                    dy_voltage.append(y_voltage_old)
                    
            x_old = x
            y_voltage_old = right_voltage
            y_current_old = right_current
            

    x_position.pop()
    y_current.pop()
    y_voltage.pop()
    
    # Spatially plot the inductor current and voltage
    ax['V'].bar(x_position, dy_voltage, dx_position, align = 'edge',edgecolor = 'k')
    ax['I'].bar(x_position, dy_current, dx_position, align = 'edge',edgecolor = 'k')
    
    # Now the capacitor
    # reset arrays 
    x_position =[]
    y_current =[]
    y_voltage =[]

    dx_position =[]
    dy_current =[]
    dy_voltage =[]

    is_first = True

    # get spatial interconnects for capacitor
    pos_all, value_lv, value_rv, value_lc, value_rc = get_spatial_voltage_current_at_time(Time_Enquriey, Interface , False)
    zip_out = zip(pos_all, value_lv, value_rv, value_lc, value_rc)
    
    for (position, left_voltage, right_voltage, left_current, right_current) in zip_out:
            
            x = -float(position)

            x_position.append(x)
            y_current.append(0)
            y_voltage.append(0)
            
            if(is_first):
                    x_old = position
                    y_voltage_old = left_voltage
                    y_current_old = left_current
                    interconncet_voltage_capacitor = left_voltage
                    interconnect_current_capacitor = left_current
                    is_first = False
            else:
                    dx_position.append(x - x_old)
                    dy_current.append(y_current_old)
                    dy_voltage.append(y_voltage_old)
            
            x_old = x
            
            y_voltage_old = right_voltage
            y_current_old = right_current

    x_position.pop()
    y_current.pop()
    y_voltage.pop()
    
    # plot capacitor voltages and currents
    ax['V'].bar(x_position, dy_voltage, dx_position, align = 'edge',edgecolor = 'k')
    ax['I'].bar(x_position, dy_current, dx_position, align = 'edge',edgecolor = 'k')
    
    # return interconnect values
    if(kwargs['return_data']) :
        return interconncet_voltage_capacitor, interconncet_voltage_inductor, interconnect_current_capacitor, interconnect_current_inductor

def plot_time_interconnect_and_intercepts_at_time(Time_Enquriey : Decimal, data_output_ordered : Ordered_Output_Data ,**kwargs):
    """plots all the interconnect voltages and/or currents of the tansmission lines on two sperate axes, one axis for voltage and one for current.
    Shows the magnitude of the interconnect values at a particualr time intercept as horizontla lines. 
    Combined with :py:func:`make_spatial_voltage_and_current` to make py:func:`spatial_interconnect_investigator`,
    which has an interactive form using ipywidgets, py:func:`interactive.interact_spatial`.
    See code-block bellow.

    :param Time_Enquriey:
    :type Time_Enquriey: Decimal
    :param data_output_ordered: ordered data, can also be interface data
    :type data_output_ordered: Ordered_Output_Data or Interface_Data
    :**kwargs**:
        - **ax_voltage** (*axis or bool*) - the axis to plot the voltage on, leave empty to not plot. Default is False.
        - **ax_current** (*axis or bool*) - the axis to plot the current on, leave empty to not plot. Default is False.
    
    
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import plot_time_interconnect_and_intercepts_at_time
        import matplotlib.pyplot as plt
        from decimal import Decimal

        # simulate interface
        interface_data = generate_interface_data(L_time='0.7' , C_time='3.2')

        time_enquirey = Decimal('25')

        # plot both voltage and current

        fig_both, ax_both = plt.subplots(2,1)
        # define axes with kwargs 'ax_voltage=' and 'ax_current='
        plot_time_interconnect_and_intercepts_at_time(time_enquirey,interface_data,
                                                    ax_voltage = ax_both[0],
                                                    ax_current = ax_both[1])

        # lets plot just the voltage this time, also we will progress the time enquirey
        # we will leave out 'ax_current='
        fig_single,ax_single = plt.subplots()
        time_enquirey += Decimal('5')
        plot_time_interconnect_and_intercepts_at_time(time_enquirey,interface_data,ax_voltage = ax_single)

        plt.show()
        
    """
    
    default_kwargs = {
        'ax_voltage' : False,
        'ax_current' : False
    }
    
    kwargs = handle_default_kwargs(kwargs,default_kwargs)
    
    if (kwargs['ax_voltage']==False and kwargs['ax_current']==False ):
        raise ValueError('no axis was supplied for ax_voltage or for ax_current')
    
    data_output_ordered = handle_interface_to_ordered(data_output_ordered)
    # get closest index to event passed
    index_closest,_ = closest_event_to_time(data_output_ordered.Time, Time_Enquriey,False)
    
    # Voltage
    if(not isinstance(kwargs['ax_voltage'],bool)):
        # plot inductor voltage and get intercept value
        data_VL = plot_time_interconnect(data_output_ordered,kwargs['ax_voltage'],'voltage inductor',True,return_data = True)
        interconncet_voltage_inductor = data_VL[index_closest]
        
        # plot capacitor voltage on same axes and get intercept value
        data_VC = plot_time_interconnect(data_output_ordered,kwargs['ax_voltage'],'voltage capacitor',True,return_data = True)
        interconncet_voltage_capacitor = data_VC[index_closest]
        
        # label voltage axis
        kwargs['ax_voltage'].set_title('Voltage at Interconnect')
        kwargs['ax_voltage'].legend(['Inductor','Capacitor'],loc='upper right')
        kwargs['ax_voltage'].axhline(interconncet_voltage_inductor,linestyle='--',c='C0')
        kwargs['ax_voltage'].axhline(interconncet_voltage_capacitor,linestyle='--',c='C1')
        kwargs['ax_voltage'].axvline(Time_Enquriey,linestyle='--',c='gray')
        kwargs['ax_voltage'].xaxis.set_major_formatter(EngFormatter('s'))
        kwargs['ax_voltage'].yaxis.set_major_formatter(EngFormatter('V'))
        kwargs['ax_voltage'].set_xlabel('time')
        kwargs['ax_voltage'].set_ylabel('voltage')
        kwargs['ax_voltage'].axhline(y=0,c='k',linewidth=0.5)
    
    # Current
    if(not isinstance(kwargs['ax_current'],bool)):
        # plot inductor current and get intercept value
        data_IL = plot_time_interconnect(data_output_ordered,kwargs['ax_current'],'current inductor',True,return_data = True)
        interconncet_current_inductor = data_IL[index_closest]
        
        # plot capacitor current  on samw current axis, get intercept value
        data_IC = plot_time_interconnect(data_output_ordered,kwargs['ax_current'],'current capacitor',True,return_data = True)
        interconnect_current_capacitor = data_IC[index_closest]
        
        # label current axis
        kwargs['ax_current'].set_title('Current at Interconnect')
        kwargs['ax_current'].legend(['Inductor','Capacitor'],loc='upper right')
        kwargs['ax_current'].axhline(interconncet_current_inductor,linestyle='--',c='C0')
        kwargs['ax_current'].axhline(interconnect_current_capacitor,linestyle='--',c='C1')
        kwargs['ax_current'].axvline(Time_Enquriey,linestyle='--',c='gray')
        kwargs['ax_current'].xaxis.set_major_formatter(EngFormatter('s'))
        kwargs['ax_current'].yaxis.set_major_formatter(EngFormatter('A'))
        kwargs['ax_current'].set_xlabel('time')
        kwargs['ax_current'].set_ylabel('current')
        kwargs['ax_current'].axhline(y=0,c='k',linewidth=0.5)
    
def make_3d_spatial(Time_Enquriey: Decimal,interface: Input_Data,input_ax = False):
    """an experimanetal plot that shows spatial distribution of voltage and current at a time as a 3D bar graph.
    One dimension is space, one dimenstion is voltage and the final dimension is current. 
    See :py:func:`make_spatial_voltage_and_current` for a less dense representation of the same data. 

    :param Time_Enquriey: time ate wwich the spatial information is investigated
    :type Time_Enquriey: Decimal
    :param interface: interface simulation storage object
    :type interface: Input_Data
    :param input_ax: an optional axis to prevent plotting object to be made internally, default is False
    :type input_ax: matplotlib Axes (projection='3d')
    
    .. code-block ::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import make_3d_spatial
        import matplotlib.pyplot as plt
        from decimal import Decimal

        # simulate interface
        interface_data = generate_interface_data(L_time='12' , C_time='13')

        make_3d_spatial(Decimal('53.56'),interface_data)
        
        # or
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        make_3d_spatial(Decimal('67.07'),interface_data,ax)
        
        plt.show()
        
    .. warning::
    
        if `input_ax` is provided, it must have 'projection='3d' else the plot will error.
    
    """
    if (isinstance(input_ax,bool)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = input_ax
    
    ax.xaxis.set_major_formatter(EngFormatter('m'))
    ax.yaxis.set_major_formatter(EngFormatter('A'))
    ax.zaxis.set_major_formatter(EngFormatter('V'))
            
    ax.set_xlabel('position')
    ax.set_ylabel('current')
    ax.set_zlabel('voltage')
    
    is_Inductive =True

    # INDUCTOR
    pos_all, value_lv, value_rv, value_lc, value_rc = get_spatial_voltage_current_at_time(Time_Enquriey, interface,is_Inductive)
    zip_out = zip(pos_all, value_lv, value_rv, value_lc, value_rc)

    #arrays
    x_position =[]
    y_current =[]
    z_voltage =[]

    dx_position =[]
    dy_current =[]
    dz_voltage =[]

    #initiate variables
    x_old = 0
    y_voltage_old = 0
    y_current_old = 0

    is_first = True

    for (position, left_voltage, right_voltage, left_current, right_current) in zip_out:
            
            x = position
                   
            x_position.append(x)
            y_current.append(0)
            z_voltage.append(0)
            
            if(is_first):
                    x_old = position
                    y_voltage_old = left_voltage
                    y_current_old = left_current
                    is_first = False
            else:
                    dx_position.append(x - x_old)
                    dy_current.append(y_current_old)
                    dz_voltage.append(y_voltage_old)
                    
            x_old = x
            
            y_voltage_old = right_voltage
            y_current_old = right_current
            

    x_position.pop()
    y_current.pop()
    z_voltage.pop()
    
    # plot inductor current and voltage
    ax.bar3d(x_position, y_current, z_voltage, dx_position, dy_current, dz_voltage )
    # ax.bar3d(x_position, y_current, z_voltage, dx_position, [-x for x in dy_current], dz_voltage )
    
    
    # The capacitor
    # setup arrays
    x_position =[]
    y_current =[]
    z_voltage =[]

    dx_position =[]
    dy_current =[]
    dz_voltage =[]

    is_first = True

    # get positions of voltages and currents
    pos_all, value_lv, value_rv, value_lc, value_rc = get_spatial_voltage_current_at_time(Time_Enquriey, interface,not is_Inductive)
    zip_out = zip(pos_all, value_lv, value_rv, value_lc, value_rc)

    for (position, left_voltage, right_voltage, left_current, right_current) in zip_out:
            
            x = -position
            
            x_position.append(x)
            z_voltage.append(0)
            y_current.append(0)
            
            if(is_first):
                    x_old = position
                    y_voltage_old = left_voltage
                    y_current_old = left_current
                    is_first = False
            else:
                    dx_position.append(x - x_old)
                    dy_current.append(y_current_old)
                    dz_voltage.append(y_voltage_old)
            
            x_old = x
            
            y_voltage_old = right_voltage
            y_current_old = right_current

    x_position.pop()
    y_current.pop()
    z_voltage.pop()

    # plot capacitor currents and voltages
    ax.bar3d(x_position, y_current, z_voltage, dx_position, dy_current, dz_voltage )
    # ax.bar3d(x_position, y_current, z_voltage, dx_position, [-x for x in dy_current], dz_voltage )

def save_spatial_interconnect(Interface : Interface_Data,**kwargs):
    """a function that saves an animation of the spatial distribution of voltage and current compared to time interconncect plots.
    Is the combination of :py:func:`make_spatial_voltage_and_current` and :py:func:`plot_time_interconnect_and_intercepts_at_time`.
    It is effectively :py:func:`interactive.interact_spatial`, however smoother as computation is not 'real-time'

    :param Interface: the interface data to be saved.
    :type Interface: Interface_Data
    :**kwargs**:
        - **auto_zoom** (*bool*) - Whether to automatically zoom the plot or to have the axes aligned. Default is False.
        - **start_time** (*str*) - The start time enquirey for the plot. Default is '0'.
        - **end_time** (*float*) - The end time enquirey for the plot. Default is the value of `Interface.data_input.Simulation_Stop_Time`.
        - **fps** (*str*) - The frames per second for the video. Default is '30'.
        - **video_runtime** (*str*) - The runtime of the video in seconds. Default is '60'.
        - **dpi** (*str*) - The dots per inch for the video. Default is '300'.
        - **fig_size** (*tuple of ints*) - The size of the figure. Default is (14, 8).
        - **meta_data** (*dict*) - The metadata for the video. Default is {'title': 'Distributed Modelling', 'artist': 'Jonathan Meerholz'}.
        - **save_name** (*str*) - The name to save the video as. Default is a string with the values of 'spatial_and_time_{ZL}_{ZC}ohm_{TL}_{TC}s'
    
    .. warning::
    
        the default values of 60s runtime with 30 fps will result in a computation that will often take longer than 10 mins. 
        be sure to alter these values if you dont want to wait!
        
    .. code-block::
    
        from wavefronts.generation import generate_interface_data
        from wavefronts.plotting import save_spatial_interconnect

        # simulate an interface
        interface_data = generate_interface_data(L_time = '3.6',C_time = '3.2')

        save_spatial_interconnect(interface_data, video_runtime = '5',
                                start_time = '0', end_time = '30')
   
    """

    #Default Values
    kwarg_options = dict([
        ('auto_zoom',False),
        ('start_time','0'), ('end_time',Interface.data_input.Simulation_Stop_Time), 
        ('fps','30'),('video_runtime','60'),('dpi','300'),
        ('fig_size',(14, 8)),
        ('meta_data',dict(title='Distributed Modelling', artist='Jonathan Meerholz')),
        ('save_name',f'spatial_and_time_{Interface.data_input.Inductor_Impedance}_{Interface.data_input.Capacitor_Impedance}ohm_{Interface.data_input.Inductor_Time*2}_{Interface.data_input.Capacitor_Time*2}s')
        ])
         
    kwarg_options = handle_default_kwargs(kwargs,kwarg_options)
    
    # fig_save_2d, ax_save_2d = plt.subplots(2,2,figsize=kwarg_options['fig_size'],constrained_layout = True)
    fig_save_2d, ax_save_2d = plt.subplot_mosaic([['V', 'inter-V'],
                                                  ['I', 'inter-I']],figsize=kwarg_options['fig_size'],constrained_layout = True)
    

    save_name = kwarg_options['save_name']
    save_name = save_name.replace('.',',')

    start_time = Decimal(kwarg_options['start_time'])
    end_time = Decimal(kwarg_options['end_time'])

    fps = Decimal(kwarg_options['fps'])
    video_runtime = Decimal(kwarg_options['video_runtime'])
    dpi = kwarg_options['dpi']

    number_frames =  video_runtime*fps
    time_increment = (end_time - start_time)/number_frames

    metadata = kwarg_options['meta_data']
    writer = FFMpegWriter(fps=float(fps), metadata=metadata)

    time = start_time
    frame_counter = 0
    with writer.saving(fig_save_2d, (save_name+".mp4"), float(dpi)):

        for i in tqdm(range(0,int(number_frames))):
            make_spatial_voltage_and_current(time,Interface,ax=ax_save_2d)
            plot_time_interconnect_and_intercepts_at_time(time,Interface.data_output_ordered,ax_voltage=ax_save_2d['inter-V'],ax_current =ax_save_2d['inter-I'])
            
            if(kwarg_options['auto_zoom'] == False):
                ax_save_2d['V'].set_ylim(ax_save_2d['inter-V'].get_ylim())
                ax_save_2d['I'].set_ylim(ax_save_2d['inter-I'].get_ylim())
            
            writer.grab_frame()
            
            time += time_increment
            frame_counter +=1
            
            clear_subplot(ax_save_2d.values())

    plt.close(fig_save_2d)
    print(f"Spatial video generation completed, video saved as {save_name}.mp4")

