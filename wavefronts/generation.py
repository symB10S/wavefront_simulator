"""This module is responsible for processing various levels of wavefront simulaiton and storing them in their relevant Data_Storage arrays (see :mod:`storage` module).

Notable functions are:
    - :py:func:`generate_commutative_data` - generates the production of wavefronts from a Input_Data input variable array. Makes use of commutative merging (described in the associated paper) which make it possible for the efficient simualiton of wavefronts.  
"""

from decimal import Decimal, getcontext, FloatOperation
from collections import deque
import numpy as np
import math
import copy
import warnings

from wavefronts.storage import *
from wavefronts.misc import *

getcontext().traps[FloatOperation] = True

def generate_commutative_data(input_data : Input_Data):
    """The commutative generaion algorithm. Generates a :py:class:`storage.Output_Data` object from the calculated input variables stored in a :py:class:`storage.Input_Data object`. 
    
    :param input_data: Input data object containing simulation input variables
    :type input_data: Input_Data
    :return: output data (a collection of commutative fanouts in form of np.ndarrays)
    :rtype: Output_Data
    
    Resposible for generating wavefronts and simultaneously commutatively merging the wavefronts. 
    The simaltaneous commutative merging of wavefronts is mandatory for longer simulation times.
    """
    # The algorithm to follow makes use of SPARSE storage arrays,
    # this is done as to store both capacitve and inductive wavefronts in a single array
    # (there will be two spares arrays, one for AWAY wavefronts and one for RETURNING wavefronts).

    # SPARSE FANOUT STORAGE ARRAY FORMAT FOR 5 LAYERS:
    # (horizontal = inductive axis, vertical = capacitive axis)
    # x = major gird node, → = wavefront inductive, ↓ = wavefront capacitve, 0 = 'blank entry'
    #   0 1 2 3 4 5 6 7 8 9
    #   ____________________
    #0 |x → x → x → x → x →
    #1 |↓ 0 ↓ 0 ↓ 0 ↓ 0 ↓ 
    #2 |x → x → x → x →
    #3 |↓ 0 ↓ 0 ↓ 0 ↓ 
    #4 |x → x → x → 
    #5 |↓ 0 ↓ 0 ↓
    #6 |x → x →
    #7 |↓ 0 ↓
    #8 |x →
    #9 |↓
    
    # The two Sparse storage arrays:
    # ------------------------------
    Wavefronts_Away = np.full((2*(input_data.Number_of_Layers+1),2*(input_data.Number_of_Layers+1)),Wavefront_Source(input_data,0,0))
    Wavefronts_Return = np.full((2*(input_data.Number_of_Layers+1),2*(input_data.Number_of_Layers+1)),Wavefront_Source(input_data,0,0))
    
    
    # These Sparse arrays are then post-porcessed in a 'gemoetric" way to extract magnitude data in a dense format.
    # Dense format arrays will store data as a function of major nodes, and will have no 'blank entries'.
    # An example of a Dense array would be "Wavefronts sent from the Inductor" (Wavefronts_Sending_Inductor):
    
    # DENSE FANOUT STORAGE ARRAY FORMAT FOR 5 LAYERS:
    # (horizontal = inductive axis, vertical = capacitive axis)
    # x = major gird node, → = wavefront inductive, ↓ = wavefront capacitve, 0 = 'blank entry'
    #   0 1 2 3 4 
    #   __________
    #0 |→ → → → →
    #1 |→ → → →  
    #2 |→ → → 
    #3 |→ → 
    #4 |→ 

    # Dense format arrays tracked:
    # ----------------------------
    Time = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)),Decimal('0'))
    
    Voltage_Interconnect_Inductor = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)),Decimal('0'))
    Current_Interconnect_Inductor = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)),Decimal('0'))

    Voltage_Interconnect_Capacitor = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)),Decimal('0'))
    Current_Interconnect_Capacitor = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)),Decimal('0'))
    
    Wavefronts_Sending_Inductor = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)), Wavefront())
    Wavefronts_Sending_Capacitor = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)), Wavefront())
    
    Wavefronts_Returning_Inductor = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)), Wavefront())
    Wavefronts_Returning_Capacitor = np.full(((input_data.Number_of_Layers+1),(input_data.Number_of_Layers+1)), Wavefront())
    
    # POPULATE THE SPARSE STORAGE ARRAYS
    # ===================================
    #Deques of wavefronts thare are used to temporarily store wavefronts as they are being processed.
    Wavefronts_Away_deque : Wavefront = deque()
    Wavefronts_Returning_deque : Wavefront = deque()

    # Generate Intial Away Waves from voltage excitation
    temp_wavefront = Wavefront_Source(input_data,0,input_data.Voltage_Souce_Magnitude)
    temp_wavefront.generate_and_store(Wavefronts_Away_deque)

    # Generate Intial Return Waves,
    # Get Intial Sending wavefront, this will be an inductive wavefront
    temp_wavefront_inductive = Wavefronts_Away_deque.popleft()
    temp_wavefront_inductive.generate_and_store(Wavefronts_Returning_deque)
    Wavefronts_Away[1,0] = temp_wavefront_inductive
    
    # Get Next Initial Sending wavefront, this will be a capacitive wavefront
    temp_wavefront_capacitive = Wavefronts_Away_deque.popleft()
    temp_wavefront_capacitive.generate_and_store(Wavefronts_Returning_deque)
    Wavefronts_Away[0,1] = temp_wavefront_capacitive

    # GENERATE WAVEFRONTS AND MERGE COMMUTATIVELY
    for layer_number in range(1,input_data.Number_of_Layers):

        # RETURNING WAVEFRONTS
        # --------------------
        
        # set Index    
        Wavefront_Index_x = 2*layer_number-1
        Wavefront_Index_y = 0
        
        # process first Returning Wavefront:

        # (will be inductive wavefront) 
        # (first wavefront does not merge)
        temp_wavefront = Wavefronts_Returning_deque.popleft()

        # generate away wavefronts,  
        temp_wavefront.generate_and_store(Wavefronts_Away_deque)
        # store returning wavefront, 
        Wavefronts_Return[Wavefront_Index_x,Wavefront_Index_y] = temp_wavefront
        # shift index
        Wavefront_Index_x = Wavefront_Index_x - 1
        Wavefront_Index_y = Wavefront_Index_y + 1
        
        # process remaining Returning Wavefronts:

        while len(Wavefronts_Returning_deque) > 0:
            # Get a Returning wavefront 
            # (will be capacitve)
            temp_wavefront = Wavefronts_Returning_deque.popleft()
            
            if len(Wavefronts_Returning_deque) == 0 : # It is the last wave?
                # (Last wavefront does not merge)
                # generate away wavefronts and store in Away wavefronts deque
                temp_wavefront.generate_and_store(Wavefronts_Away_deque)
                # store returning wavefronts
                Wavefronts_Return[Wavefront_Index_x,Wavefront_Index_y] = temp_wavefront
                # shift index
                Wavefront_Index_x = Wavefront_Index_x - 1
                Wavefront_Index_y = Wavefront_Index_y + 1

            else: # It is not the last wave :
                
                # merge children of 'adjacent' returning wavefronts:

                # get next returning wavefront 
                # (will be inductive)
                temp_next_wavefront = Wavefronts_Returning_deque.popleft()

                # get children of the two current wavefronts
                temp_wavefront_inductive, temp_wavefront_capacitve = temp_wavefront.generate_and_return()
                temp_next_wavefront_inductive, temp_next_wavefront_capacitve = temp_next_wavefront.generate_and_return()

                # commutatively merge the children appropriately 
                temp_wavefront_inductive.merge(temp_next_wavefront_inductive)
                temp_wavefront_capacitve.merge(temp_next_wavefront_capacitve)

                # add newly merged children in Away wavefronts deque
                Wavefronts_Away_deque.append(temp_wavefront_inductive)
                Wavefronts_Away_deque.append(temp_wavefront_capacitve)
                
                # Store current returning wavefronts in their completion storage array
                # capacitive returning wavefront
                Wavefronts_Return[Wavefront_Index_x,Wavefront_Index_y] = temp_wavefront
                # Shift index
                Wavefront_Index_x = Wavefront_Index_x - 1
                Wavefront_Index_y = Wavefront_Index_y + 1
                
                # inductive returning wavefront
                Wavefronts_Return[Wavefront_Index_x,Wavefront_Index_y] = temp_next_wavefront
                # shift index
                Wavefront_Index_x = Wavefront_Index_x - 1
                Wavefront_Index_y = Wavefront_Index_y + 1
        
        # AWAY WAVEFRONTS
        # ================
        # Set Index for Away wavefronts in Sparse array
        # (will be one ahead of returning) 
        Wavefront_Index_x = 2*(layer_number+1)-1
        Wavefront_Index_y = 0
        
        while len(Wavefronts_Away_deque)> 0:
            # get an away wavefront in the away wavefront deque
            # (will be inductive)
            temp_wavefront_inductive = Wavefronts_Away_deque.popleft()
            # Generate and store its returning children
            temp_wavefront_inductive.generate_and_store(Wavefronts_Returning_deque)
            # store processed away wavefront
            Wavefronts_Away[Wavefront_Index_x, Wavefront_Index_y] = temp_wavefront_inductive
            # shift index
            Wavefront_Index_x = Wavefront_Index_x - 1
            Wavefront_Index_y = Wavefront_Index_y + 1
            
            # Get the next Away wavefront 
            # (will be capacitive)
            temp_wavefront_capacitve = Wavefronts_Away_deque.popleft()
            # Generate and store its returning children
            temp_wavefront_capacitve.generate_and_store(Wavefronts_Returning_deque)
            # store processed away wavefront
            Wavefronts_Away[Wavefront_Index_x, Wavefront_Index_y] = temp_wavefront_capacitve
            # shift index
            Wavefront_Index_x = Wavefront_Index_x - 1
            Wavefront_Index_y = Wavefront_Index_y + 1

    # POST PORCESSING OF SPARSE ARRAY
    # ===============================
    
    for layer_number in range(0,input_data.Number_of_Layers):
        # Get major grid node coords for first node in layer
        Sparse_Major_Node_Index_x = 2*layer_number
        Sparse_Major_Node_Index_y = 0
        
        Dense_Major_Node_Index_x = layer_number
        Dense_Major_Node_Index_y = 0
        
        for node_number in range(0,layer_number+1):
            # Get indexes of surrounding wavefronts
            # -------------------------------------
            # AWAY FROM major grid node inductive wavefront index:
            Away_Index_Inductor_x = Sparse_Major_Node_Index_x + 1
            Away_Index_Inductor_y = Sparse_Major_Node_Index_y
            wavefront_sending_inductor = Wavefronts_Away[Away_Index_Inductor_x,Away_Index_Inductor_y]
            # RETURNING TO major grid node inductive wavefront index:
            Return_Index_Inductor_x = Sparse_Major_Node_Index_x - 1
            Return_Index_Inductor_y = Sparse_Major_Node_Index_y
            wavefront_returning_inductor = Wavefronts_Return[Return_Index_Inductor_x,Return_Index_Inductor_y]
            
            # AWAY FROM major grid node capacitive wavefront index:
            Away_Index_Capacitor_x = Sparse_Major_Node_Index_x 
            Away_Index_Capacitor_y = Sparse_Major_Node_Index_y + 1
            wavefront_sending_capacitor = Wavefronts_Away[Away_Index_Capacitor_x,Away_Index_Capacitor_y]
            # RETURNING TO major grid node capacitive wavefront index:
            Return_Index_Capacitor_x = Sparse_Major_Node_Index_x 
            Return_Index_Capacitor_y = Sparse_Major_Node_Index_y - 1
            wavefront_returning_capacitor = Wavefronts_Return[Return_Index_Capacitor_x,Return_Index_Capacitor_y]
            
            # store AWAY wavefronts in major node position ("away from")
            Time[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_inductor.time_start
            Wavefronts_Sending_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_inductor
            Wavefronts_Sending_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_capacitor
            # store RETURNING wavefronts in major node position (also "away from")
            # (returning wavefronts are stored in assoicated to the their AWAY wavefront parents major grid node)
            # (this is not the same as the RETURNING TO format used to calculate interconncet changes)
            Wavefronts_Returning_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = Wavefronts_Return[Away_Index_Inductor_x,Away_Index_Inductor_y]
            Wavefronts_Returning_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = Wavefronts_Return[Away_Index_Capacitor_x,Away_Index_Capacitor_y]

            if(node_number == 0 and layer_number ==0): 
                    # origin node
                    # inductor interconncet magnitude for origin node has only a sent wavefront to consider
                    Voltage_Interconnect_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_inductor.magnitude_voltage 
                    Current_Interconnect_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_inductor.magnitude_current
                    
                    # capacitor interconncet magnitude for origin node has only a sent wavefront to consider
                    Voltage_Interconnect_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_capacitor.magnitude_voltage 
                    Current_Interconnect_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_capacitor.magnitude_current

            elif(node_number == 0 ): 
                    # first node is an INDUCTIVE UNIQUE NODE
                    # inductor interconnect magnitudes of inductive unique nodes are affected by both returning and arriving inductive wavefronts
                    Voltage_Interconnect_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = (wavefront_sending_inductor.magnitude_voltage  + wavefront_returning_inductor.magnitude_voltage) 
                    Current_Interconnect_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = (wavefront_sending_inductor.magnitude_current + wavefront_returning_inductor.magnitude_current ) 
                    
                    # capacitor interconnect magnitudes of inductive unique nodes are only affected by wavefronts sent into the capaitor
                    Voltage_Interconnect_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_capacitor.magnitude_voltage 
                    Current_Interconnect_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_capacitor.magnitude_current

            elif(node_number == layer_number): 
                    # last node is a CAPACITVE UNIQUE NODE
                    # inductor interconnect magnitudes of capacitive unique nodes are only affected by wavefronts sent into the inductor
                    Voltage_Interconnect_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_inductor.magnitude_voltage  
                    Current_Interconnect_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = wavefront_sending_inductor.magnitude_current
                    
                    # capacitor interconnect magnitudes of capcitive unique nodes are affected by both returning and arriving capacitor wavefronts
                    Voltage_Interconnect_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = (wavefront_sending_capacitor.magnitude_voltage  + wavefront_returning_capacitor.magnitude_voltage) 
                    Current_Interconnect_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = (wavefront_sending_capacitor.magnitude_current + wavefront_returning_capacitor.magnitude_current )
            else:
                    # general node
                    # interconnect values of the inductor for general nodes are a sum of both sending and returning wavefronts
                    Voltage_Interconnect_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = (wavefront_sending_inductor.magnitude_voltage  + wavefront_returning_inductor.magnitude_voltage) 
                    Current_Interconnect_Inductor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = (wavefront_sending_inductor.magnitude_current + wavefront_returning_inductor.magnitude_current ) 
                    
                    # interconnect values of the capacitor for general nodes are a sum of both sending and returning wavefronts
                    Voltage_Interconnect_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = (wavefront_sending_capacitor.magnitude_voltage  + wavefront_returning_capacitor.magnitude_voltage)
                    Current_Interconnect_Capacitor[Dense_Major_Node_Index_x,Dense_Major_Node_Index_y] = (wavefront_sending_capacitor.magnitude_current + wavefront_returning_capacitor.magnitude_current )
            
            # update index and go to next layer     
            Sparse_Major_Node_Index_x -= 2
            Sparse_Major_Node_Index_y += 2
            
            Dense_Major_Node_Index_x -= 1
            Dense_Major_Node_Index_y += 1
    
    return Output_Data(
        Time, # merge Times
        Voltage_Interconnect_Inductor, # Values at interconnect 
        Current_Interconnect_Inductor, # Values at interconnect
        Voltage_Interconnect_Capacitor, # Values at interconnect
        Current_Interconnect_Capacitor, # Values at interconnect
        Wavefronts_Sending_Inductor, # Specific Wavefronts at Nodes
        Wavefronts_Sending_Capacitor, # Specific Wavefronts at Nodes
        Wavefronts_Returning_Inductor, # Specific Wavefronts at Nodes
        Wavefronts_Returning_Capacitor, # Specific Wavefronts at Nodes
        False, # indicated that multiplicative merging has not occured
        )

def multiplicative_merging_single_cycle(input_array:np.ndarray,Inductor_LCM_Factor:int,Capacitor_LCM_Factor:int):
    """Completes a single merging cycle of a mangitude fanout along the inductive axis.
    A single cycle consitis of splitting -> shift -> merging.

    :param input_array: An output array from Datat_Output_Storage class., i.e. data_output.Voltage_Interconnect_Inductor
    :type input_array: np.ndarray
    :param Inductor_LCM_Factor: The co-factor of the time-delay for the inductor, KL. KL x TL = LCM(TL,TC)
    :type Inductor_LCM_Factor: int
    :param Capacitor_LCM_Factor: The co-factor of the time-delay for the capacitor axis, KC. KC x TC = LCM(TL,TC)
    :type Capacitor_LCM_Factor: int
    :return: returns the input_array after one more subsequent merging cycle.
    :rtype: np.ndarray
    """
    # split arrays along C = Capacitor_LCM_Factor,
    stationary_array,translated_array = split_and_translate_to_L_axis(input_array,Capacitor_LCM_Factor)
    # shift translated array by L = Inductor_LCM_Factor
    array_merge_ready = translate_along_L_axis(translated_array,Inductor_LCM_Factor)
    # merging regions will now be aligned, can now merge. 
    array_merged = stationary_array + array_merge_ready
    
    return array_merged

def multiplicative_merging(input_array:np.ndarray,Inductor_LCM_Factor:int ,Capacitor_LCM_Factor:int ,layer_number_limit:int):
    """recursively apply the merging process on an input array until merged. 

    :param input_array: array to be merged
    :type input_array: np.ndarray
    :param Inductor_LCM_Factor: Inductor LCM cofactor KL
    :type Inductor_LCM_Factor: int
    :param Capacitor_LCM_Factor: Capacitor LCM cofactor KC
    :type Capacitor_LCM_Factor: int
    :param layer_number_limit: up to what layer the array must be mrged to 
    :type layer_number_limit: int
    :return: merged array
    :rtype: np.ndarray
    """
    number_merge_cycles:int = math.ceil(layer_number_limit/Capacitor_LCM_Factor) + 1
    
    for _ in range (0,number_merge_cycles):
        input_array:np.ndarray = multiplicative_merging_single_cycle(input_array,Inductor_LCM_Factor,Capacitor_LCM_Factor)

    return input_array[:,0:Capacitor_LCM_Factor]

def transform_merged_array_to_capacitor_axis(data_input : Input_Data,merged_array):
    """Transform merged data output array to a C-axis merging representation

    :param data_input: input data for merged array
    :type data_input: Input_Data
    :param merged_array: merged array aligne to the C-axis
    :type merged_array: np.ndarray[Decimal]
    :return: merged array aligned to the C-axis
    :rtype: np.ndarray[Decimal]
    """
    
    def extract_merging_region(data_input : Input_Data,merged_array, KL_index):
        # extract a mergign region along the inductive axis
        KL = data_input.Inductor_LCM_Factor
        KC = data_input.Capacitor_LCM_Factor
    
        return merged_array[KL_index*KL:KL_index*KL+KL,0:KC]

    # get first meging region
    new_array = extract_merging_region(data_input,merged_array,0)
    # determine number of merging regions
    number_of_KLs = int((merged_array.shape[0])/data_input.Inductor_LCM_Factor)
    for i in range(1,number_of_KLs):
        # rearrange and add merging regions allong the C-axis
        new_merging_region = extract_merging_region(data_input,merged_array,i)
        new_array = np.concatenate((new_array,new_merging_region),axis =1)
        
    return new_array

def generate_multiplicative_data(input_data : Input_Data,commutative_output_data : Output_Data):
    """Multiplicatively merges all commutatively merged data if applicable. Produces a Output_Data object with merged data.

    :param input_data: input data of interface
    :type input_data: Input_Data
    :param commutative_output_data: the commutatively merged data to be multiplicatively merged
    :type commutative_output_data: Output_Data
    :return: a merged Output_Data storage object, merged version of the supplied commutative_output_data parameter
    :rtype: Output_Data
    """
    commutative_output_data = copy.deepcopy(commutative_output_data)
    
    if(input_data.is_Higher_Merging):
        Voltage_Interconnect_Inductor_merged = multiplicative_merging(commutative_output_data.Voltage_Interconnect_Inductor,input_data.Inductor_LCM_Factor,input_data.Capacitor_LCM_Factor,input_data.Number_of_Layers)
        Current_Interconnect_Inductor_merged = multiplicative_merging(commutative_output_data.Current_Interconnect_Inductor,input_data.Inductor_LCM_Factor,input_data.Capacitor_LCM_Factor,input_data.Number_of_Layers)
        
        Voltage_Interconnect_Capacitor_merged = multiplicative_merging(commutative_output_data.Voltage_Interconnect_Capacitor,input_data.Inductor_LCM_Factor,input_data.Capacitor_LCM_Factor,input_data.Number_of_Layers)
        Current_Interconnect_Capacitor_merged = multiplicative_merging(commutative_output_data.Current_Interconnect_Capacitor,input_data.Inductor_LCM_Factor,input_data.Capacitor_LCM_Factor,input_data.Number_of_Layers)
        
        Wavefronts_Sending_Inductor_merged = multiplicative_merging(commutative_output_data.Wavefronts_Sending_Inductor,input_data.Inductor_LCM_Factor,input_data.Capacitor_LCM_Factor,input_data.Number_of_Layers)
        Wavefronts_Sending_Capacitor_merged = multiplicative_merging(commutative_output_data.Wavefronts_Sending_Capacitor,input_data.Inductor_LCM_Factor,input_data.Capacitor_LCM_Factor,input_data.Number_of_Layers)

        Wavefronts_Returning_Inductor_merged = multiplicative_merging(commutative_output_data.Wavefronts_Returning_Inductor,input_data.Inductor_LCM_Factor,input_data.Capacitor_LCM_Factor,input_data.Number_of_Layers)
        Wavefronts_Returning_Capacitor_merged = multiplicative_merging(commutative_output_data.Wavefronts_Returning_Capacitor,input_data.Inductor_LCM_Factor,input_data.Capacitor_LCM_Factor,input_data.Number_of_Layers)
        
        Time_cut = commutative_output_data.Time[:,0:input_data.Capacitor_LCM_Factor]
    else:
        Voltage_Interconnect_Inductor_merged = commutative_output_data.Voltage_Interconnect_Inductor
        Current_Interconnect_Inductor_merged = commutative_output_data.Current_Interconnect_Inductor
        
        Voltage_Interconnect_Capacitor_merged = commutative_output_data.Voltage_Interconnect_Capacitor
        Current_Interconnect_Capacitor_merged = commutative_output_data.Current_Interconnect_Capacitor
        
        Wavefronts_Sending_Inductor_merged = commutative_output_data.Wavefronts_Sending_Inductor
        Wavefronts_Sending_Capacitor_merged = commutative_output_data.Wavefronts_Sending_Capacitor

        Wavefronts_Returning_Inductor_merged = commutative_output_data.Wavefronts_Returning_Inductor
        Wavefronts_Returning_Capacitor_merged = commutative_output_data.Wavefronts_Returning_Capacitor
        
        Time_cut = commutative_output_data.Time
    
    
    return Output_Data(
        Time_cut,
        Voltage_Interconnect_Inductor_merged ,
        Current_Interconnect_Inductor_merged ,
        Voltage_Interconnect_Capacitor_merged ,
        Current_Interconnect_Capacitor_merged ,
        Wavefronts_Sending_Inductor_merged ,
        Wavefronts_Sending_Capacitor_merged ,
        Wavefronts_Returning_Inductor_merged ,
        Wavefronts_Returning_Capacitor_merged ,
        True # indicates Higher order merging has occured
    )

def order_multiplicative_data(input_data : Input_Data , Data_Output_Merged : Output_Data):
    """Order the merged wavefront data into single dimension chronologically occuring 'lists'. Uses a Breadth First Search type algorithm.

    :param input_data: the input data of the interface
    :type input_data: Input_Data
    :param Data_Output_Merged: the merged data to be ordered
    :type Data_Output_Merged: Output_Data
    :raises warnings.warn: should be used on a merged data storage object, ekse results may be incorrect
    :return: ordered merged data
    :rtype: Ordered_Output_Data
    """
    if (Data_Output_Merged.has_merged == False):
        raise warnings.warn("Provided Output_Data object to be ordered has not been merged yet. This can produce incorrect results if merging is not accounted for.")
    
    
    def store_options(merged_array,x,y,option_array,index_array):
        """Store neighbouring cells as options and mark on grid.

        :param merged_array: array being orderd
        :type merged_array: np.ndarray
        :param x: current x-index
        :type x: int
        :param y: current y-index
        :type y: int
        :param option_array: stores the value from the merged array as an option
        :type option_array: list
        :param index_array: stores the index of the option
        :type index_array: list
        """
        # get shape to make sure not out of bounds
        x_size,y_size = merged_array.shape
        
        # if x+1 neighbour is not marked, store as an option and mark
        if(x+1 < x_size and Marked[x+1,y] == 0):
            option_array.append(merged_array[x+1,y])
            index_array.append([x+1,y])
            Marked[x+1,y] = 1
        
        # if y+1 neighbour is not marked, store as an option and mark
        if(y+1 < y_size and Marked[x,y+1] == 0):
            option_array.append(merged_array[x,y+1])
            index_array.append([x,y+1])
            Marked[x,y+1] = 1
        
    def get_best_option_value_index(option_array,index_array):
        """return the value and index of the option with the minimum value. 
        delete best option from list.

        :param option_array: possible options
        :type option_array: List[Decimal]
        :param index_array: possible options indexes
        :type index_array: List[(int,int)]
        :return: (value, index)
        :rtype:  tuple ( Decimal, (int,int))
        """
        index_of_min_option = np.argmin(option_array)
        
        value = option_array[index_of_min_option]
        del option_array[index_of_min_option]
        
        index = index_array[index_of_min_option]
        del index_array[index_of_min_option]
        
        return value, index
    
    # Storage Arrays
    out_time = []

    out_voltage_inductor = []
    out_current_inductor = []
    out_voltage_capacitor = []
    out_current_capacitor = []

    out_wavefront_sending_inductor = []
    out_wavefront_sending_capacitor = []
    
    out_wavefront_returning_inductor = []
    out_wavefront_returning_capacitor = []

    out_indexes = []

    # Ordering Utilities
    latest_time = 0

    x_index = 0
    y_index = 0

    option_time = []
    option_indexes =[]
    
    # array marking options that have been explored.
    Marked = np.zeros(Data_Output_Merged.Time.shape, dtype=Data_Output_Merged.Time.dtype)

    # Store Initial Point
    out_time.append(Data_Output_Merged.Time[0,0])
    out_indexes.append([0,0])
    out_voltage_inductor.append(Data_Output_Merged.Voltage_Interconnect_Inductor[0,0])
    out_current_inductor.append(Data_Output_Merged.Current_Interconnect_Inductor[0,0])
    out_voltage_capacitor.append(Data_Output_Merged.Voltage_Interconnect_Capacitor[0,0])
    out_current_capacitor.append(Data_Output_Merged.Current_Interconnect_Capacitor[0,0])
    out_wavefront_sending_inductor.append(Data_Output_Merged.Wavefronts_Sending_Inductor[0,0])
    out_wavefront_sending_capacitor.append(Data_Output_Merged.Wavefronts_Sending_Capacitor[0,0])
    out_wavefront_returning_inductor.append(Data_Output_Merged.Wavefronts_Returning_Inductor[0,0])
    out_wavefront_returning_capacitor.append(Data_Output_Merged.Wavefronts_Returning_Capacitor[0,0])

    # mark initial point
    Marked[0,0] = 1

    while latest_time < input_data.Simulation_Stop_Time:
        # store options at location
        store_options(Data_Output_Merged.Time,x_index,y_index,option_time,option_indexes)
        
        if(len(option_time) > 0):
            # get best option
            best_time, best_time_index = get_best_option_value_index(option_time,option_indexes)
            
            # append the approriate vlaues using the index
            out_time.append(best_time)
            out_indexes.append(best_time_index)
            
            out_voltage_inductor.append(Data_Output_Merged.Voltage_Interconnect_Inductor[best_time_index[0],best_time_index[1]])
            out_current_inductor.append(Data_Output_Merged.Current_Interconnect_Inductor[best_time_index[0],best_time_index[1]])
            
            out_voltage_capacitor.append(Data_Output_Merged.Voltage_Interconnect_Capacitor[best_time_index[0],best_time_index[1]] )
            out_current_capacitor.append(Data_Output_Merged.Current_Interconnect_Capacitor[best_time_index[0],best_time_index[1]] )
            
            out_wavefront_sending_inductor.append(Data_Output_Merged.Wavefronts_Sending_Inductor[best_time_index[0],best_time_index[1]])
            out_wavefront_sending_capacitor.append(Data_Output_Merged.Wavefronts_Sending_Capacitor[best_time_index[0],best_time_index[1]])
            
            out_wavefront_returning_inductor.append(Data_Output_Merged.Wavefronts_Returning_Inductor[best_time_index[0],best_time_index[1]])
            out_wavefront_returning_capacitor.append(Data_Output_Merged.Wavefronts_Returning_Capacitor[best_time_index[0],best_time_index[1]])
            
            Marked[best_time_index[0],best_time_index[1]]  = 2
        
        latest_time = best_time
        x_index, y_index = best_time_index
    
    # Crop merged to the maximum occuring index as it merged along each axis
    
    max_x_index = np.max([x[0] for x in out_indexes])
    max_x_index += 1
    
    max_y_index = np.max([y[1] for y in out_indexes])
    max_y_index += 1
    
    Data_Output_Merged.Time = Data_Output_Merged.Time[0:max_x_index,0:max_x_index]

    Data_Output_Merged.Voltage_Interconnect_Inductor =  Data_Output_Merged.Voltage_Interconnect_Inductor[0:max_x_index,0:max_y_index]
    Data_Output_Merged.Current_Interconnect_Inductor = Data_Output_Merged.Current_Interconnect_Inductor[0:max_x_index,0:max_y_index]

    Data_Output_Merged.Voltage_Interconnect_Capacitor = Data_Output_Merged.Voltage_Interconnect_Capacitor[0:max_x_index,0:max_y_index]
    Data_Output_Merged.Current_Interconnect_Capacitor = Data_Output_Merged.Current_Interconnect_Capacitor[0:max_x_index,0:max_y_index]

    Data_Output_Merged.Wavefronts_Sending_Inductor = Data_Output_Merged.Wavefronts_Sending_Inductor[0:max_x_index,0:max_y_index]
    Data_Output_Merged.Wavefronts_Sending_Capacitor = Data_Output_Merged.Wavefronts_Sending_Capacitor[0:max_x_index,0:max_y_index]

    Data_Output_Merged.Wavefronts_Returning_Inductor = Data_Output_Merged.Wavefronts_Returning_Inductor[0:max_x_index,0:max_y_index]
    Data_Output_Merged.Wavefronts_Returning_Capacitor = Data_Output_Merged.Wavefronts_Returning_Capacitor[0:max_x_index,0:max_y_index]
            
        
    return Ordered_Output_Data(
        out_time ,
        out_voltage_inductor ,
        out_current_inductor ,
        out_voltage_capacitor ,
        out_current_capacitor ,
        out_wavefront_sending_inductor ,
        out_wavefront_sending_capacitor ,
        out_wavefront_returning_inductor ,
        out_wavefront_returning_capacitor ,
        True ,
        out_indexes
    )        

def generate_interface_data(optional_data_input : Input_Data  = False,**input_values):
    """Do full simualiton of the interface and produce a Interface_Data object with all the simualted data.
    The simulation procedure is as follows: 
    calcualte input vatiables -> generate wavefront with commutative merging -> multiplicatively merge these wavefronts -> chronologically order wavefronts. 

    Is initialised using the same key-word arguments to intitalise Input_Data. OPTIONALLY a Input_Data array can be supplied directly to bypass internal creation of input data if it has been customized.
    
    All values with the provided keys are of type string. 
    This each input variable is converterted to a Decimal value to be used for precision calculations.
    The possible parameters to change and their defualt values are as follows, parameters are all optional
    
    :param L_impedance: Characteristic impedance of the inductor, assigned to self.Inductor_Impedance (default:'100')
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

    :return: Interface Data object
    :rtype: Interface_Data
    
    .. code-block ::
    
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
    if ( isinstance(optional_data_input,bool)):
        data_input = Input_Data(**input_values)
    elif(isinstance(optional_data_input,Input_Data)):
        data_input = optional_data_input
    else:
        raise TypeError("optional input data is of incorrect type. Either supply values using ke-word arguments or supply a Data_Input_Storag object.")
    
    data_output_commutative = generate_commutative_data(data_input)
    data_output_merged = generate_multiplicative_data(data_input,data_output_commutative)
    data_output_ordered = order_multiplicative_data(data_input,data_output_merged)
    
    return Interface_Data(data_input,data_output_commutative,data_output_merged,data_output_ordered)

def get_spatial_voltage_current_at_time(Time_Enquriey : Decimal, Interface : Interface_Data , is_Inductor : bool):
    """Calcualte the postions of wavefronts on a transmission line and get the spatial distribution of voltage and current on either sides og the points.

    :param Time_Enquriey: The time at which spatial behaviour is investigated
    :type Time_Enquriey: Decimal
    :param Interface: The data stroage object of the interface 
    :type Interface: Interface_Data
    :param is_Inductor: if the transmission line investigate is the inductor or capacitor.
    :type is_Inductor: bool
    :return: [intercept_postiions, left_voltage, right_voltage, left_current, right_current] left means closer to the interface.
    :rtype: tuple[list, list, list, list, list]
    """
    
    Time_Enquriey = Decimal(Time_Enquriey)
    
    # Exctract wavefront interceptions at a specific time
    # 1. get sending + returning wavefronts
    # 2. determine DC line values
    # 3. get intercept positions, voltages and currents
    
    sending_wavefronts = []
    returning_wavefronts = []
    
    dc_voltage = Decimal('0')
    dc_current = Decimal('0')

    sending_intercept_positions = []
    sending_intercept_voltages = []
    sending_intercept_currents = []

    returning_intercept_positions = []
    returning_intercept_voltages = []
    returning_intercept_currents = []
    
    if(is_Inductor):
        termination_length = Interface.data_input.Inductor_Length
        sending_wavefronts = Interface.data_output_ordered.Wavefronts_Sending_Inductor
        returning_wavefronts = Interface.data_output_ordered.Wavefronts_Returning_Inductor
    else:
        termination_length = Interface.data_input.Capacitor_Length
        sending_wavefronts = Interface.data_output_ordered.Wavefronts_Sending_Capacitor
        returning_wavefronts = Interface.data_output_ordered.Wavefronts_Returning_Capacitor

    for sending_wavefront,returning_wavefront in zip(sending_wavefronts,returning_wavefronts):
        
        # x = time enquirey
        # -s-> = sending wavefront
        # -r-> = returning wavefront
        
        # x-s->-r-> before 
        if(sending_wavefront.time_start > Time_Enquriey): # Finished
            break
        
        # -s->-r->x after
        elif(returning_wavefront.time_end <= Time_Enquriey): # Both DC
            dc_voltage += sending_wavefront.magnitude_voltage
            dc_current += sending_wavefront.magnitude_current
                
            dc_voltage += returning_wavefront.magnitude_voltage
            dc_current += returning_wavefront.magnitude_current
        
        # -s->-x-r-> returning intercept
        elif(returning_wavefront.time_end >= Time_Enquriey and returning_wavefront.time_start < Time_Enquriey): # Returning Intercept, Sending DC
            returning_intercept_positions.append(returning_wavefront.position_at_time(Time_Enquriey))
            returning_intercept_voltages.append(returning_wavefront.magnitude_voltage)
            returning_intercept_currents.append(returning_wavefront.magnitude_current)
                
            dc_voltage += sending_wavefront.magnitude_voltage
            dc_current += sending_wavefront.magnitude_current
        
        # -x-s->-r-> sending intercept
        elif(sending_wavefront.time_end >= Time_Enquriey and sending_wavefront.time_start <= Time_Enquriey): # Sending Intercept
            sending_intercept_positions.append(sending_wavefront.position_at_time(Time_Enquriey))
            sending_intercept_voltages.append(sending_wavefront.magnitude_voltage)
            sending_intercept_currents.append(sending_wavefront.magnitude_current)
                
        else:
            raise Exception("Somethings wrong, wavefront has to be intecepted/ stored or done")
    
    # setting dc values on the line 
    termination_voltage = dc_voltage
    interconnect_voltage =  dc_voltage
    termination_current = dc_current
    interconnect_current =  dc_current

    # combined intercept arrays
    intercept_positions = []
    intercept_voltage_left = []
    intercept_voltage_right = []
    intercept_current_left = []
    intercept_current_right = []

    # store sending intercept positions, 
    # initialise these positons to start with DC values
    for i, pos in enumerate(sending_intercept_positions):
        intercept_positions.append(pos)
            
        intercept_voltage_left.append(dc_voltage)
        intercept_voltage_right.append(dc_voltage)
        interconnect_voltage += sending_intercept_voltages[i]
            
        intercept_current_left.append(dc_current)
        intercept_current_right.append(dc_current)
        interconnect_current += sending_intercept_currents[i]
            
    # Next, store returning intercept positions, 
    # initialise these positons to start with DC values
    for i, pos in enumerate(returning_intercept_positions):
        intercept_positions.append(pos)
            
        intercept_voltage_left.append(dc_voltage)
        intercept_voltage_right.append(dc_voltage)
        termination_voltage += returning_intercept_voltages[i]
            
        intercept_current_left.append(dc_current)
        intercept_current_right.append(dc_current)
        termination_current += returning_intercept_currents[i]
            
    # for each intercept position account for sending and returning intercepts
    for i,position in enumerate(intercept_positions):
        for j, send_pos in enumerate(sending_intercept_positions):
            if(send_pos> position):
                intercept_voltage_left[i] += sending_intercept_voltages[j]
                intercept_voltage_right[i] += sending_intercept_voltages[j]
                    
                intercept_current_left[i] += sending_intercept_currents[j]
                intercept_current_right[i] += sending_intercept_currents[j]
                    
            if (send_pos == position ):
                intercept_voltage_left[i] += sending_intercept_voltages[j]
                
                intercept_current_left[i] += sending_intercept_currents[j]
                
        for j, return_pos in enumerate(returning_intercept_positions):
            if(return_pos< position):
                intercept_voltage_left[i] += returning_intercept_voltages[j]
                intercept_voltage_right[i] += returning_intercept_voltages[j]
                    
                intercept_current_left[i] += returning_intercept_currents[j]
                intercept_current_right[i] += returning_intercept_currents[j]
                    
            if (return_pos == position ):
                intercept_voltage_right[i] += returning_intercept_voltages[j]
                    
                intercept_current_right[i] += returning_intercept_currents[j]
                    
    # append interconnect values
    intercept_positions.append(0)
    intercept_voltage_left.append(interconnect_voltage)
    intercept_voltage_right.append(interconnect_voltage)
    intercept_current_left.append(interconnect_current)
    intercept_current_right.append(interconnect_current)

    # append termination values
    intercept_positions.append(termination_length)
    intercept_voltage_left.append(termination_voltage)
    intercept_voltage_right.append(termination_voltage)
    intercept_current_left.append(termination_current)
    intercept_current_right.append(termination_current)

    # sort values based on interconncet positions
    zip_positions_voltage_current = sorted(zip(intercept_positions,intercept_voltage_left,intercept_voltage_right,intercept_current_left,intercept_current_right))
    intercept_positions, intercept_voltage_left, intercept_voltage_right, intercept_current_left, intercept_current_right = zip(*zip_positions_voltage_current)
        
    # convert to lists
    intercept_positions = list(intercept_positions)
    intercept_voltage_left = list(intercept_voltage_left)
    intercept_voltage_right = list(intercept_voltage_right)
    intercept_current_left = list(intercept_current_left)
    intercept_current_right = list(intercept_current_right)
        
    # merge neighbours
    found_duplicate = True
    while found_duplicate:
        found_duplicate = False
        for index,position in enumerate(intercept_positions):
            if(index < len(intercept_positions)-1):
                
                if(position == intercept_positions[index+1]):                  
                    del intercept_positions[index +1]
                    del intercept_voltage_left[index +1]
                    del intercept_voltage_right[index +1]
                    del intercept_current_left[index +1]
                    del intercept_current_right[index +1]

                    found_duplicate = True
                        
    return intercept_positions,intercept_voltage_left,intercept_voltage_right,intercept_current_left,intercept_current_right