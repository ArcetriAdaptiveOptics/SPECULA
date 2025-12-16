from specula import np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.recmat import Recmat


class MirrorCommandsCombinator(BaseProcessingObj):
    '''Mirror Commands Combinator'''

    def __init__(self,
                 k_vector,
                 recmat: Recmat,
                 dims_LO: list=[],
                 dims_P: int=1,
                 dims_F: int=1,
                 out_dims: list=[],
                 # for all processing objects:                 
                 target_device_idx: int=None,
                 precision: int=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)        
                
        self.A_matrix = recmat
        self.k_vector = self.xp.asarray(k_vector)
        
        self.dims_LO = np.array(dims_LO, dtype=np.int32)    # 2, 0, 3
        self.dims_P = dims_P                
        self.dims_F = dims_F                                # 1
        self.out_dims = np.array(out_dims, dtype=np.int32)  # 4000, 1000, 1000

        self.dims_HO = self.out_dims - self.dims_LO
        self.dims_HO[0] -= self.dims_F                      # 3997, 1000, 997
        
        self.dims_LO_cum = np.cumsum(self.dims_LO)
        self.dims_HO_cum = np.cumsum(self.dims_HO)

        self.result_commands1 = BaseValue('First chunk of output commands',
                                          target_device_idx=target_device_idx,
                                          precision=precision)
        self.result_commands2 = BaseValue('Second chunk of output commands',
                                          target_device_idx=target_device_idx,
                                          precision=precision)
        self.result_commands3 = BaseValue('Third chunk of output commands',
                                          target_device_idx=target_device_idx,
                                          precision=precision)

        self.inputs['in_commandsHO'] = InputValue(type=BaseValue)      # could be 6000 elements, dims[0]
        self.inputs['in_commandsLO'] = InputValue(type=BaseValue)      # could be 5 elements, dims[1]
        self.inputs['in_commandsF'] = InputValue(type=BaseValue)       # could be 1 element, dims[2]
        self.inputs['in_commandsP'] = InputValue(type=BaseValue)       # could be 6 elements, dims[3]
                
        self.outputs['out_result_commands1'] = self.result_commands1       # could be 4000, out_dims[0]
        self.outputs['out_result_commands2'] = self.result_commands2       # could be 1000, out_dims[1]
        self.outputs['out_result_commands3'] = self.result_commands3       # could be 1000, out_dims[2]

        # Pre-Allocate output data
        self.result_commands1.value = self.xp.zeros(self.out_dims[0], dtype=self.dtype)
        self.result_commands2.value = self.xp.zeros(self.out_dims[1], dtype=self.dtype)
        self.result_commands3.value = self.xp.zeros(self.out_dims[2], dtype=self.dtype)        

        self.z1 = self.xp.zeros( self.dims_LO[0], dtype=self.dtype)
        self.z2 = self.xp.zeros( self.out_dims[0]-self.dims_LO[0]-self.dims_LO[2], dtype=self.dtype)

    def trigger_code(self):
        x_HO = self.local_inputs['in_commandsHO'].value
        x_LO = self.local_inputs['in_commandsLO'].value
        x_F = self.local_inputs['in_commandsF'].value
        x_P = self.local_inputs['in_commandsP'].value

        x_LO1 = x_LO[0:self.dims_LO_cum[0]]
        # this is a Null vector for now: 
        # x_LO2 = x_LO[self.dims_LO_cum[0]:self.dims_LO_cum[1]]
        x_LO3 = x_LO[self.dims_LO_cum[1]:self.dims_LO_cum[2]]

        x_HO1 = x_HO[0:self.dims_HO_cum[0]]
        x_HO2 = x_HO[self.dims_HO_cum[0]:self.dims_HO_cum[1]]
        x_HO3 = x_HO[self.dims_HO_cum[1]:self.dims_HO_cum[2]]

        v11 = self.xp.concatenate( ( x_LO1, x_F, x_HO1 ) )
        v12 = self.xp.concatenate( ( self.z1, self.k_vector * x_LO3, self.z2 ) )

        y1 = v11 + v12
        if self.dims_P > 0:
            y1 += self.A_matrix.recmat[:self.out_dims[0],:] @ x_P        
        y2 = x_HO2
        y3 = self.xp.concatenate( ( x_LO3, x_HO3 ) )

        #print(f'{len(y1)=}, {len(y2)=}, {len(y3)=}')
        self.result_commands1.value[:] = y1
        self.result_commands2.value[:] = y2
        self.result_commands3.value[:] = y3

    def post_trigger(self):
        super().post_trigger()
        # note that this cannot be done in the trigger when stream is used
        self.result_commands1.generation_time = self.current_time
        self.result_commands2.generation_time = self.current_time
        self.result_commands3.generation_time = self.current_time
