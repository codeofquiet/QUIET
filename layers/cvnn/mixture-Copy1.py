import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras import layers as La
from keras.models import Model, Input



class ComplexMixture(Layer):

    def __init__(self, average_weights=False, **kwargs):
        # self.output_dim = output_dim
        self.average_weights = average_weights
        super(ComplexMixture, self).__init__(**kwargs)


    def get_config(self):
        config = {'average_weights': self.average_weights}
        base_config = super(ComplexMixture, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(input_shape) != 3 and len(input_shape) != 2 :
             raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')


        super(ComplexMixture, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        
#         #################
#         print("experiment/qnn/layers/cvnn/mixture.py ----inputs")
#         print(inputs)
#         print('\n')
#         #################

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(inputs) != 3 and len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2/3 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        
        ndims = len(inputs[0].shape)###句子的长度
        input_real = K.expand_dims(inputs[0]) #shape: (None, 60, 300, 1)
        input_imag = K.expand_dims(inputs[1]) #shape: (None, 60, 300, 1)
        ###input 分别为 seq_embedding_real seq_embedding_imag self.weight
        
#         #################
        print("experiment/qnn/layers/cvnn/mixture.py ----input_real input_imag weight ndims")
        print(input_real)
        print(input_imag)
        print(inputs[2])
        print(ndims)
        print('\n')
#         #################
        
        input_real_transpose = K.expand_dims(inputs[0], axis = ndims-1) #shape: (None, 60, 1, 300)
        input_imag_transpose = K.expand_dims(inputs[1], axis = ndims-1) #shape: (None, 60, 1, 300)
        
#         #################
        print("experiment/qnn/layers/cvnn/mixture.py ----input_real_transpose input_imag_transpose")
        print(input_imag_transpose)
        print(input_imag_transpose)
        print('\n')
#         #################

        ###
        #output = (input_real+i*input_imag)(input_real_transpose-i*input_imag_transpose)
        ###
        ## 原来的代码 ##output_real = K.batch_dot(input_real_transpose,input_real, axes = [ndims-1,ndims])+ K.batch_dot(input_imag_transpose, input_imag,axes = [ndims-1,ndims])  #shape: (None, 60, 300, 300)
        output_real = La.multiply([input_real_transpose,input_real])+ La.multiply([input_imag_transpose, input_imag])
        
        ## 原来的代码 ##output_imag = K.batch_dot(input_real_transpose, input_imag,axes = [ndims-1,ndims])-K.batch_dot(input_imag_transpose,input_real, axes = [ndims-1,ndims])  #shape: (None, 60, 300, 300)
        output_imag = La.multiply([input_real_transpose, input_imag])-La.multiply([input_imag_transpose,input_real])
#         ############
        print("experiment/qnn/layers/cvnn/mixture.py ---- output_real output_imag first time")
        print(output_real)
        print(output_imag)
        print('\n')
#         ############

        # output_real = K.permute_dimensions(output_real, (0,2,3,1))
        # weight = K.permute_dimensions(weight, (0,2,1))
        # print(weight.shape)
        # print(output_real.shape)
        if self.average_weights:
            print('experiment/qnn/layers/cvnn/mixture.py self.average_weights yes')
            output_r = K.mean(output_real, axis = ndims-2, keepdims = False)
            output_i = K.mean(output_imag, axis = ndims-2, keepdims = False)
#             ############
#             print("experiment/qnn/layers/cvnn/mixture.py ---- output_r output_i")
#             print(output_r)
#             print(output_i)
#             print('\n')
#             ############
        else:
            #For embedding layer inputs[2] is (None, embedding_dim,1)
            #For test inputs[2] is (None, embedding_dim)
            if len(inputs[2].shape) == ndims-1:
                ############
                print("im in the first weight")
                ############
                weight = K.expand_dims(K.expand_dims(inputs[2]))
#                 ############
#                 print(weight)
#                 ############
                weight = K.repeat_elements(weight, output_real.shape[-1], axis = ndims-1)
#                 ############
#                 print(weight)
#                 print('\n')
#                 ############
            else:
                ############
                ###运行的是这个
                print("im in the second weight\n")
                ############
                weight = K.expand_dims(inputs[2])
                ############
#                 print(weight)
                ############
#             ##############
#             print("experiment/qnn/layers/cvnn/mixture.py ---- weight")
#             print(weight)
#             print('\n')
#             ##############
            weight = K.repeat_elements(weight, output_real.shape[-1], axis = ndims)#######最后一维的数据重复了一遍 由 42 1 50 1 变成 42 1 50 50
#             #################打印排错 单独运行本文件时，连这里都没有打印
#             print("experiment/qnn/layers/cvnn/mixture.py ----output_real weight")
#             print(output_real)
#             print(weight)
#             print('\n')
#             #################
            #####这一行报错了
            output_real = output_real*weight #shape: (None, 300, 300)
            print("qnn/layers/cvnn/mixture.py output_real after multy")
            print(output_real)
            print('\n')
            
            ####在ndims-2这个维度进行求和,在第0维度进行求和，也就是第56维度进行求和，维度变为？ 50 50
            output_r = K.sum(output_real, axis = ndims-2)
            print("qnn/layers/cvnn/mixture.py output_r after sum")
            print(output_r)
            print('\n')

            output_imag = output_imag*weight
            output_i = K.sum(output_imag, axis = ndims-2)

        return [output_r, output_i]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        one_input_shape = list(input_shape[0])
        one_output_shape = []
        for i in range(len(one_input_shape)):
            if not i== len(one_input_shape)-2:
                one_output_shape.append(one_input_shape[i])
        one_output_shape.append(one_output_shape[-1])
#        one_output_shape = [one_input_shape[0], one_input_shape[2], one_input_shape[2]]
#        print(one_output_shape)
#        print('Input shape of mixture layer:{}'.format(input_shape))
#        print([tuple(one_output_shape), tuple(one_output_shape)])
        return [tuple(one_output_shape), tuple(one_output_shape)]


def main():
    input_2 = Input(shape=(3,5,), dtype='float')
    input_1 = Input(shape=(3,5,), dtype='float')
    weights = Input(shape = (3,), dtype = 'float')
    #[output_1, output_2] = ComplexMixture(average_weights = True)([input_1, input_2, weights])
    [output_1, output_2] = ComplexMixture( )([input_1, input_2, weights])


    model = Model([input_1, input_2, weights], [output_1, output_2])
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()

    x = np.random.random((3,3,5))
    x_2 = np.random.random((3,3,5))
    weights = np.random.random((3,3))
    output = model.predict([x,x_2, weights])
    print(output)

    # print(x)
    # print(x_2)
    # output = model.predict([x,x_2])
    # print(output)

if __name__ == '__main__':
    main()
