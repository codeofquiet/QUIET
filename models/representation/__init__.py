from models.representation.RealNN import RealNN
from models.representation.QDNN import QDNN
#from models.representation.QDNN-Copy1 import QDNN
from models.representation.ComplexNN import ComplexNN
from models.representation.QDNNAblation import QDNNAblation
from models.representation.LocalMixtureNN import LocalMixtureNN
from models.representation.QDNN_double_real import QDNN_double_real
from models.representation.QDNN_three_slot import QDNN_three_slot

def setup(opt):
    print("representation network type: " + opt.network_type)
    if opt.network_type == "qdnn":
        print('qdnn--------------------')
        model = QDNN(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
