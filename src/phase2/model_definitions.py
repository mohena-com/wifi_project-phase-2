from config_reader import ConfigReader

from DS_WifiCSIDataset import WifiCSIDataset
from DL_CSILSTMNet import CSILSTMNet
from DL_DenseNet1D import DenseNet1D
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_MobileNetV3 import MobileNetV3_1D_LSTM



cr = ConfigReader("csi_id_config.properties")

list_batch_size = cr.get_int_list("batch_size")
list_learning_rate = cr.get_float_list("learning_rate")
list_optimizer = cr.get_list("optimizer", type_func=str)
list_weight_decay = cr.get_float_list("weight_decay")   
list_epochs = cr.get_int_list("epochs")


var_model_definitions = {
    "CSILSTMNet": (
        CSILSTMNet,
        {
            'csi_input_size': [99],
            'meta_input_size': [12],
            'window_size': [128],
            'num_classes': [31],
            'lr': list_learning_rate,
            'batch_size': list_batch_size,
            'optimizer': list_optimizer,
            'weight_decay': list_weight_decay,
            'epochs': list_epochs,
        },
    ),
    "DenseNet1D": (
        DenseNet1D,
        {
            'csi_channels': [99],
            'meta_feature_dim': [12],
            'num_classes': [31],
            'lr': list_learning_rate,
            'batch_size': list_batch_size,
            'optimizer': list_optimizer,
            'weight_decay': list_weight_decay,
            'epochs': list_epochs,
        },
    ),
    "EfficientNet1DLSTM": (
        EfficientNet1DLSTM,
        {
            'in_channels': [99],
            'meta_seq_len': [128],
            'meta_feature_dim': [12],
            'num_classes': [31],
            'lr': list_learning_rate,
            'batch_size': list_batch_size,
            'optimizer': list_optimizer,
            'weight_decay': list_weight_decay,
            'epochs': list_epochs,
        },
    ),
    "MobileNetV3_1D_LSTM": (
        MobileNetV3_1D_LSTM,
        {
            'csi_channels': [99],
            'meta_feature_dim': [12],
            'num_classes': [31],
            'lr': list_learning_rate,
            'batch_size': list_batch_size,
            'optimizer': list_optimizer,
            'weight_decay': list_weight_decay,
            'epochs': list_epochs,
        },
    ),
}