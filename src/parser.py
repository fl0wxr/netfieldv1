
import json
import sys
from copy import deepcopy
import idx2numpy
import tensor_module

from cnn import cnn


class parse_platform_configuration:

    def __init__(self):

        cnn_locations_node_format = 'json'

        ## Separating options from locations
        self.loc_arguments = []
        self.options = []
        for _ in range(len(sys.argv)):
            if '--' in sys.argv[_]:
                self.options.append(sys.argv[_][2:])
            else:
                self.loc_arguments.append(sys.argv[_])

        self.tm_ = tensor_module.cpu()
        for _ in range(len(self.options)):
            if 'gpu' == self.options[_]:
                ## This variable stores the object corresponding a module responsible for the tensor array type and its manupulations.
                self.tm_ = tensor_module.gpu()
                break

        ## Contains various important locations of files
        self.loc_cnn_locations_node = self.loc_arguments[1]
        assert self.loc_cnn_locations_node.split('.')[-1] == cnn_locations_node_format, 'Exception: Wrong configuration path or format. File format has to be in %s.'%cnn_locations_node_format

        with open(self.loc_cnn_locations_node, 'r') as file:
            self.cnn_locations_node = json.load(file)

        self.normalize_input = bool(self.cnn_locations_node['normalize_input'])

        ## Format specification of the dataset
        self.dataset_parsing_mode = self.cnn_locations_node['dataset_parsing_mode']

        ## Location of .txt file containing the CNNs label name for each class
        self.loc_class_names = self.cnn_locations_node['class_names']

        ## Location of .json file containing the CNNs configuration
        self.loc_cnn_config = self.cnn_locations_node['cnn_config']

        ## File location from where the trainable parameters can be loaded
        self.loc_lparams_load_file = self.cnn_locations_node['lparams_load_file']

        ## Directory location where the trainable parameters can be saved
        self.loc_lparams_save_dir = self.cnn_locations_node['lparams_save_dir']

        self.class_names = parse_class_names(self.tm_, self.loc_class_names)

        self.training_config, self.layers_hparams = parse_cnn_config(self.loc_cnn_config)

        self.convnet = cnn(self.tm_, self.normalize_input, self.class_names, self.layers_hparams, self.loc_lparams_load_file, self.loc_lparams_save_dir)

    def parse_training_config(self):
        '''
        Description:
            Initializes a CNN object that includes its training configuration variables.
        '''

        ## Location of .txt file containing the locations of each training set element
        loc_training_features_lc = self.cnn_locations_node['training_set']['training_features']
        loc_training_target_variables_lc = self.cnn_locations_node['training_set']['training_target_variables']

        ## Location of .txt file containing the locations of each cross validation set element
        loc_val_features_lc = self.cnn_locations_node['val_set']['val_features']
        loc_val_target_variables_lc = self.cnn_locations_node['val_set']['val_target_variables']

        self.training_set = form_dataset(self.tm_, self.class_names, self.dataset_parsing_mode, loc_training_features_lc, loc_training_target_variables_lc, self.layers_hparams[0][1])
        self.training_set = (self.training_set[0], self.training_set[1]) ## [:10*(2**7)] (!)

        self.val_set = form_dataset(self.tm_, self.class_names, self.dataset_parsing_mode, loc_val_features_lc, loc_val_target_variables_lc, self.layers_hparams[0][1])
        self.val_set = (self.val_set[0], self.val_set[1]) ## [:2*(2**7)] (!)

        # print('WARNING: A data import limit has been added for debugging purposes! Please proceed with repairing this change when the debugging is completed.')

        self.convnet.train_store(self.training_config, self.training_set, self.val_set)

        return self.convnet

    def parse_classification_config(self):
        '''
        Description:
            Initializes a CNN object that includes its classification configuration variables.
        '''

        feature_data_loc = self.loc_arguments[2]

        feature_data = parse_featureset(self.tm_, self.dataset_parsing_mode, [feature_data_loc])
        self.convnet.classify_store(feature_data)

        return self.convnet


def form_dataset(tm_, class_names, dataset_parsing_mode, loc_features_lc, loc_target_variables_lc, input_layer_hparams):

    ## ! Parsing the locations of each individual example: Begin

    features_lc = filetext2listelement(loc_features_lc)
    target_variables_lc = filetext2listelement(loc_target_variables_lc)

    ## ! Parsing the locations of each individual example: End

    ## Verifying lengths
    assert len(features_lc) == len(target_variables_lc), 'Exception: Unequal dataset sizes between features and target variables.'

    ## ! Parsing each individual example/file: Begin

    features = parse_featureset(tm_, dataset_parsing_mode, features_lc)

    target_variables = parse_labelset(tm_, dataset_parsing_mode, target_variables_lc, class_names)

    ## ! Parsing each individual example/file: End

    features_example_shape = features[0].shape

    ## Checking first data element with the input shape given on the CNN configuration file (all data elements have equal shapes so this is means that it checks all the example shapes on the dataset)
    assert input_layer_hparams['input_height'] == features_example_shape[0] and input_layer_hparams['input_width'] == features_example_shape[1] and input_layer_hparams['input_channels'] == features_example_shape[2] , 'Exception: Input shape mismatch detected. The dataset is consisted by examples of different shape than the one given in the configuration file on its input layer.'

    return (features, target_variables)

def parse_cnn_config(loc_cnn_config):

    with open(loc_cnn_config, 'r') as file:
        cnn_config = json.load(file, object_pairs_hook=list)

    ## CNN configuration structure
    # [cnn_config_struct, cnn_essential_config_groups] = cnn_config_structure()

    ## Checking config names syntax
    # check_config_syntax(cnn_config, cnn_config_struct, cnn_essential_config_groups)
    training_config = training_config_format_build(cnn_config[0][1])
    del cnn_config[0]

    layers_hparams = layer_format_build(cnn_config)
    del cnn_config

    return (training_config, layers_hparams)

def parse_class_names(tm_, loc_class_names):

    class_names = filetext2listelement(loc_class_names)

    return tm_.np.array(class_names)

def parse_featureset(tm_, parsing_mode, features_lc):
    '''
    Inputs:
        <tm_>: module, used for tensor manipulations.
        <parsing_mode>: str, can only take 'agent_wise' and 'four_files'. The 'agent_wise' option refers to the occassion where each file is an image example and the 'four_files' option refers to the occassion where the dataset is stored in a single file.
        <features_lc>: list, contains the feature location(s).
    '''

    if parsing_mode=='agent_wise':
        dataset_size = len(features_lc)
        features = tm_.tm.array([tm_.tm.load(features_lc[i]) for i in range(dataset_size)])

    elif parsing_mode=='four_files':
        if features_lc[0].split('.')[-1] != 'npy':
            features = idx2numpy.convert_from_file(features_lc[0])
        else:
            features = tm_.np.load(features_lc[0])

        features = tm_.tm.array(features)

    if len(features[0].shape) == 2:
        features = tm_.tm.expand_dims(features, axis=-1)
        tm_.memory_deall()

    return features

def parse_labelset(tm_, parsing_mode, target_variables_lc, class_names):
    '''
    Inputs:
        <tm_>: module, used for tensor manipulations.
        <target_variables_lc>: list, contains the target variable location(s).
        <class_names>: list, contains the class name stings.
        <parsing_mode>: str, can only take 'agent_wise' and 'four_files'. The 'agent_wise' option refers to the occassion where each file is an image example and the 'four_files' option refers to the occassion where the dataset is stored in a single file.
    '''

    if parsing_mode=='agent_wise':
        dataset_size = len(target_variables_lc)

        target_variables = [None for i in range(dataset_size)]
        for i in range(dataset_size):
            with open(target_variables_lc[i], 'r') as label_file_pointer:
                label = label_file_pointer.read().rstrip('\n')
            ## Convert to one hot encoding format
            target_variables[i] = label == class_names
        target_variables = tm_.tm.array(target_variables)


    elif parsing_mode=='four_files':
        if target_variables_lc[0].split('.')[-1] != 'npy':
            target_variables_raw = idx2numpy.convert_from_file(target_variables_lc[0])
        else:
            target_variables_raw = tm_.np.load(target_variables_lc[0])

        target_variables_raw = tm_.tm.array(target_variables_raw).astype(tm_.tm.int64)
        # target_variables_raw = tm_.tm.load(target_variables_lc[0]).astype(tm_.tm.int64)
        ## Convert to one hot encoding format
        target_variables = tm_.tm.zeros((target_variables_raw.size, len(class_names)))

        target_variables[tm_.tm.arange(target_variables_raw.size), target_variables_raw] = 1

    return target_variables

def layer_format_build(in_list):

    out_list = [None for idx in range(len(in_list))]
    for idx in range(len(in_list)):
        out_list[idx] = [in_list[idx][0], dict(in_list[idx][1])]

    return out_list

def training_config_format_build(in_list):

    out_dict = {}
    out_dict = dict(in_list)
    for key1 in out_dict.keys():
        if type(out_dict[key1]) is list:
            out_dict[key1] = dict(out_dict[key1])
        if type(out_dict[key1]) is dict:
            for key2 in out_dict[key1].keys():
                if type(out_dict[key1][key2]) is list:
                    out_dict[key1][key2] = dict(out_dict[key1][key2])

    return out_dict

def filetext2listelement(loc):
    with open(loc, 'r') as file:
        str_ = file.read()

    if str_[-1] == '\n':
        str_ = str_[:-1]

    list_ = str_.split('\n')

    return list_
