from parser import parse_platform_configuration


if __name__ == '__main__':

    platform_config_obj = parse_platform_configuration()

    convnet = platform_config_obj.parse_classification_config(); del platform_config_obj

    convnet.classify()

#
