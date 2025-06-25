class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'crackseg9k':
            return 'C:\\Users\\ajc3xc\ml\\datasets\\Final-Dataset-Vol1'
        elif dataset == 'concrete3k':
            return 'C:\\Users\\ajc3xc\\ml\\datasets\\concrete3k\\concrete3k'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
