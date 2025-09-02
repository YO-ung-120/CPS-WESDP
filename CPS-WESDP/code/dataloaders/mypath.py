class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            #return './szVOCdevkit/VOC2012/'
            #return './lidcVOCdevkit1/lidcVOCdevkit1/VOC2012/'
            return './VOCdevkit/VOC2012/'
            #return './pzVOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'cityscapes':
            return 'ImageData/CityScapesDataset/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError