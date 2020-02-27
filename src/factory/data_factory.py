from src.database.data.emotion import Emotion

class DataFactory:
    products = {
        'emotion': Emotion
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, data_name=None):
        if cfg.DB.DATA not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.DB.DATA if data_name is None else data_name](
                        path=cfg.DB.PATH,
                        branch=cfg.DB.DATA if data_name is None else data_name,
                        coco_target=cfg.COCO.TARGET,
                        num_keypoints=cfg.DB.NUM_KEYPOINTS,
                        num_classes=cfg.DB.NUM_CLASSES,
                        output_stride=cfg.MODEL.STRIDE,
                        is_merge=cfg.REID.MERGE,
                        use_train=cfg.DB.USE_TRAIN,
                        use_test=cfg.DB.USE_TEST,
                    )

