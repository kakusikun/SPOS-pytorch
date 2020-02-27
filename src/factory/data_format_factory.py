from src.database.data_format.classification import build_image_dataset

class DataFormatFactory:
    products = {
        'classification': build_image_dataset,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, 
        cfg, 
        data=None, 
        data_format=None,
        transform=None, 
        build_func=None,
        return_indice=False):

        if cfg.DB.DATA_FORMAT not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.DB.DATA_FORMAT if data_format is None else data_format](
                        data=data,
                        transform=transform,
                        build_func=build_func,          # coco
                        return_indice=return_indice     # reid
                    )
