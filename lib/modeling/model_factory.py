from modeling.model_builder import Generalized_RCNN
from modeling.context_model_builder import Context_RCNN
from modeling.contrast_model_builder import CC_RCNN
from modeling.CE3D_model_builder import CE3D_RCNN
from modeling.multi_modality_builder import MULTI_MODALITY_RCNN
from modeling.mm_test_builder import MM_TEST_RCNN
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg


def GetRCNNModel():
    """ RCNN model factory """
    if cfg.CONTEXT_ROI_POOLING.ENABLED:
        maskRCNN = Context_RCNN()
    elif cfg.CONTRASTED_CONTEXT.ENABLED:
        maskRCNN = CC_RCNN()
    elif cfg.LESION.USE_3DCE:
        maskRCNN = CE3D_RCNN()
    elif cfg.LESION.MULTI_MODALITY:
        maskRCNN = MULTI_MODALITY_RCNN()
    elif cfg.LESION.MM_TEST:
        maskRCNN = MM_TEST_RCNN()
    else:
        maskRCNN = Generalized_RCNN()

    return maskRCNN

