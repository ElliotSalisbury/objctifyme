import pickle

import numpy as np
import msgpack
import os
import scipy.io
from processing.Beautifier.face3D.faceFeatures3D import BFM_FACEFITTING

def export_model_to_JSON(model):
    shape_pca_model = model.get_shape_model()
    color_pca_model = model.get_color_model()
    expression_pca_model = model.get_expression_model()

    outdata = {}
    outdata["shape"] = export_eos_PCA_model_to_JSON(shape_pca_model)
    outdata["color"] = export_eos_PCA_model_to_JSON(color_pca_model)
    outdata["expression"] = export_eos_PCA_model_to_JSON(expression_pca_model)

    outdata["faces"] = np.array(shape_pca_model.get_triangle_list()).tolist()
    outdata["UVs"] = np.array(model.get_texture_coordinates()).tolist()

    return outdata


def export_eos_PCA_model_to_JSON(pca_model):
    outdata = {}
    outdata["EV"] = np.array(pca_model.get_eigenvalues()).tolist()
    outdata["PC"] = np.array(pca_model.get_rescaled_pca_basis()).tolist()
    outdata["MU"] = np.array(pca_model.get_mean()).tolist()

    return outdata

def export_sklearn_PCA_model_to_JSON(pca):
    outdata = {}
    outdata["EV"] = np.array(pca.explained_variance_).tolist()
    outdata["PC"] = np.array(pca.components_).tolist()
    outdata["MU"] = np.array(pca.mean_).tolist()
    return outdata

if __name__ == "__main__":
    model_bfm = BFM_FACEFITTING.model
    pca_path = "pca.p"
    outpath = "."
    tex_pca = None

    if os.path.exists(pca_path):
        print("texture pca found")
        with open(pca_path, "rb") as file:
            tex_pca = pickle.load(file)
    else:
        print("NO texture pca found")


    with open(os.path.join(outpath, "bfm.msg"), 'wb') as outfile:
        modeljson = export_model_to_JSON(model_bfm)

        # if tex_pca:
        #     modeljson["texture"] = export_sklearn_PCA_model_to_JSON(tex_pca)
        msgpack.dump(modeljson, outfile)