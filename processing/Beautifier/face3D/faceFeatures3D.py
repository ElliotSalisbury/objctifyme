import eos
import os

import cv2
import numpy as np

from processing.Beautifier.faceFeatures import getLandmarks

EOS_SHARE_PATH = os.environ['EOS_DATA_PATH']

LANDMARK_IDS = list(map(str, range(1, 69)))  # generates the numbers 1 to 68, as strings


class FaceFitting:
    def __init__(self, model_path, blendshapes_path, landmarks_mapping_path, edge_topology_path, contour_landmarks_path,
                 model_contour_path):
        self.model = eos.morphablemodel.load_model(model_path)

        if self.model.get_expression_model() is None and blendshapes_path:
            blendshapes = eos.morphablemodel.load_blendshapes(blendshapes_path)
            self.model = eos.morphablemodel.MorphableModel(shape_model=self.model.get_shape_model(),
                                                           expression_model=blendshapes,
                                                           color_model=self.model.get_color_model(),
                                                           texture_coordinates=self.model.get_texture_coordinates())

        self.landmark_mapper = eos.core.LandmarkMapper(landmarks_mapping_path)

        self.edge_topology = eos.morphablemodel.load_edge_topology(edge_topology_path)
        self.contour_landmarks = eos.fitting.ContourLandmarks.load(contour_landmarks_path)
        self.model_contour = eos.fitting.ModelContour.load(model_contour_path)

        self.landmarks_2_vert_indices = [self.landmark_mapper.convert(l) for l in LANDMARK_IDS]
        self.landmarks_2_vert_indices = np.array([int(i) if i else -1 for i in self.landmarks_2_vert_indices])

    def getMeshFromLandmarks(self, landmarks, im, num_iterations=50, num_shape_coefficients_to_fit=-1,
                             shape_coeffs_guess=[], blendshape_coeffs_guess=[]):
        image_width = im.shape[1]
        image_height = im.shape[0]

        landmarks_eos = []
        for i in range(len(landmarks)):
            landmarks_eos.append(eos.core.Landmark(str(i+1), [float(landmarks[i][0]), float(landmarks[i][1])]))

        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphable_model=self.model,
                                                                                       landmarks=landmarks_eos,
                                                                                       landmark_mapper=self.landmark_mapper,
                                                                                       image_width=image_width, image_height=image_height,
                                                                                       edge_topology=self.edge_topology,
                                                                                       contour_landmarks=self.contour_landmarks,
                                                                                       model_contour=self.model_contour)
        return mesh, pose, shape_coeffs, blendshape_coeffs

    def getMeshFromMultiLandmarks(self, landmarkss, ims, num_iterations=5, num_shape_coefficients_to_fit=-1,
                                  shape_coeffs_guess=[], blendshape_coeffs_guess=[]):
        image_widths = []
        image_heights = []
        for im in ims:
            image_widths.append(im.shape[1])
            image_heights.append(im.shape[0])

        return self.getMeshFromMultiLandmarks_IWH(landmarkss, image_widths, image_heights,
                                                  num_iterations=num_iterations,
                                                  num_shape_coefficients_to_fit=num_shape_coefficients_to_fit,
                                                  shape_coeffs_guess=shape_coeffs_guess,
                                                  blendshape_coeffs_guess=blendshape_coeffs_guess)

    def getFaceFeatures3D(self, ims, landmarkss=None, num_iterations=5, num_shape_coefficients_to_fit=-1):
        imswlandmarks = []
        if landmarkss is None or len(ims) != len(landmarkss):
            landmarkss = []
            for im in ims:
                landmarks = getLandmarks(im)
                if landmarks is not None:
                    landmarkss.append(landmarks)
                    imswlandmarks.append(ims)
        else:
            imswlandmarks = ims

        if len(landmarkss) == 0:
            return None

        meshs, poses, shape_coeffs, blendshape_coeffs = self.getMeshFromMultiLandmarks(landmarkss, imswlandmarks,
                                                                                       num_iterations=num_iterations,
                                                                                       num_shape_coefficients_to_fit=num_shape_coefficients_to_fit)

        return shape_coeffs

    def getMeshFromShapeCeoffs(self, shape_coeffs=[], expression_coeffs=[], color_coeffs=[]):
        mesh = self.model.draw_sample(shape_coefficients=shape_coeffs, expression_coefficients=color_coeffs, color_coefficients=expression_coeffs)
        shape_verts = np.array(self.model.get_shape_model().draw_sample(shape_coeffs)).reshape([-1,3])
        expr_verts = np.array(self.model.get_expression_model().draw_sample(expression_coeffs)).reshape([-1,3])
        mesh.vertices = shape_verts + expr_verts
        mesh.colors = self.getColorModelFromColorCoeffs(color_coeffs)
        return mesh

    def getColorModelFromColorCoeffs(self, color_coeffs=[]):
        return np.array(self.model.get_color_model().draw_sample(color_coeffs)).reshape([-1,3])

    def getPoseFromMesh(self, landmarks, mesh, im):
        image_height = im.shape[0]
        image_width = im.shape[1]

        verts = np.array(mesh.vertices)
        image_points = []
        model_points = []
        for i, vert_id in enumerate(self.landmarks_2_vert_indices):
            if vert_id == -1:
                continue
            vert = np.ones(4)
            vert[:3] = verts[vert_id, :]

            image_points.append(landmarks[i])
            model_points.append(vert)

        ortho = eos.fitting.estimate_orthographic_projection_linear(image_points, model_points, True, image_height)
        pose = eos.fitting.RenderingParameters(ortho, image_width, image_height)
        return pose

    def getTextureFromMesh(self, image, mesh, pose, texture_size=1024):
        return eos.render.extract_texture(mesh, pose, image, compute_view_angle=False, isomap_resolution=texture_size)


# model = eos.morphablemodel.load_model()
# blendshapes = eos.morphablemodel.load_blendshapes()
# landmark_mapper = eos.core.LandmarkMapper()
# edge_topology = eos.morphablemodel.load_edge_topology()
# contour_landmarks = eos.fitting.ContourLandmarks.load()
# model_contour = eos.fitting.ModelContour.load()
# model_bfm = eos.morphablemodel.load_model(os.path.join(EOS_SHARE_PATH,"bfm_small.bin"))
# landmark_mapper_bfm = eos.core.LandmarkMapper(os.path.join(EOS_SHARE_PATH,"ibug_to_bfm_small.txt"))
# landmarks_2_vert_indices_bfm = [landmark_mapper_bfm.convert(l) for l in landmark_ids]
# landmarks_2_vert_indices_bfm = np.array([int(i) if i else -1 for i in landmarks_2_vert_indices_bfm])
SFM_FACEFITTING = FaceFitting(model_path=os.path.join(EOS_SHARE_PATH, "sfm_shape_3448.bin"),
                              blendshapes_path=os.path.join(EOS_SHARE_PATH, "expression_blendshapes_3448.bin"),
                              landmarks_mapping_path=os.path.join(EOS_SHARE_PATH, "ibug_to_sfm.txt"),
                              edge_topology_path=os.path.join(EOS_SHARE_PATH, "sfm_3448_edge_topology.json"),
                              contour_landmarks_path=os.path.join(EOS_SHARE_PATH, "ibug_to_sfm.txt"),
                              model_contour_path=os.path.join(EOS_SHARE_PATH, "sfm_model_contours.json"))

BFM_FACEFITTING = FaceFitting(model_path=os.path.join(EOS_SHARE_PATH, "bfm_expression.bin"),
                              blendshapes_path=None,  # os.path.join(EOS_SHARE_PATH, "expression_blendshapes_3448.bin"),
                              landmarks_mapping_path=os.path.join(EOS_SHARE_PATH, "ibug_to_bfm_expression.txt"),
                              edge_topology_path=os.path.join(EOS_SHARE_PATH,
                                                              "bfm_expression_edge_topology.json"),
                              contour_landmarks_path=os.path.join(EOS_SHARE_PATH, "ibug_to_bfm_expression.txt"),
                              model_contour_path=os.path.join(EOS_SHARE_PATH,
                                                              "bfm_expression_model_contours.json"))


def createTextureMap(mesh, pose, im):
    return eos.render.extract_texture(mesh, pose, im)


def exportMeshToJSON(mesh):
    verts = np.array(mesh.vertices)[:, 0:3].flatten().tolist()

    uvs = np.array(mesh.texcoords)
    uvs[:, 1] = 1 - uvs[:, 1]
    uvs = uvs.flatten().tolist()

    triangles = np.array(mesh.tvi)
    faces = np.zeros((triangles.shape[0], 1 + 3 + 3), dtype=triangles.dtype)
    faces[:, 0] = 8
    faces[:, 1:4] = triangles
    faces[:, 4:7] = triangles
    faces = faces.flatten().tolist()

    outdata = {}
    outdata["metadata"] = {
        "version": 4,
        "type": "geometry",
        "generator": "GeometryExporter"
    }
    outdata["vertices"] = verts
    outdata["uvs"] = [uvs]
    outdata["faces"] = faces

    return outdata


def ensureImageLessThanMax(im, maxsize=512):
    height, width, depth = im.shape
    if height > maxsize or width > maxsize:

        if width > height:
            ratio = maxsize / float(width)
            width = maxsize
            height = int(height * ratio)
        else:
            ratio = maxsize / float(height)
            height = maxsize
            width = int(width * ratio)
        im = cv2.resize(im, (width, height))
    return im
