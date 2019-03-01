import numpy as np
import cv2
from processing.Beautifier.warpFace import warpFace, warpTriangle
from processing.Beautifier.face3D.opengl import render_mesh
# from opengltests2 import render_mesh

MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
ALL_FACE_LANDMARKS = MOUTH_POINTS + RIGHT_BROW_POINTS + LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS

def project(p, modelview, proj, viewport):
    tmp = modelview * np.append(p,1)[:, np.newaxis]
    tmp = proj * tmp

    tmp = tmp/tmp[3]
    tmp = tmp*0.5 + 0.5
    tmp[0] = tmp[0] * viewport[2] + viewport[0]
    tmp[1] = tmp[1] * viewport[3] + viewport[1]

    return np.array(tmp[0:2]).flatten()

def projectVertsTo2D(verts, pose, image):
    modelview = np.matrix(pose.get_modelview())
    proj = np.matrix(pose.get_projection())
    viewport = np.array([0,image.shape[0], image.shape[1], -image.shape[0]])

    verts2d = np.zeros((verts.shape[0],2),dtype=np.float64)
    for i, vert in enumerate(verts):
        verts2d[i,:] = project(vert, modelview, proj, viewport)

    return verts2d

def projectMeshTo2D(mesh, pose, image):
    verts = np.array(mesh.vertices)
    return projectVertsTo2D(verts, pose, image)

def getVisibleFacesIndexs(mesh, pose):
    verts = np.array(mesh.vertices)[:,:3]
    norms = np.ones((len(mesh.tvi), 4), dtype=np.float64)
    modelview = np.matrix(pose.get_modelview())
    modelview[0, 3] = 0
    modelview[1, 3] = 0

    for i, triangle in enumerate(mesh.tvi):
        p0, p1, p2 = verts[triangle]
        v1 = p1 - p0
        v2 = p2 - p1

        norm = np.cross(v1,v2)
        norm = norm / np.linalg.norm(norm)
        norms[i,:3] = norm

    rotatedNorms = np.zeros_like(norms)
    for i, norm in enumerate(norms):
        rotatedNorms[i] = np.array(modelview * norm[:, np.newaxis]).flatten()

    return np.where(rotatedNorms[:,2] > 0)

def renderFaceTo2D(im, mesh, pose, isomap):
    verts2d = projectMeshTo2D(mesh, pose, im)
    uvcoords = np.array(mesh.texcoords) * np.array([isomap.shape[1], isomap.shape[0]])

    visibleFaceIndexs = getVisibleFacesIndexs(mesh, pose)
    visibleVertIndexs = np.unique(np.array(mesh.tvi)[visibleFaceIndexs].flatten())

    renderedFace = warpFace(isomap[:,:,:3], uvcoords[visibleVertIndexs], verts2d[visibleVertIndexs], justFace=True, output_shape=(im.shape[0], im.shape[1]))
    meshFace = drawMesh(im, mesh, pose, isomap)
    cv2.imshow("orig", im)
    cv2.imshow("mesh", meshFace)


    blackIs = np.where(
        np.logical_and(np.logical_and(renderedFace[:, :, 0] == 0, renderedFace[:, :, 1] == 0), renderedFace[:, :, 2] == 0))
    renderedFace[blackIs] = im[blackIs]
    cv2.imshow("rendered", renderedFace)

    # warpFace3D(im, mesh, pose, newMesh)

    renderedFace2 = np.zeros_like(im)
    for i, triangle in enumerate(np.array(mesh.tvi)[visibleFaceIndexs]):
        # if i > 500:
        #     break
        srcT = uvcoords[triangle].astype(np.int64)
        dstT = verts2d[triangle].astype(np.int64)

        renderedFace2 = warpTriangle(isomap[:,:,:3], renderedFace2, srcT, dstT)
    cv2.imshow("rendered2", renderedFace2.astype(np.uint8))

    cv2.waitKey(-1)

def drawMesh(im, mesh, pose):
    verts2d = projectMeshTo2D(mesh, pose, im)
    visibleFaceIndexs = getVisibleFacesIndexs(mesh, pose)

    drawIm = im.copy()
    for triangle in np.array(mesh.tvi)[visibleFaceIndexs]:
        p0, p1, p2 = verts2d[triangle].astype(np.int64)

        p02d = (p0[0], p0[1])
        p12d = (p1[0], p1[1])
        p22d = (p2[0], p2[1])

        cv2.line(drawIm, p02d, p12d, (0, 255, 0), thickness=1)
        cv2.line(drawIm, p12d, p22d, (0, 255, 0), thickness=1)
        cv2.line(drawIm, p22d, p02d, (0, 255, 0), thickness=1)

    return drawIm

def warpFace3D(im, oldMesh, pose, newMesh, accurate=True, fitter=None):
    oldVerts2d_full = projectMeshTo2D(oldMesh, pose, im)
    newVerts2d_full = projectMeshTo2D(newMesh, pose, im)

    visibleFaceIndexs = getVisibleFacesIndexs(oldMesh, pose)

    if not accurate and fitter is not None:
        ALL_FACE_MESH_VERTS = fitter.landmarks_2_vert_indices[ALL_FACE_LANDMARKS]
        ALL_FACE_MESH_VERTS = np.delete(ALL_FACE_MESH_VERTS, np.where(ALL_FACE_MESH_VERTS == -1)).tolist()

        oldConvexHullIndexs = cv2.convexHull(oldVerts2d_full.astype(np.float32), returnPoints=False)
        warpPointIndexs = oldConvexHullIndexs.flatten().tolist() + ALL_FACE_MESH_VERTS

        oldVerts2d = oldVerts2d_full[warpPointIndexs]
        newVerts2d = newVerts2d_full[warpPointIndexs]
    else:
        visibleVertIndexs = np.unique(np.array(oldMesh.tvi)[visibleFaceIndexs].flatten())

        oldVerts2d = oldVerts2d_full[visibleVertIndexs]
        newVerts2d = newVerts2d_full[visibleVertIndexs]

    warpedIm = warpFace(im, oldVerts2d, newVerts2d)
    # warpedIm = im.copy()
    color_delta = getColorDelta(im, oldMesh, newMesh, pose)

    #apply the color delta
    warpedIm = np.clip(warpedIm + color_delta, 0, 255).astype(np.uint8)

    return warpedIm

def getColorDelta(im, oldMesh, newMesh, pose):
    print("getting color delta")

    new_colors = np.array(newMesh.colors).copy()
    newMesh.colors = np.array(oldMesh.colors).copy()
    im_old = render_mesh(im, newMesh, pose)

    newMesh.colors = new_colors
    im_new = render_mesh(im, newMesh, pose)

    delta = im_new.astype(np.float64) - im_old.astype(np.float64)

    return delta

def drawLandmarksMapper(im, landmarks, mesh, pose, facefitting, color=(0,255,0)):
    newIm = im.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    verts = np.array(mesh.vertices)

    for i, vert_id in enumerate(facefitting.landmarks_2_vert_indices):
        if vert_id == -1:
            continue

        vert = np.array([verts[vert_id, :],])
        landmark = projectVertsTo2D(vert, pose, im)[0].astype(np.int)

        cv2.circle(newIm, tuple(landmarks[i]), 2, (255, 0, 0), -1)
        cv2.line(newIm, tuple(landmarks[i]), tuple(landmark), (255,0,0), thickness=1)

        cv2.putText(newIm, str(i), tuple(landmark), font, 0.25, color, 1, cv2.LINE_AA)
        cv2.circle(newIm, tuple(landmark), 2, color, -1)




    return newIm

if __name__ == '__main__':
    from processing.Beautifier.faceFeatures import getLandmarks
    from processing.Beautifier.face3D.faceFeatures3D import SFM_FACEFITTING, BFM_FACEFITTING

    image_orig = cv2.imread(r"C:\eos\eos2\examples\data\image_0010.png")
    image_orig = cv2.resize(image_orig, (400,400))
    landmarks = getLandmarks(image_orig)

    mesh, pose, shape_coeffs, blendshape_coeffs = BFM_FACEFITTING.getMeshFromLandmarks(landmarks, image_orig)
    mesh2, pose, shape_coeffs, blendshape_coeffs = BFM_FACEFITTING.getMeshFromLandmarks(landmarks, image_orig)
    mesh.colors = BFM_FACEFITTING.getColorModelFromColorCoeffs([])
    mesh2.colors = BFM_FACEFITTING.getColorModelFromColorCoeffs([3])

    warped = warpFace3D(image_orig, mesh, pose, mesh2, accurate=False, fitter=BFM_FACEFITTING)
    cv2.imshow("orig", image_orig)
    cv2.imshow("warped", warped)
    cv2.waitKey(-1)