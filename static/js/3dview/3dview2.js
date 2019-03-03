var camera, controls, scene, renderer;
var model;

var currMesh;
var default_face_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
var default_color_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
var default_expr_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

function downloadMsgPack(url, callback) {
    var oReq = new XMLHttpRequest();
    oReq.open("GET", url, true);
    oReq.responseType = "arraybuffer";

    oReq.onload = function (oEvent) {
        var arrayBuffer = oReq.response; // Note: not oReq.responseText
        if (arrayBuffer) {
            var byteArray = new Uint8Array(arrayBuffer);
            var data = msgpack.decode(byteArray);
            callback(data);
        }
    };
    oReq.send();
}

function modelReady(data) {
    model = data;
    console.log("model downloaded");

    var meshverts = meshFromFeatures(model, default_face_features, default_expr_features).valueOf();
    var meshcolors = colorFromFeatures(model, default_color_features).valueOf();
    createMesh(meshverts, model.faces, model.UVs, meshcolors);
}

function meshFromFeatures(model, features, expr_features) {
    alpha = math.matrix(features);
    var shape = math.multiply(model.shape.PC, alpha);
    shape = math.add(model.shape.MU, shape);

    if (expr_features !== undefined) {
        alpha = math.matrix(expr_features);
        var expr = math.multiply(model.expression.PC, alpha);
        expr = math.add(model.expression.MU, expr);
        shape = math.add(shape, expr);
    }

    var numVert = parseInt(shape._size[0] / 3);
    var verts = shape.reshape([numVert, 3]);

    return verts;
}

function colorFromFeatures(model, features) {
    alpha = math.matrix(features);
    var colors = math.multiply(model.color.PC, alpha);
    colors = math.add(model.color.MU, colors);
    colors = math.divide(colors, 255);

    var numVert = parseInt(colors._size[0] / 3);
    var verts = colors.reshape([numVert, 3]);

    return verts;
}

function createMesh(verts, faces, uvs, colors) {
    scene.remove(currMesh);

    var geometry = new THREE.Geometry();
    geometry.dynamic = true;

    for (var i = 0; i < verts.length; i++) {
        var vert = verts[i];
        geometry.vertices.push(new THREE.Vector3(vert[0], vert[1], vert[2]));

        var v_color = colors[i];
        geometry.colors.push(new THREE.Color(v_color[0], v_color[1], v_color[2]));
    }
    for (var i = 0; i < faces.length; i++) {
        var face = faces[i];
        var geo_face = new THREE.Face3(face[0], face[1], face[2]);

        geo_face.vertexColors[0] = geometry.colors[face[0]];
        geo_face.vertexColors[1] = geometry.colors[face[1]];
        geo_face.vertexColors[2] = geometry.colors[face[2]];

        geometry.faces.push(geo_face);
    }

    geometry.computeFaceNormals();

    // var material = new THREE.MeshPhongMaterial( { map: THREE.ImageUtils.loadTexture('img/isomap.jpg') } );
    // var material = new THREE.MeshNormalMaterial();
    var material = new THREE.MeshBasicMaterial({vertexColors: THREE.VertexColors});
    currMesh = new THREE.Mesh(geometry, material);
    scene.add(currMesh);
}

function init_3d(canvas_id, pca_model_url) {
    var container = document.getElementById(canvas_id);

    var w = 512;//container.clientWidth;
    var h = 512;//container.clientHeight;

    scene = new THREE.Scene();

    renderer = new THREE.WebGLRenderer();
    renderer.setClearColor(0x444444);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);


    container.appendChild(renderer.domElement);

    camera = new THREE.OrthographicCamera(w / -2, w / 2, h / 2, h / -2, 1, 1000);

    camera.position.x = 50;
    camera.position.y = 0;
    camera.position.z = 200;
    camera.zoom = 2.4;
    camera.updateProjectionMatrix();


    light = new THREE.AmbientLight(0xcccccc);
    scene.add(light);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target = new THREE.Vector3(0, 0, 100);
    controls.enableDamping = true;
    controls.dampingFactor = 1.25;
    controls.enableZoom = true;

    downloadMsgPack(pca_model_url, modelReady);


    // handle the resizing of the webgl container
    function onWindowResize() {
        resize_container(container);
    }
    window.addEventListener('resize', onWindowResize, false);
}

function resize_container(container) {
    var w = 512;//container.clientWidth;
    var h = 512;//container.clientHeight;

    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    renderer.setSize(w, h);
}


function animate() {
    requestAnimationFrame(animate);

    controls.update(); // required if controls.enableDamping = true, or if controls.autoRotate = true

    render();
}

var prevTime = Date.now();
function render() {
    renderer.render(scene, camera);
}