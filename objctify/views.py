from django.http import HttpResponse, HttpResponseServerError
from django.shortcuts import render_to_response, render
from django.db import transaction
from django.template import RequestContext

from objctify.tasks import faceBeautificationTask
from objctify.models import ImageProcessingRequest, UploadedImage
from ipware.ip import get_ip
import json

FACE_BEAUTIFICATION = "1"

TASKS = {FACE_BEAUTIFICATION: faceBeautificationTask}

# @transaction.atomic()
def save_upload_request(request, type, images, task):
    ipr = ImageProcessingRequest(ip=get_ip(request), type=type)
    ipr.save()
    uis = []
    for image in images:
        ui = UploadedImage(image=image, filename=image.name, request=ipr)
        ui.save()
        uis.append(ui)

    return ipr, uis

def index(request):
    return render(request, 'objctify/index.html')
    # return render_to_response('objctify/index.html', context_instance=RequestContext(request))
def about(request):
    return render_to_response('objctify/about.html')
def author(request):
    return render_to_response('objctify/author.html')

def averageFaces(request):
    return render_to_response('objctify/3D/averageFaces.html')
def morphFaces(request):
    return render_to_response('objctify/3D/morphFaces.html')

def upload(request):
    if request.method == 'POST':
        type = request.POST.get("type", FACE_BEAUTIFICATION)
        gender = request.POST.get("gender", "F")
        images = request.FILES.getlist('images')

        IPR, UIs = save_upload_request(request, type, images, TASKS[type])
        UIids = [ui.id for ui in UIs]

        result = TASKS[type].apply_async((UIids, gender), expires=60 * 3)
        request.session["taskId"] = result.task_id

        reply = {"type": type, "taskId":result.task_id,}# "result":result}

        return HttpResponse(json.dumps(reply), content_type="application/json")
    else:
        return HttpResponseServerError("Must Use POST")

def startImageProcessing(request):
    imageb64 = request.POST.get("imageb64", None)
    type = request.POST.get("taskType", FACE_BEAUTIFICATION)

    #we didnt get the uploaded image, return an error
    if imageb64 is None:
        return HttpResponseServerError("Image Upload Error")

    #send an image to be processed, but ignore the task if its taking longer than 3 minutes
    result = TASKS[type].apply_async((imageb64,), expires=60*3)

    return HttpResponse(result.id, content_type="text/plain")

def getSwap(request):
    taskId = request.GET.get("taskId", None)
    type = request.GET.get("type", FACE_BEAUTIFICATION)

    reply = {"taskId":taskId, "type":type}

    if taskId == -1 and 'taskId' in request.session:
        taskId = request.session["taskId"]

    if taskId and taskId != -1:
        #get the results from celery
        result = TASKS[type].AsyncResult(taskId)
        reply["status"] = result.status

        if result.status == "FAILURE":
            reply["error"] = str(result.result)
        elif result.status == "SUCCESS":
            #get the results
            reply["result"] = result.get()
            #if no image has been returned (probably no faces)
            if reply["result"] is None:
                #return the task as failed, so that JS stops polling
                reply["status"] = "FAILURE"
                reply["error"] = "No image was returned from the processing task."
        return HttpResponse(json.dumps(reply), content_type="application/json")

    return HttpResponseServerError("No TaskId")