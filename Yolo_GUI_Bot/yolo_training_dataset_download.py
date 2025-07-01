from roboflow import Roboflow
rf = Roboflow(api_key="sDGj59B8nxqppNo4zubo")
project = rf.workspace("web-detection").project("thirdeye")
dataset = project.version(2).download("yolov8")
quit()


from roboflow import Roboflow
rf = Roboflow(api_key="sDGj59B8nxqppNo4zubo")
project = rf.workspace("utkarsh-bb0fd").project("ui-f13oe")
dataset = project.version(1).download("yolov8")
quit()

from roboflow import Roboflow
rf = Roboflow(api_key="sDGj59B8nxqppNo4zubo")
project = rf.workspace("thiago-dantas-h3fz8").project("ui-detection")
dataset = project.version(2).download("yolov8")

quit()

from roboflow import Roboflow
rf = Roboflow(api_key="sDGj59B8nxqppNo4zubo")
project = rf.workspace("bot-wtice").project("clicker")
dataset = project.version(1).download("yolov8")
