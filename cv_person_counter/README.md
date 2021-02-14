
<h1 align="center">
  Personal Counter (Computer Vision)
</h1>


#### Overview

This project performs how to count number of person entering the store using Computer Vision. Detecting where is the human in video data, tracking objects, recognize that person entered the store and counting are the primary functions demonstrated in this project.

#### General Block Diagram

This is the block diagram expresses the project flow.

<img src="https://github.com/carfirst125/portfolio/blob/main/cv_person_counter/image/cv_person_counter_BlockDiagram.png?raw=true"/>

The data will be got from Camera IP, splits into frame, implements some data processing such as Color Conversion, Image resizing before Object Detection.

The **YOLOv3** is used in this demo project for Object Detection. Here, that is human object. The YOLOv3 already trained with human detection, so dont need to retrain, just load the model and its weights and apply. 

**Deep Sort** is a extreme powerful algorithm for Object Tracking. This permits you to track an object via multi-frame of video. This helps you in tracking moving path of the object. It is the foundation for you to analyse, count and process to consult who enters the store.


#### Implemetation DEMO


*(click to view)*

[![DEMO](https://img.youtube.com/vi/oYkED5rL1X8/mqdefault.jpg)](https://youtu.be/watch?v=oYkED5rL1X8 "Click to view")

#### Play around

**Step 1**: Clone the project

**Step 2**: Extract code.rar in the same folder

**Step 3**: Download YoLo Model at http://bit.ly/37aefvK , extract and copy-paste ./model_data folder under ./code folder.
       
**Step 4**: Put your input video under *./video folder* with name *input_video.mp4*. You need to change checker (rule for checking if a person entered store) in person_counter_1_3.py to count. For sample input video, please email me.

**Step 5**: Open Pycharm, create new project, create env with python=3.7 and *pip install -r requirements.txt* 

**Step 6**: Run command: *python person_counter_1_3.py*
       
- - - - - 
Feeling Free to contact me if you have any question around.

    Nhan Thanh Ngo
    Email: ngothanhnhan125@gmail.com
    Skype: ngothanhnhan125
    Phone: (+84) 938005052






