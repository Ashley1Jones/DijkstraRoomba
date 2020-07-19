# DijkstraRoomba
This code uses VREP's python API to simulate a robot navigating an office environment.  The workflow is as such;
- Take Depth Map from VREP (VREP.py)
- Calculate XY coordinates (PointCloud.py)
- Calculate angles of point cloud groups (PointCloud.py)
- Plot angles for memory (Mapping.py)
- Find closest unsearched area (closest is judged by cost value) (Navigation.py)
- Send desired wheel speeds back to (VREP.py)

This code uses multiprocessing to share cpu load.  The current setup passses on values serially so it is not completely utlising the cpu's potential; however, this can be changed.  Lastly, extra stuff with object detection (YOLO.py) is currently being tested for later work.
