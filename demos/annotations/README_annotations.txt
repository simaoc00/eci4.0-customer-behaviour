==========================
Annotation Files
==========================

Annotations files include three types of annotations per clip
1) Event file (selected events annotated)
2) Mapping file (from event to object)
3) Object file (all objects annotated)

The detailed formats of the above three types of (linked) files are described below.


1) Event file format

Files are named as '%s.viratdata.events.txt' where %s is clip id.
Each line in event file captures information about a bounding box of an event at the corresponding frame

Event File Columns 
1: event ID        (unique identifier per event within a clip, same eid can exist on different clips)
2: event type      (event type)
3: duration        (event duration in frames)
4: start frame     (start frame of the event)
5: end frame       (end frame of the event)
6: current frame   (current frame number)
7: bbox lefttop x  (horizontal x coordinate of left top of bbox, origin is lefttop of the frame)
8: bbox lefttop y  (vertical y coordinate of left top of bbox, origin is lefttop of the frame)
9: bbox width      (horizontal width of the bbox)
10: bbox height    (vertical height of the bbox)

Event Type ID (for column 2 above)
1: Person loading an Object to a Vehicle
2: Person Unloading an Object from a Car/Vehicle
3: Person Opening a Vehicle/Car Trunk
4: Person Closing a Vehicle/Car Trunk
5: Person getting into a Vehicle
6: Person getting out of a Vehicle
7: Person gesturing
8: Person digging
9: Person carrying an object
10: Person running
11: Person entering a facility
12: Person exiting a facility


2) Object file format

Files are named as '%s.viratdata.objects.txt'
Each line captures informabiont about a bounding box of an object (person/car etc) at the corresponding frame.
Each object track is assigned a unique 'object id' identifier. 
Note that:
- an object may be moving or static (e.g., parked car).
- an object track may be fragmented into multiple tracks.

Object File Columns
1: Object id        (a unique identifier of an object track. Unique within a file.)
2: Object duration  (duration of the object track)
3: Currnet frame    (corresponding frame number)
4: bbox lefttop x   (horizontal x coordinate of the left top of bbox, origin is lefttop of the frame)
5: bbox lefttop y   (vertical y coordinate of the left top of bbox, origin is lefttop of the frame)
6: bbox width       (horizontal width of the bbox)
7: bbox height      (vertical height of the bbox)
8: Objct Type       (object type)

Object Type ID (for column 8 above for object files)
1: person
2: car              (usually passenger vehicles such as sedan, truck)
3: vehicles         (vehicles other than usual passenger cars. Examples include construction vehicles)
4: object           (neither car or person, usually carried objects)
5: bike, bicylces   (may include engine-powered auto-bikes)


3) Mapping file format

Files are named as '%s.viratdata.mapping.txt'
Each line in mapping file captures information between an event (in event file) and associated objects (in object file)

Mapping File Columns
1: event ID         (unique event ID, points to column 1 of event file)
2: event type       (event type, points to column 2 of event file)
3: event duration   (event duration, points to column 3 of event file)
4: start frame      (start frame of event)
5: end frame        (end frame of event)
6: number of obj    (total number of associated objects)
7-end:              (variable number of columns which captures the associations maps for variable number of objects in the clip. 
                     If '1', the event is associated with the object. Otherwise, if '0', there's none.
                     The corresponding oid in object file can be found by 'column number - 7')