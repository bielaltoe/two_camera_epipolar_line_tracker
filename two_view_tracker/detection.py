class ObjectDetection:
    def __init__(self, cam, id, bbox, frame, centroids_list, single_centroid, name):
        self.cam = cam
        self.id = id
        self.bbox = bbox
        self.frame = frame
        self.centroids_list = centroids_list
        self.single_centroid = single_centroid
        self.name = name
        self.color = None
        

    def __str__(self):
        return f"ObjectDetection(cam={self.cam}, id={self.id}, bbox={self.bbox}, frame={self.frame}, centroids_list={self.centroids_list}, single_centroid={self.single_centroid})"

