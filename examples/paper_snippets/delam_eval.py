from deladect.detection import DelaminationDetector

# creates the interface object for the delamination detection
# the specimen s is already defined in Section 3.1
interface = s.add_interface(name="i0", lower_ply=ply0, upper_ply=ply90)

# detector object
detector = DelaminationDetector(s, interface)

edge_params = {"window_edge": (1, 60),
               "seed_ratio": 0.01}


diffuse_params = {"window_diffuse": (30, 30),
                   "diffuse_dx": 20.0,
                   "diffuse_dy": 20.0}

# runs the detection of both edge and diffuse delaminations
result = detector.detect_both_delaminations(
    cracks=cracks,
    edge_params=edge_params,
    diffuse_params=diffuse_params,
)
