from deladect.specimen import Specimen
from deladect.detection import crack_analysis

# specimen object
s = Specimen(
    name="sample-1",
    scale_px_mm=31.953,
    path_full="data/sample-1",
    image_types=["png"]
)

# add plies to the specimen
ply90 = s.add_ply(name="ply_90", orientation_deg=90.0, avg_crack_width_px=8.0, min_crack_length_px=90.0)

ply0 = s.add_ply(name="ply_0", orientation_deg=0.0, avg_crack_width_px=8.0, min_crack_length_px=90.0)

# crack detection for the 0 and 90 deg plies
cracks = crack_analysis(s, save_cracks=True)
