import rembg
import numpy as np
import open3d as o3d
from PIL import Image
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from transformers import GLPNImageProcessor, GLPNForDepthEstimation



def remove_background(input_path="input_image.jpg", output_path="output_image.jpg"):
    input_image = Image.open(input_path)
    input_array = np.array(input_image)
    output_array = rembg.remove(input_array)
    output_image = Image.fromarray(output_array)
    output_image = output_image.convert("RGB") ## was in RGBA scale
    output_image.save(output_path)

def visualize_obj_model(model_path):
    """Visualizes a .obj 3D model using Open3D."""
    mesh = o3d.io.read_triangle_mesh(model_path)
    if not mesh.has_vertex_normals():
        print("Mesh has no normals.Computing normals......")
        mesh.compute_vertex_normals()
    #Mesh Smoothing: Apply smoothing algorithms to the mesh for better visual quality.
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    o3d.visualization.draw_geometries([mesh], window_name="3D Model Viewer")


# Load image
remove_background()
image_path = "output_image.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Load feature extractor and model
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")


# Prepare image for model
inputs = feature_extractor(images=image, return_tensors="pt")

# Run the model
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Resize to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],  # (height, width)
    mode="bicubic",
    align_corners=False,
).squeeze().cpu().numpy()

# Normalize depth for visualization
depth_min = prediction.min()
depth_max = prediction.max()
normalized_depth = (prediction - depth_min) / (depth_max - depth_min)


#Get original image and depth map
rgb_image = np.array(image)  # From PIL image
depth_map = normalized_depth.astype(np.float32)

# Convert to Open3D image format
rgb_o3d = o3d.geometry.Image(rgb_image)
depth_o3d = o3d.geometry.Image((depth_map * 1000).astype(np.uint16))  # Scale to millimeters

# Create RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color=rgb_o3d,
    depth=depth_o3d,
    convert_rgb_to_intensity=False,
    depth_scale=1000.0,  # Match depth_map * 1000  # 1000 for mm -> meters, adjust if needed
    depth_trunc=3.0      # Truncate distances beyond 3 meters
)

# Define camera intrinsics
# Focal lengths and principal point (assumes fx = fy and center = image center)
## Assuming a simple pinhole camera model

height, width = depth_map.shape
fx = fy = 0.8 * width  # You can tune this value
cx = width / 2
cy = height / 2
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

#  Generate point cloud
## Calculate 3D coordinates
#z = depth_map
#x = (x - cx) * z / fx
#y = (y - cy) * z / fy
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# Outlier Removal: After generating the point cloud, remove outliers to improve mesh quality.
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Flip it (Open3D uses different coordinate system)
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])


pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

## this is from github 
#pcd.orient_normals_to_align_with_direction()



# Mesh generation (Poisson)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10,n_threads=1)
#mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)



# Crop mesh to the original bounding box
bbox = pcd.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

#rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
#mesh.rotate(rotation, center=(0, 0, 0))

# Save as .obj
o3d.io.write_triangle_mesh("mesh_output.obj", mesh)
#o3d.io.write_triangle_mesh("mesh_output.obj", pcd)

print("Mesh saved as mesh_output.obj")

# Visualize the point cloud
#o3d.visualization.draw_geometries([pcd]) ## the results looks very goood 
visualize_obj_model("mesh_output.obj")


