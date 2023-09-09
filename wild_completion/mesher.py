import open3d as o3d
import torch
from wild_completion.utils import ForceKeyErrorDict, create_voxel_grid, convert_sdf_voxels_to_mesh, decode_sdf, get_time

class MeshExtractor(object):
    def __init__(self, decoder, code_len=64, voxels_dim=64, cube_radius=1.0):
        self.decoder = decoder
        self.code_len = code_len
        self.voxels_dim = voxels_dim
        self.cube_radius = cube_radius
        with torch.no_grad():
            self.voxel_points = create_voxel_grid(vol_dim=self.voxels_dim).cuda() * self.cube_radius # in the [-1,1] cube

    def extract_mesh_from_code(self, code):
        # input code should already be torch tensor
        start = get_time()
        latent_vector = code
        sdf_tensor = decode_sdf(self.decoder, latent_vector, self.voxel_points)
        vertices, faces = convert_sdf_voxels_to_mesh(sdf_tensor.view(self.voxels_dim, self.voxels_dim, self.voxels_dim), self.cube_radius)
        vertices = vertices.astype("float32")
        faces = faces.astype("int32")
        end = get_time()
        print("Extract mesh takes %f seconds" % (end - start))
        return ForceKeyErrorDict(vertices=vertices, faces=faces)

    def complete_mesh(self, latent, transform, color):
        cur_mesh = self.extract_mesh_from_code(latent)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cur_mesh.vertices), o3d.utility.Vector3iVector(cur_mesh.faces))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color)
        mesh_o3d = mesh_o3d.transform(transform)

        return mesh_o3d