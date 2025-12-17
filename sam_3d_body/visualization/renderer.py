# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
from typing import List, Optional

import cv2
import numpy as np
import torch
import trimesh

# Try to import pyrender, but allow graceful fallback
try:
    import pyrender
    PYRENDER_AVAILABLE = True
    # Test if it actually works
    try:
        test_renderer = pyrender.OffscreenRenderer(100, 100)
        test_renderer.delete()
    except Exception:
        PYRENDER_AVAILABLE = False
except Exception:
    PYRENDER_AVAILABLE = False

if not PYRENDER_AVAILABLE:
    print("PyRender not available - using matplotlib software rendering")


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def create_raymond_lights() -> List:
    """
    Return raymond light nodes for the scene.
    """
    if not PYRENDER_AVAILABLE:
        return []
    
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )

    return nodes


class Renderer:

    def __init__(self, focal_length, faces=None):
        """
        Wrapper around the pyrender renderer to render meshes.
        Falls back to matplotlib if pyrender is unavailable.
        """
        self.focal_length = focal_length
        self.faces = faces
        self.pyrender_available = PYRENDER_AVAILABLE

    def _render_rgba_matplotlib(
        self,
        vertices,
        camera_translation,
        mesh_base_color=(1.0, 1.0, 0.9),
        rot_axis=[1, 0, 0],
        rot_angle=0,
        scene_bg_color=(0, 0, 0),
        render_res=[256, 256],
    ):
        """
        Render RGBA using matplotlib (software rendering)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # Apply transformations to vertices
        verts = vertices.copy() + camera_translation
        
        # Apply rotation
        if rot_angle != 0:
            rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
            verts = trimesh.transform_points(verts, rot)
        
        # Apply 180 degree flip
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        verts = trimesh.transform_points(verts, rot)
        
        # Create figure
        dpi = 100
        fig = plt.figure(figsize=(render_res[0]/dpi, render_res[1]/dpi), dpi=dpi)
        fig.patch.set_facecolor(scene_bg_color)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(scene_bg_color)
        
        # Create mesh collection
        mesh_data = []
        for face in self.faces:
            triangle = verts[face]
            mesh_data.append(triangle)
        
        # Add mesh to plot
        collection = Poly3DCollection(
            mesh_data,
            facecolors=mesh_base_color,
            edgecolors='none',
            alpha=1.0,
            shade=True,
            lightsource=None
        )
        ax.add_collection3d(collection)
        
        # Set limits based on vertices
        max_range = np.array([
            verts[:, 0].max() - verts[:, 0].min(),
            verts[:, 1].max() - verts[:, 1].min(),
            verts[:, 2].max() - verts[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
        mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
        mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set viewing angle
        ax.view_init(elev=0, azim=-90)
        
        # Remove axes, grid, and labels
        ax.set_axis_off()
        ax.grid(False)
        
        # Tight layout
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # Render to array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get RGBA array
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        plt.close(fig)
        
        # Resize if needed
        if rgba.shape[0] != render_res[1] or rgba.shape[1] != render_res[0]:
            rgba = cv2.resize(rgba, (render_res[0], render_res[1]))
        
        # Create alpha channel based on non-background pixels
        rgba_float = rgba.astype(np.float32) / 255.0
        
        # Create alpha mask: pixels that are not the background color
        bg_color = np.array([*scene_bg_color, 1.0])
        is_background = np.all(np.abs(rgba_float - bg_color) < 0.1, axis=2)
        rgba_float[:, :, 3] = (~is_background).astype(np.float32)
        
        return rgba_float

    def _render_trimesh_simple(
        self,
        vertices,
        image,
        mesh_base_color=(1.0, 1.0, 0.9),
        side_view=False,
        top_view=False,
        rot_angle=90,
    ):
        """
        Simple software rendering fallback using matplotlib
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        h, w = image.shape[:2]
        
        # Create figure with the same aspect ratio as image
        dpi = 100
        fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Apply rotations if needed
        verts = vertices.copy()
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0]
            )
            verts = trimesh.transform_points(verts, rot)
        elif top_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [1, 0, 0]
            )
            verts = trimesh.transform_points(verts, rot)
        
        # Flip for proper orientation
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        verts = trimesh.transform_points(verts, rot)
        
        # Plot the mesh
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=self.faces,
            color=mesh_base_color,
            alpha=0.9,
            edgecolor='none',
            shade=True
        )
        
        # Set viewing angle
        ax.view_init(elev=10, azim=90)
        
        # Remove axes
        ax.set_axis_off()
        ax.set_xlim([verts[:, 0].min(), verts[:, 0].max()])
        ax.set_ylim([verts[:, 1].min(), verts[:, 1].max()])
        ax.set_zlim([verts[:, 2].min(), verts[:, 2].max()])
        
        # Render to array
        fig.tight_layout(pad=0)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Convert to numpy array
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        plt.close(fig)
        
        # Resize to match image size if needed
        if rgba.shape[:2] != (h, w):
            rgba = cv2.resize(rgba, (w, h))
        
        # Composite with original image
        color = rgba.astype(np.float32) / 255.0
        alpha = color[:, :, 3:4]
        
        # Simple alpha blending
        output_img = color[:, :, :3] * alpha + (1 - alpha) * image
        
        return output_img.astype(np.float32)

    def __call__(
        self,
        vertices: np.array,
        cam_t: np.array,
        image: np.ndarray,
        full_frame: bool = False,
        imgname: Optional[str] = None,
        side_view=False,
        top_view=False,
        rot_angle=90,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        tri_color_lights=False,
        return_rgba=False,
        camera_center=None,
    ) -> np.array:
        """
        Render meshes on input image
        """
        
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)
        image = image / 255.0
        
        # Use trimesh fallback if pyrender not available
        if not self.pyrender_available:
            return self._render_trimesh_simple(
                vertices, image, mesh_base_color, side_view, top_view, rot_angle
            )
        
        # Original pyrender code
        try:
            h, w = image.shape[:2]

            renderer = pyrender.OffscreenRenderer(
                viewport_height=h,
                viewport_width=w,
            )

            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.0

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode="OPAQUE",
                baseColorFactor=(
                    mesh_base_color[2],
                    mesh_base_color[1],
                    mesh_base_color[0],
                    1.0,
                ),
            )
            mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

            if side_view:
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(rot_angle), [0, 1, 0]
                )
                mesh.apply_transform(rot)
            elif top_view:
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(rot_angle), [1, 0, 0]
                )
                mesh.apply_transform(rot)

            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)

            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

            scene = pyrender.Scene(
                bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
            )
            scene.add(mesh, "mesh")

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_translation
            if camera_center is None:
                camera_center = [image.shape[1] / 2.0, image.shape[0] / 2.0]
            camera = pyrender.IntrinsicsCamera(
                fx=self.focal_length,
                fy=self.focal_length,
                cx=camera_center[0],
                cy=camera_center[1],
                zfar=1e12,
            )
            scene.add(camera, pose=camera_pose)

            light_nodes = create_raymond_lights()
            if tri_color_lights:
                colors = [
                    np.array([1, 0.2, 0.3]),
                    np.array([0.2, 1, 0.2]),
                    np.array([0.2, 0.2, 1]),
                ]
                for ln, color in zip(light_nodes, colors):
                    ln.light.color = color
                    ln.light.intensity = 2.0

            for node in light_nodes:
                scene.add_node(node)

            color, _rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

            color = color.astype(np.float32) / 255.0
            renderer.delete()

            if return_rgba:
                return color

            valid_mask = (color[:, :, -1])[:, :, np.newaxis]
            output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image

            output_img = output_img.astype(np.float32)
            return output_img
            
        except Exception as e:
            print(f"PyRender failed: {e}")
            print("Falling back to trimesh rendering")
            self.pyrender_available = False
            return self._render_trimesh_simple(
                vertices, image, mesh_base_color, side_view, top_view, rot_angle
            )

    def vertices_to_trimesh(
        self,
        vertices,
        camera_translation,
        mesh_base_color=(1.0, 1.0, 0.9),
        rot_axis=[1, 0, 0],
        rot_angle=0,
    ):
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        mesh = trimesh.Trimesh(
            vertices.copy() + camera_translation,
            self.faces.copy(),
            vertex_colors=vertex_colors,
        )

        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
        self,
        vertices: np.array,
        cam_t=None,
        rot=None,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        camera_z=3,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        render_res=[256, 256],
    ):
        """
        Render mesh to RGBA image
        """
        
        if cam_t is not None:
            camera_translation = cam_t.copy()
        else:
            camera_translation = np.array(
                [0, 0, camera_z * self.focal_length / render_res[1]]
            )
        
        # Use matplotlib fallback if pyrender not available
        if not self.pyrender_available:
            print("Using matplotlib for render_rgba")
            return self._render_rgba_matplotlib(
                vertices,
                camera_translation,
                mesh_base_color,
                rot_axis,
                rot_angle,
                scene_bg_color,
                render_res,
            )

        try:
            renderer = pyrender.OffscreenRenderer(
                viewport_width=render_res[0], viewport_height=render_res[1], point_size=1.0
            )

            mesh = self.vertices_to_trimesh(
                vertices, camera_translation, mesh_base_color, rot_axis, rot_angle
            )
            mesh = pyrender.Mesh.from_trimesh(mesh)

            scene = pyrender.Scene(
                bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
            )
            scene.add(mesh, "mesh")

            camera_pose = np.eye(4)
            camera_center = [render_res[0] / 2.0, render_res[1] / 2.0]
            camera = pyrender.IntrinsicsCamera(
                fx=self.focal_length,
                fy=self.focal_length,
                cx=camera_center[0],
                cy=camera_center[1],
                zfar=1e12,
            )

            camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
            scene.add_node(camera_node)
            self.add_point_lighting(scene, camera_node)
            self.add_lighting(scene, camera_node)

            light_nodes = create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)

            color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            renderer.delete()

            return color
            
        except Exception as e:
            print(f"render_rgba with pyrender failed: {e}")
            print("Falling back to matplotlib rendering")
            self.pyrender_available = False
            return self._render_rgba_matplotlib(
                vertices,
                camera_translation,
                mesh_base_color,
                rot_axis,
                rot_angle,
                scene_bg_color,
                render_res,
            )

    def render_rgba_multiple(
        self,
        vertices: List[np.array],
        cam_t: List[np.array],
        rot_axis=[1, 0, 0],
        rot_angle=0,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        render_res=[256, 256],
        focal_length=None,
    ):
        if not self.pyrender_available:
            print("Warning: render_rgba_multiple not fully supported without pyrender")
            # Render just the first mesh as fallback
            if len(vertices) > 0 and len(cam_t) > 0:
                return self._render_rgba_matplotlib(
                    vertices[0],
                    cam_t[0],
                    mesh_base_color,
                    rot_axis,
                    rot_angle,
                    scene_bg_color,
                    render_res,
                )
            return np.zeros((render_res[1], render_res[0], 4), dtype=np.float32)

        try:
            renderer = pyrender.OffscreenRenderer(
                viewport_width=render_res[0], viewport_height=render_res[1], point_size=1.0
            )
            MESH_COLORS = [
                [0.000, 0.447, 0.741],
                [0.850, 0.325, 0.098],
                [0.929, 0.694, 0.125],
                [0.494, 0.184, 0.556],
                [0.466, 0.674, 0.188],
                [0.301, 0.745, 0.933],
            ]
            mesh_list = [
                pyrender.Mesh.from_trimesh(
                    self.vertices_to_trimesh(
                        vvv,
                        ttt.copy(),
                        MESH_COLORS[n % len(MESH_COLORS)],
                        rot_axis,
                        rot_angle,
                    )
                )
                for n, (vvv, ttt) in enumerate(zip(vertices, cam_t))
            ]

            scene = pyrender.Scene(
                bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
            )
            for i, mesh in enumerate(mesh_list):
                scene.add(mesh, f"mesh_{i}")

            camera_pose = np.eye(4)
            camera_center = [render_res[0] / 2.0, render_res[1] / 2.0]
            focal_length = focal_length if focal_length is not None else self.focal_length
            camera = pyrender.IntrinsicsCamera(
                fx=focal_length,
                fy=focal_length,
                cx=camera_center[0],
                cy=camera_center[1],
                zfar=1e12,
            )

            camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
            scene.add_node(camera_node)
            self.add_point_lighting(scene, camera_node)
            self.add_lighting(scene, camera_node)

            light_nodes = create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)

            color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            renderer.delete()

            return color
            
        except Exception as e:
            print(f"render_rgba_multiple failed: {e}")
            return np.zeros((render_res[1], render_res[0], 4), dtype=np.float32)

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        if not self.pyrender_available:
            return
            
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        if not self.pyrender_available:
            return
            
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)