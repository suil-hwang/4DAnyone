# fdanyone/geometry/ops.py
import roma
import torch

def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].clone().flatten(-2)


def rotation_6d_to_matrix(value):
    return roma.special_gramschmidt(value.unflatten(-1, (2, 3)).mT, epsilon=1e-12).mT


def matrix_to_quaternion(matrix):
    quat = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(matrix))
    return torch.where(quat[..., :1] < 0, -quat, quat)


def quaternion_to_matrix(quat):
    return roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(quat / quat.norm(dim=-1, keepdim=True)))


def quaternion_to_axis_angle(quat):
    return roma.unitquat_to_rotvec(roma.quat_wxyz_to_xyzw(quat), shortest_arc=False)


def euler_angles_to_matrix(angles, convention):
    return roma.euler_to_rotmat(convention, angles)


def knn_points(p1, p2, *, K=1, return_nn=False):
    distances, indices = torch.cdist(p1, p2).square().topk(K, dim=-1, largest=False)
    neighbors = None
    if return_nn:
        points = p2[:, None].expand(-1, p1.shape[1], -1, -1)
        neighbors = points.gather(2, indices[..., None].expand(-1, -1, -1, p2.shape[-1]))
    return distances, indices, neighbors
