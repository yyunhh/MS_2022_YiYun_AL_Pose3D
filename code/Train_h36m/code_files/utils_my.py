
"""
1012
- Ref: https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/lib/utils/image.py

- 用於dataloader_my.py裡面的function

"""

import numpy as np
import cv2

def flip(img):
  return img[:, :, ::-1].copy()


def shuffle_lr(x, shuffle_ref):
    for e in shuffle_ref:
        x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy() # 交換位置


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_h = scale_tmp[1]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

'''

# 投射
# https://opencv-python-tutorials.readthedocs.io/zh/latest/4.%20OpenCV%E4%B8%AD%E7%9A%84%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/4.2.%20%E5%9B%BE%E5%83%8F%E7%9A%84%E5%87%A0%E4%BD%95%E5%8F%98%E6%8D%A2/
def get_affine_transform(center, scale, rot, output_size, shift = np.array([0, 0], dtype=np.float32), inv=0):

    
    #shift: [0, 0] /shape: (2,)
    
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale =np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size [0]
    dst_h = output_size [1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w*-0.5], rot_rad)
    dst_dir = np.array([0, dst_w*-0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32) # [[0.,0.], [0.,0.], [0.,0.]]
    dst = np.zeros((3, 2), dtype=np.float32) # [[0.,0.], [0.,0.], [0.,0.]]
    #print(src[0,:].shape)# (2,)
    #print(center.shape) # (2,) None
    #print(center) #[595.67426 520.89325]
    #print(scale_tmp.shape) # (2, 2) None
    #print(shift) #[0. 0.]
    #exit()
    # print(scale_tmp)
    # [[340.1340873 340.1340873]
    # [340.1340873 340.1340873]]
    src[0, :] = center + scale_tmp * shift  # center(2,) + scale_tmp(2,2)*shift() = [595.67426 520.89325] + [[340.1340873 340.1340873] [340.1340873 340.1340873]] * [0. 0.]
    src[1, :] = center + src_dir + scale_tmp * shift # src[1,:].shape -> (2,)

    dst[0, :] = [dst_w*0.5, dst_h*0.5]
    dst[1, :] = np.array([dst_w*0.5, dst_h*0.5])+dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
'''

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a-b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    # print(new_pt)
    # exit()
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size): #coords:? type(center)? type(scale)? output_size:? #(joint, 0:1)
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)

    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)


    return target_coords


def crop(img, center, scale, output_size, rot = 0):
    trans = get_affine_transform(center, scale, rot, output_size)
    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    try:
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    except:
        print('center', center)
        print('gx, gy', g_x, g_y)
        print('img_x, img_y', img_x, img_y)
    return heatmap

def adjust_aspect_ratio(s, aspect_ratio, fit_short_side=False):
    w, h = s[0], s[1]
    if w > aspect_ratio * h:
        if fit_short_side:
            w = h * aspect_ratio
        else:
            h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        if fit_short_side:
            h = w * 1.0 / aspect_ratio
        else:
            w = h * aspect_ratio
    return np.array([w, h])
