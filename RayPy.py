import matplotlib.pyplot as plt
import numpy as np
import math
import time
from PIL import Image
import os
from multiprocessing import Pool
import random

# OPTIONS SETUP:
# WARNING: if glossy OR antialiasing enable REMEMBER to halved image resolution
enable_glossy = False
enable_antialiasing = False
enable_texture = True
enable_obj = False
only_obj = False


class Camera:
    def __init__(self, position, lookat, cameraup, fov, ar):
        self.position = position
        self.lookAt = lookat
        self.cameraUp = cameraup.normalize()
        self.cameraW = (self.lookAt - self.position).normalize()
        self.cameraU = self.cameraW.cross(self.cameraUp).normalize()
        self.cameraV = self.cameraU.cross(self.cameraW).normalize()
        self.fov = fov
        self.ar = ar
        self.h = math.tan(fov)
        self.w = self.h * ar

    def ray_cam2ray(self, x, y):
        dir = self.cameraW + (x *self.w * self.cameraU) + \
                (y * self.h * self.cameraV)
        return Ray(self.position, dir)


class Scene:
    def __init__(self, camera, objects, lights):
        self.camera = camera
        self.objects = objects
        self.lights = lights

    def get_intersection(self, ray):
        """
        Check if ray hit an object in scene
        """
        hit = None
        for obj in self.objects:
            dist = obj.intersect(ray)
            if dist is not None and (hit is None or dist < hit[1]):
                hit = obj, dist
        return hit


class Light:
    def __init__(self, point, color):
        self.point = point
        self.color = color
        self.radius = 1

    def intersect(self, ray):
        """
        Intersection between ray and sphere if exist
        otherwise return 'None'
        """
        T = ray.O - self.point
        a = ray.D ** 2
        b = 2 * ray.D * T
        c = T ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            delta = ( -b - math.sqrt(delta)) * .5
            if delta > 0:
                return delta


class Sphere:
    """
    Sphere object
    """
    def __init__(self, origin, radius, material):
        self.origin = origin
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """
        Intersection between ray and sphere if exist
        otherwise return 'None'
        """
        T = ray.O - self.origin
        a = ray.D ** 2
        b = 2 * ray.D * T
        c = T ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            delta = ( -b - math.sqrt(delta)) * .5
            if delta > 0:
                return delta

    def surface_normal(self, point):
        # return the surface normal at 'point'
        return (point - self.origin).normalize()


class Mesh:
    """
    A triangular mesh
    """
    def __init__(self, vertex, material):
        self.vertex1 = vertex[0]
        self.vertex2 = vertex[1]
        self.vertex3 = vertex[2]
        self.material = material

    def intersect(self, ray):

        EPSILON = 0.0000001
        edge1 = self.vertex2 - self.vertex1
        edge2 = self.vertex3 - self.vertex1
        h = ray.D.cross(edge2)
        a = edge1 * h
        if a > -EPSILON and a < EPSILON:
            return None
        f = 1 / a
        s = ray.O - self.vertex1
        u = f * (s * h)
        if u < .0 or u > 1:
            return None
        q = s.cross(edge1)
        v = f * (ray.D * q)
        if v < .0 or u + v > 1:
            return None
        t = f * (edge2 * q)
        if t > EPSILON:
            return t
        else:
            return None

    def surface_normal(self, point):
        v12 = self.vertex2 - self.vertex1
        v13 = self.vertex3 - self.vertex1
        N = v12.cross(v13)
        return N.normalize()

    def get_area(self):
        # used to barycentric interpolation
        v1_2 = self.vertex1 - self.vertex2
        v1_3 = self.vertex1 - self.vertex3
        return (v1_2.cross(v1_3)).norm()

    def test_pixel(self, point):
        # to compute area for barycentric tri
        v1_p = self.vertex1 - point
        v2_p = self.vertex2 - point
        v3_p = self.vertex3 - point

        z1 = v2_p.cross(v3_p)
        z2 = v3_p.cross(v1_p)
        z3 = v1_p.cross(v2_p)

        return [z1, z2, z3]


class Object:
    """
    Generic obj model class
    Work only with v (vertex) and f (face) and TRIANGULAR MESH
    """
    def __init__(self, position, filename, scalefactor, material):
        self.position = position
        self.file = filename
        self.material = material
        self.scalefactor = scalefactor

    def createStruct(self):
        # create P (points) and F (faces) array
        with open(self.file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        P = []
        F = []
        for line in content:
            if line[0] == 'v':
                tmp = np.asfarray(line[2:].split())
                point = Vec3(tmp[0], tmp[1], tmp[2])
                point *= self.scalefactor
                point += self.position
                P.append(point)
            if line[0] == 'f':
                face = np.asfarray(line[2:].split()).astype(int)
                F.append(Mesh([P[face[0]-1],
                               P[face[1]-1], P[face[2]-1]], self.material))

        return P, F


class Texture:
    def __init__(self, texture, bump=None):
        self.path = texture
        im = Image.open(self.path)
        self.width, self.height = im.size
        self.image = im.convert('RGB')
        self.bump = bump
        if self.bump:
            raw_bump = Image.open(self.bump)
            self.bump_width, self.bump_height = raw_bump.size
            self.bump_img = raw_bump.convert('RGB')


class LocalLighting:
    # standard light: ambient, diffusive, specular
    def __init__(self, ambient=.2, diffusive=.4, specular=0):
        self.ambient = float(ambient)
        self.diffusive = float(diffusive)
        self.specular = float(specular)


class Material:
    """
    Material properties
    """
    def __init__(self, color, localLighting=LocalLighting(),
                 shininess_exp=0, reflective=0, texture=None,
                 glossy_exp=None, dielectric=None):
        self.color = color                  # Rgb() class
        self.localLighting = localLighting  # LocalLighting() class
        self.shininess = shininess_exp      # numeric param
        self.reflective = reflective        # numeric param, shininess exponent
        self.texture = texture              # Texture() class
        self.glossy = glossy_exp            # glossy props, is the exponent from 1 to 100'000 - 1 extremely glossy
        self.dielectric = dielectric        # dielectric props


class Ray:
    """
    Ray of ray tracing engine
    """
    def __init__(self, origin, direction):
        self.O = origin
        self.D = direction.normalize()

    def point_at_dist(self, dist):
        return self.O + (self.D * dist)


class Vec3:
    """
    Vec3 class
    """
    def __init__(self, x=0., y=0., z=0.):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def norm(self):
        return math.sqrt(sum(num * num for num in self))

    def normalize(self):
        return self / math.sqrt(sum(num * num for num in self))

    def reflect(self, normal):
        # reflection vector around normal
        other = normal.normalize()
        return self - 2 * (self * other) * other

    def refract(self, norm, n2):
        # refraction
        # check cos between -1 e 1
        cosi = np.clip(self * norm, -1, 1)
        etai = 1  # air refraction
        etat = n2
        if cosi < 0:
            cosi = -cosi
        else:
            etai, etat = etat, etai
            norm = -norm
        eta = etai / etat   # n1 / n2
        k = 1 - eta * eta * (1 - cosi * cosi)
        if k < 0:
            return 0
        else:
            return eta * self + (eta * cosi - math.sqrt(k)) * norm

    # define vector operation
    def cross(self, other):
        if isinstance(other, Vec3):
            c = [self.y*other.z - self.z*other.y,
                 self.z*other.x - self.x*other.z,
                 self.x*other.y - self.y*other.x]
            return Vec3(c[0], c[1], c[2])
        else:
            raise ValueError("Cross product must be between two Vec3")

    def __str__(self):
        return "Vec3({}, {}, {})".format(*self)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __pow__(self, exp):
        if exp != 2:
            raise ValueError("Exponent can only be two")
        else:
            return self * self

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


class Rgb:
    """
    Class to manage Rgb color easily
    """
    def __init__(self, R=0, G=0, B=0):
        (self.R, self.G, self.B) = R, G, B

    def __add__(self, other):
        return Rgb(self.R + other.R, self.G + other.G, self.B + other.B)

    def __mul__(self, other):
        return Rgb(self.R * other, self.G * other, self.B * other)

    def __truediv__(self, other):
        return Rgb(self.R / other, self.G / other, self.B / other)

    def getRGB(self):
        if self.R > 1: self.R = 1
        if self.G > 1: self.G = 1
        if self.B > 1: self.B = 1
        return Rgb(round(self.R * 255), round(self.G * 255), round(self.B * 255))

    def __str__(self):
        return "Rgb({}, {}, {})".format(self.R, self.G, self.B)


def changeRange(OldValue, OldMax, OldMin, NewMax, NewMin):
    return (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

def texturizer(obj, n, intersection):
    # mapping texture: point to uv
    try:
        if isinstance(obj, Mesh):
            Zs = obj.test_pixel(intersection)
            area = obj.get_area()

            z1 = Zs[0]
            z2 = Zs[1]
            z3 = Zs[2]

            t1 = [0, 0]
            t2 = [1, 0]
            t3 = [0, 1]

            v1 = (z1.norm() * .5) / area
            v2 = (z2.norm() * .5) / area
            v3 = (z3.norm() * .5) / area

            Cx = t1[0] * v1 + t2[0] * v2 + t3[0] * v3
            Cy = t1[1] * v1 + t2[1] * v2 + t3[1] * v3

        else:
            # object is a sphere
            Cx = np.arcsin(n.x) / np.pi + .5
            Cy = np.arcsin(n.y) / np.pi + .5

        Ut = round((Cx * (obj.material.texture.width - 1)) + 1)
        Vt = round((Cy * (obj.material.texture.height - 1)) + 1)
        r, g, b = obj.material.texture.image.getpixel((Ut-1, Vt-1))
        return Rgb(r / 255, g / 255, b / 255)
    except:
        print('W: ',obj.material.texture.width, ' - H: ', obj.material.texture.height, ' - Ut,Vt: ', Ut,',', Vt)

def solidangleSampling(ray, exponent, samples, norm):
    # random rays sampling, ruled by exponent
    rays = []
    d = pow(ray.D * norm, exponent)
    for i in range(samples):
        x = -d / 2 + random.uniform(0,1) * d
        y = -d / 2 + random.uniform(0, 1) * d
        z = -d / 2 + random.uniform(0, 1) * d
        vec = ray.D + 2 * Vec3(x, y, z)
        # perturbed ray
        rays.append(Ray(ray.O, vec))
    return rays

def fresnel(I, N, n2):
    """
    Compute fraction of ray reflected and refracted
    """
    # cos between -1 e 1
    cosi = np.clip(I * N, -1, 1)
    etai = 1
    etat = n2
    if cosi > 0:
        etai, etat = etat, etai
    # snell law
    sint = etai / etat * math.sqrt( max(0, 1 - cosi * cosi))
    if sint >= 1:
        # total internal reflection
        Kr = 1
    else:
        cost = math.sqrt( max(0, 1 - sint * sint))
        cosi = abs(cosi)
        Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
        Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))
        Kr = (Rs * Rs + Rp * Rp) / 2
    return Kr

def ray_tracing(scene, ray, depth=0, max_depth=5):

    color = Rgb()

    if depth >= max_depth:
        return color

    intersection = scene.get_intersection(ray)
    if intersection is None:
        return color # black
    obj, dist = intersection

    inter_point = ray.point_at_dist(dist)
    surf_norm = obj.surface_normal(inter_point)

    obj_color = obj.material.color

    # texture
    if obj.material.texture and enable_texture:
        obj_color = texturizer(obj, surf_norm, inter_point)

    # ambient light
    color += obj_color * obj.material.localLighting.ambient

    for light in scene.lights:
        # diffusive component
        point_to_light_vec = (light.point - inter_point).normalize()
        point_to_light_ray = Ray(inter_point, point_to_light_vec)

        hitObj = scene.get_intersection(point_to_light_ray)

        light_in_the_middle = False
        if hitObj is not None:
            objL, distL = hitObj

            light_dist = light.intersect(point_to_light_ray)
            if light_dist is not None:
                if light_dist < distL:
                    light_in_the_middle = True

        if (hitObj is None) or light_in_the_middle :
            lambert_intensity = surf_norm * point_to_light_vec
            if lambert_intensity > 0:
                color += obj_color * obj.material.localLighting.diffusive * lambert_intensity

            # specular (reflective) light
            # Blinn-Phong
            viewDir = (scene.camera - inter_point).normalize()
            lightDir = (light.point - inter_point).normalize()
            H = (lightDir + viewDir).normalize()
            spec = pow(max(surf_norm * H, 0), obj.material.shininess)
            color += light.color * spec * obj.material.localLighting.specular

    if obj.material.reflective > 0:
        # compute reflection ray
        ray_reflected = Ray(inter_point, ray.D.reflect(surf_norm))
        color += ray_tracing(scene, ray_reflected, depth + 1, max_depth) * obj.material.reflective

    if obj.material.glossy and enable_glossy:
        # glossy
        ray_reflected = Ray(inter_point, ray.D.reflect(surf_norm))
        gloss_rgb = Rgb()
        samples = 40
        # random rays inside soldid angle
        rnd_ray = solidangleSampling(ray_reflected, obj.material.glossy, samples, surf_norm)
        for r in rnd_ray:
            gloss_rgb += ray_tracing(scene, r, depth + 1, max_depth)
        gloss_rgb /= samples
        color += gloss_rgb

    if obj.material.dielectric:
        # dielectric object
        reflectionColor = Rgb()
        refractionColor = Rgb()
        # compute fresnel
        Kr = fresnel(ray.D, surf_norm, obj.material.dielectric)
        # ray -> and n <- => ray*n < 0 (different direction) otherwise >= 0 (same direction)
        outside = False
        if ray.D * surf_norm < 0:
            outside = True
        bias = .0001 * surf_norm
        # compute refraction if it is not total internal reflection
        if (Kr < 1):
            refractionDir = ray.D.refract(surf_norm, obj.material.dielectric).normalize()
            if outside:
                refractionOrig = inter_point - bias
            else:
                refractionOrig = inter_point + bias
            ray_refracted = Ray(refractionOrig, refractionDir)
            refractionColor += ray_tracing(scene, ray_refracted, depth + 1, max_depth)
        # compute reflection ray
        ray_reflected = Ray(inter_point, ray.D.reflect(surf_norm))
        reflectionColor += ray_tracing(scene, ray_reflected, depth + 1, max_depth)
        color += reflectionColor * Kr + refractionColor * (1 - Kr)

    return color

if __name__ == "__main__":

    print('[*] - Start')
    dirname = os.path.dirname(__file__)
    time_t = time.time()

    # MATERIAL - ambient, diffuse, specular, shinness exp
    # CHROME - 0.25, 0.4, 0.774

    texture_marble = Texture(os.path.join(dirname, 'texture/Marble_125_1024.png'))
    texture_unknow = Texture(os.path.join(dirname, 'texture/1024px-Knockdown_texture.jpg'))
    texture_world = Texture(os.path.join(dirname, 'texture/world.jpg'), os.path.join(dirname, 'texture/world_bump.jpg'))

    objects = [
        # sphere chrome
        Sphere(Vec3(150, 120, -20), 80, Material(Rgb(.82, .82, .82), LocalLighting(.25, .4, .774), 77, 1)),
        # sphere texture marble
        Sphere(Vec3(150, 320, -100), 80, Material(Rgb(.82, .82, .82),
                                                 LocalLighting(.3, .4, .6), 80, .8, texture_marble)),
        # sphere glossy, diffusive material
        Sphere(Vec3(400, 340, -100), 50, Material(Rgb(0, 0, 1), LocalLighting(.08, .8, .0), 0)),
        # sphere 2 texture
        Sphere(Vec3(500, 300, -10), 50, Material(Rgb(0, 0, 1), LocalLighting(.25, .8, .0), 0, 0, texture_world)),
        # sphere dielectrinc
        # refractive index 1.3  = water, 1.5 = glass, 1.8 = diamond
        Sphere(Vec3(320, 180, -20), 60, Material(Rgb(1, 0, 0), LocalLighting(.08, 0, 0), 0, 0, None, None, 1.5)),
        # another sphere
        Sphere(Vec3(320, 260, -100), 60, Material(Rgb(.28, 1, .98), LocalLighting(.25, .8, .0), 0, .7)),
        # triangular mesh texture
        Mesh([Vec3(300, 20, -40), Vec3(400, 20, 250), Vec3(350, 120, -40)],
             Material(Rgb(0, 1, 0),LocalLighting(),0,0, texture_unknow)),
        # floor GLOSSY
        Mesh([Vec3(0,400,-1), Vec3(0,400,-500), Vec3(600,400,-1)],
             Material(Rgb(.82, .82, .82), LocalLighting(.25, .4, .774), 77, 0, None, 4)),
        Mesh([Vec3(600, 400, -1), Vec3(0, 400, -500), Vec3(600, 400, -500)],
             Material(Rgb(.82, .82, .82), LocalLighting(.25, .4, .774), 77, 0, None, 4)),
        # left wall
        Mesh([Vec3(0, 400, -1), Vec3(0, 0, -1), Vec3(0, 0, -500)], Material(Rgb(0, 1, 1))),
        Mesh([Vec3(0, 400, -1), Vec3(0, 0, -500), Vec3(0, 400, -500)], Material(Rgb(0, 1, 1))),
        # right wall
        Mesh([Vec3(600, 400, -1), Vec3(600, 0, -1), Vec3(600, 0, -500)], Material(Rgb(.13, .4, .4), LocalLighting(.3, .4, .5),.6, .8)),
        Mesh([Vec3(600, 400, -1), Vec3(600, 0, -500), Vec3(600, 400, -500)], Material(Rgb(.13, .4, .4), LocalLighting(.3, .4, .5),.6, .8)),
        # front wall
        Mesh([Vec3(0, 400, -500), Vec3(0, 0, -500), Vec3(600, 400, -500)], Material(Rgb(1, .78, 0))),
        Mesh([Vec3(600, 400, -500), Vec3(0, 0, -500), Vec3(600, 0, -500)], Material(Rgb(1, .78, 0))),
        # ceiling
        Mesh([Vec3(0, 0, -1), Vec3(0, 0, -500), Vec3(600, 0, -1)], Material(Rgb(1, 1, 1))),
        Mesh([Vec3(600, 0, -1), Vec3(0, 0, -500), Vec3(600, 0, -500)], Material(Rgb(1, 1, 1))),

    ]

    if enable_obj:
        if only_obj:
            objects = []
            V, F = Object(Vec3(200, 200, -50), 'obj/dodecahedron.obj', 60, Material(Rgb(1, 0, 0))).createStruct()
            objects.extend(F)
            V, F = Object(Vec3(300, 200, 50), 'obj/humanoid_tri.obj', 8, Material(Rgb(0, 1, 0))).createStruct()
            objects.extend(F)
        else:
            V, F = Object(Vec3(200, 200, -50), 'obj/file.obj', 60, Material(Rgb(1, 0, 0))).createStruct()
            objects.extend(F)

    lights = [
         Light(Vec3(400, 300, 10), Rgb(1, 1, 1)),
         Light(Vec3(600, 300, 100), Rgb(1, 1, 1)),
        # Light(Vec3(200, 450, 100), Rgb(1, 1, 1)),
        # Light(Vec3(100, 10, -300), Rgb(1, 1, 1)),
        # Light(Vec3(550, 10, -450), Rgb(1, 1, 1)),
        # Light(Vec3(320, 180, -300), Rgb(1, 1, 1))
    ]

    # IMAGE
    WIDTH = 800
    HEIGHT = 600

    # CAMERA
    camera = Camera(Vec3(350, 200, 400), Vec3(300, 200, 0), Vec3(0, 1, 0), math.pi / 6, WIDTH / HEIGHT)
    aspectRatio = WIDTH / HEIGHT

    for j in range(HEIGHT):
        for i in range(WIDTH):
            x = (2 * (i + 0.5) / WIDTH - 1) * aspectRatio
            y = (2 * (j + 0.5) / HEIGHT - 1)

    scene = Scene(camera.position, objects, lights )
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    # offset_px = .5
    # aa_sample = 3 # antialiasing sample

    xScreen = np.arange(0, WIDTH)
    yScreen = np.arange(0, HEIGHT)

    px_w_offset = (2 / WIDTH) * .5
    px_h_offset = (2 / HEIGHT) * .5

    rays = [
        [scene, camera.ray_cam2ray(((2 * x) / WIDTH) - 1 + px_w_offset, ((2 * y) / HEIGHT) - 1 + px_h_offset), 0, 2]
        for y in yScreen for x in xScreen]

    Rs = []
    Gs = []
    Bs = []

    # use all core available
    pool = Pool(os.cpu_count())
    results = pool.starmap(ray_tracing, rays)
    for idx, r in enumerate(results):
        color = r.getRGB()
        Rs.append(color.R)
        Gs.append(color.G)
        Bs.append(color.B)

    Rs = np.reshape(Rs, (HEIGHT, WIDTH))
    Gs = np.reshape(Gs, (HEIGHT, WIDTH))
    Bs = np.reshape(Bs, (HEIGHT, WIDTH))

    print('[OK] - Normal scene RENDERED')

    if enable_antialiasing:
        print('[*] - Start ANTIALIASING')
        Rs_aa = []
        Gs_aa = []
        Bs_aa = []

        dx = float(px_w_offset * .5)
        dy = float(px_h_offset * .5)

        for i in range(4):
            print('aa: ', i)
            R, G, B = [], [], []
            # offset: 4 point around pixel center
            if i == 0:
                aa_x = dx * -1
                aa_y = dy * 1
            elif i == 1:
                aa_x = dx * +1
                aa_y = dy * +1
            elif i == 2:
                aa_x = dx * -1
                aa_y = dy * -1
            elif i == 3:
                aa_x = dx * 1
                aa_y = dy * -1

            rays2 = [[scene, camera.ray_cam2ray(
                ((2 * x) / WIDTH) - 1 + px_w_offset + aa_x, ((2 * y) / HEIGHT) - 1 + px_h_offset + aa_y)
                         , 0, 2] for y in yScreen for x in xScreen]

            results_aa = pool.starmap(ray_tracing, rays2)

            for aa in results_aa:
                color = aa.getRGB()
                R.append(color.R)
                G.append(color.G)
                B.append(color.B)

            Rs_aa.append(np.reshape(R, (HEIGHT, WIDTH)))
            Gs_aa.append(np.reshape(G, (HEIGHT, WIDTH)))
            Bs_aa.append(np.reshape(B, (HEIGHT, WIDTH)))

        print('[OK] - AA row computation END')
        aa_R = Rs
        aa_G = Gs
        aa_B = Bs
        for i in range(4):
            aa_R += Rs_aa[i]
            aa_G += Gs_aa[i]
            aa_B += Bs_aa[i]

        # avg color of antialising image + standard image
        aa_R /= (4 + 1)
        aa_G /= (4 + 1)
        aa_B /= (4 + 1)

        Rs = aa_R
        Gs = aa_G
        Bs = aa_B

    img[..., 0] = Rs
    img[..., 1] = Gs
    img[..., 2] = Bs

    print('[OK] - END - Elapsed: ', time.time() - time_t)

    image = Image.fromarray(img)
    image.save('last_render.png')

    plt.figure(2)
    plt.imshow(img)
    plt.show()
