# Author: Grant Trebbin
from PIL import Image
import PIL.ImageOps
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, LineString, MultiPoint
import math
from random import random
from PseudoHilbert.PseudoHilbert import PseudoHilbert
import svgwrite


# restrict a number so that it is between two other numbers
def clamp(n, smallest, largest):
    if n < smallest:
        return smallest
    if n > largest:
        return largest
    return n


# Round a number up to the nearest half
def round_up_to_half(n):
    double_x = 2 * n
    round_up_x = math.ceil(double_x) // 2 * 2 + 1
    return round_up_x / 2


# Round a number down to the nearest half
def round_down_to_half(n):
    double_x = 2 * n
    round_up_x = (math.floor(double_x) - 1) // 2 * 2 + 1
    return round_up_x / 2


# Handle the management of the SVG diagram
class Diagram:
    def __init__(self, diagram_width, diagram_height):
        self.width = diagram_width
        self.height = diagram_height
        self.dwg = svgwrite.Drawing(profile='full',
                                    size=(str(self.width) + 'px',
                                          str(self.height) + 'px'),
                                    viewBox=('0 0 ' + str(self.width) +
                                             ' ' + str(self.height)))
        self.dwg.add(self.dwg.rect(insert=(0, 0),
                                   size=(width, height),
                                   fill='rgb(255,255,255)'))


class StippleConverger:
    def __init__(self, image, stipple_points):

        self.points = stipple_points

        # Precomputed horizontal integrals
        # Based on:
        # Weighted Voronoi stippling - Adrian Secord
        # 2nd International Symposium on Non-Photorealistic
        # Animation and Rendering (NPAR 2002)
        self.image_array = np.asarray(image, dtype="int32")
        cumulative_sum = np.cumsum(self.image_array, axis=1, dtype="int32")
        self.height, self.width = self.image_array.shape

        self.P_array = np.zeros((self.height, self.width + 1), dtype="int32")
        self.P_array[:, 1:] = cumulative_sum
        self.Q_array = np.cumsum(self.P_array, axis=1, dtype="int32")

        # Create a polygon the size of the original image to clip all the
        # Voronoi regions
        mask = [(0, 0),
                (0, self.height - 1),
                (self.width - 1, self.height - 1),
                (self.width - 1, 0)]
        self.mask_polygon = Polygon(mask)

        # Create exterior boundary points to stop regions in the area of
        # interest going to infinity.  The boundary points define the corners
        # of a rectangle centered on the original and image 7 times the size
        # of the input image.  That's not based on anything.  It was a
        # multiple big enough to be sure problems wouldn't occur.  3 times as
        # large should suffice.

        self.exterior_boundary_points =\
            np.array([[-3 * self.width, -3 * self.height],
                     [-3 * self.width, 4 * self.height],
                     [4 * self.width, 4 * self.height],
                     [4 * self.width, -3 * self.height]])

        self.voronoi_polygons = []
        self.voronoi_points = []

    def iterate(self):
        self.voronoi_points.clear()
        self.voronoi_polygons.clear()
        bound_points = np.append(self.points,
                                 self.exterior_boundary_points, axis=0)
        voronoi_result = Voronoi(bound_points)

        for point_index, point in enumerate(voronoi_result.points):
            # `point_index` is the index used to refer to one of the input
            # points `point` contains the vertex_coordinates of that point

            # `region_index` is that region that `point` in located in
            region_index = voronoi_result.point_region[point_index]

            # `region` is a list of the indices of the vertices of a
            # region identified by `region_index`.  -1 in this list indicates
            # a vertex at infinity
            region = (voronoi_result.regions[region_index])

            # `coord` is the x, y vertex_coordinates of `point`
            point = voronoi_result.points[point_index]

            if all(i >= 0 for i in region):
                if len(region) >= 3:
                    # Create and mask each polygon
                    vertex_coordinates = voronoi_result.vertices[region]
                    region_polygon =\
                        Polygon([tuple(l) for l in vertex_coordinates])
                    masked_polygon = self.mask_polygon.intersection(
                        region_polygon)
                    bounds = masked_polygon.bounds

                    denominator_sum = np.int64(0)
                    x_coord_sum = np.int64(0)
                    y_coord_sum = np.int64(0)
                    lower_y_bound = int(math.floor(round_up_to_half(bounds[1])))
                    upper_y_bound = int(math.floor(
                        round_down_to_half(bounds[3])))

                    # Calculate the weighted Centroid for the region
                    # The process to efficiently calculate the centroid is
                    # explained here
                    # http://www.grant-trebbin.com/2017/04/efficient-centroid-calculation-for.html
                    for y_value in range(lower_y_bound,
                                         upper_y_bound + 1):

                        retval = polygon_intersect_y(masked_polygon,
                                                     y_value + 0.5)
                        if len(retval) == 2:
                            retval.sort(key=lambda tup: tup[0])
                            x_min = int(
                                math.floor(round_up_to_half(retval[0][0])))
                            x_max = int(
                                math.floor(round_down_to_half(retval[1][0])))
                            den = (
                                self.P_array[y_value][x_max+1] -
                                self.P_array[y_value][x_min])

                            yint = (self.P_array[y_value][x_max+1] -
                                    self.P_array[y_value][x_min]) *\
                                   (y_value + 0.5)

                            xint = (
                                ((x_max+1+0.5) *
                                 self.P_array[y_value][x_max+1] - (x_min+0.5) *
                                 self.P_array[y_value][x_min]) -
                                (self.Q_array[y_value][x_max+1] -
                                 self.Q_array[y_value][x_min]))

                            denominator_sum += den
                            y_coord_sum += yint
                            x_coord_sum += xint

                    if denominator_sum != 0:
                        new_x = x_coord_sum / denominator_sum
                        new_y = y_coord_sum / denominator_sum
                    else:
                        new_x = point[0]
                        new_y = point[1]

                    self.voronoi_polygons.append(
                        masked_polygon.exterior.coords.xy)
                    self.voronoi_points.append([new_x, new_y])
            self.points = np.array(self.voronoi_points)


# Load an image file
def load_image(filename):
    img = Image.open(filename)
    img.load()
    return img


# https://stackoverflow.com/questions/32275933/python-return-y-coordinates-of-polygon-path-given-x
# Given a polygon and the coordinate for a horizontal line, find the
# intersection points
def polygon_intersect_y(poly, y_val):
    if isinstance(poly, Polygon):
        poly = poly.boundary

    horizontal_line = LineString([[poly.bounds[0]-1, y_val],
                                  [poly.bounds[2]+1, y_val]])
    intersection_points = poly.intersection(horizontal_line)

    intersection_coordinates = []

    if isinstance(intersection_points, MultiPoint):
        for intersection_point in intersection_points:
            point_coordinates_list = list(intersection_point.coords)[0]
            intersection_coordinates.append(point_coordinates_list)

    return intersection_coordinates


# https://gist.github.com/pv/8036995
image_filename = "graceKelly.jpg"
number_of_points = 200
dot_radius = 5
non_dithering_iterations = 5
dithering_iterations = 5

# Makes white areas whiter, dark areas darker. One is neutral
adjustment_parameter = 3

# Load an image and convert it to grayscale
input_image = load_image(image_filename)
grey_input = PIL.ImageOps.grayscale(input_image)

# Get the negative of the image
invert_image = PIL.ImageOps.invert(grey_input)

# Precomputed horizontal integrals
# Based on:
# Weighted Voronoi stippling - Adrian Secord
# 2nd International Symposium on Non-Photorealistic
# Animation and Rendering (NPAR 2002)
image_array_float = np.power(np.asarray(invert_image, dtype="int32"),
                             adjustment_parameter)
image_array_float = np.divide(image_array_float, np.power(255,
                              adjustment_parameter) / 255)
image_array = image_array_float.astype('uint16')


# Generate a PseudoHilbert Curve that spans the image and add the source image
# to the output.
height, width = image_array.shape
PsH = PseudoHilbert(width, height)
diagram = Diagram(width, height)
background_image = diagram.dwg.add(diagram.dwg.image(image_filename,
                                   insert=(0, 0),
                                   size=(width, height),
                                   image_rendering='optimizeSpeed'))

# Get the Coordinates of the Pseudo Hilbert Path (x,y)
pseudo_hilbert_order = np.array(PsH.index_to_coordinate)

# Mirror the curve vertically to make the coordinate system consistent with
# the image.
pseudo_hilbert_reflected = pseudo_hilbert_order.copy()
pseudo_hilbert_reflected[:, 1] = (height-1) - pseudo_hilbert_reflected[:, 1]

# Swap the coordinates of each point as they are indexed by row and then
# column when looking up the image array.  This means y followed by x.
pseudo_hilbert_reflected_swapped = pseudo_hilbert_reflected[:, [1, 0]]

# Move the curve to the centre of the pixels and convert to a Python list.
pseudo_hilbert_plot_ready = np.add(pseudo_hilbert_reflected, 0.5).tolist()

# Rearrange the image into a linear order by following the Pseudo Hilbert Curve.
linearised_image = image_array[tuple(pseudo_hilbert_reflected_swapped.T)]

# Generate initial stippling points
image_sum = np.sum(linearised_image)
amount_per_point = image_sum / number_of_points
rolling_modulus = np.mod((amount_per_point / 2 + np.cumsum(linearised_image)),
                         amount_per_point)
difference = np.diff(rolling_modulus)
index_of_points = np.where(difference < 0)[0] + 1
point_coordinates = pseudo_hilbert_reflected_swapped[index_of_points]

# Print properties
point_area = len(point_coordinates) * math.pi * dot_radius * dot_radius
print("Selected number of points", number_of_points)
print("Point Radius", dot_radius)
print("Number of points generated", len(point_coordinates))
print("Total Point Area", point_area)
print("Total sum of all pixels in image ", image_sum)
print("Sum of pixels divided by number of points", amount_per_point)

# Move points to centre of pixels and create the StipleConverger
new_point_coordinates = point_coordinates[:, [1, 0]] + 0.5
image_stippler = StippleConverger(image_array, new_point_coordinates)

# Add the current location of the points to the coordinate history
coordinate_history = []
coordinate_history.append(image_stippler.points)
dither_amount = 0.3

# Iterate the points and dither them after each step.  This helps with
# dispersion of the points
for _ in range(non_dithering_iterations):
    image_stippler.iterate()
    for point_to_dither in image_stippler.points:
        x = point_to_dither[0]
        y = point_to_dither[1]
        new_x = x + dither_amount * random()
        new_y = y + dither_amount * random()
        point_to_dither[0] = clamp(new_x, 0, width)
        point_to_dither[1] = clamp(new_y, 0, height)
    coordinate_history.append(image_stippler.points)

# Iterate the points without the dithering step to converge to a final location
for _ in range(dithering_iterations):
    image_stippler.iterate()
    coordinate_history.append(image_stippler.points)

# reshape the coordinate history so that a list of paths for each point is
# created.
coordinate_history_list = [b.tolist() for b in coordinate_history]
coordinate_paths = list(zip(*coordinate_history_list))
coordinate_paths = [[[point[0] + 0.0, point[1] + 0.0]for point in path] for
                    path in coordinate_paths]

# Create the dots for the SVG image
dots = []
for _ in range(len(coordinate_paths)):
    back_circle = diagram.dwg.add(diagram.dwg.circle(r=dot_radius,
                                                     fill="rgb(" +
                                                          str(0) + "," +
                                                          str(0) + "," +
                                                          str(0) + ")",
                                                     stroke='none'))
    dots.append(back_circle)

# position the dots
for path in range(len(coordinate_paths)):
    dots[path]['cx'] = coordinate_paths[path][-1][0]
    dots[path]['cy'] = coordinate_paths[path][-1][1]

# save the output with the background
diagram.dwg.filename = 'stippled_with_image.svg'
diagram.dwg.save()

# save the output without the background
background_image['display'] = 'none'
diagram.dwg.filename = 'stippled.svg'
diagram.dwg.save()