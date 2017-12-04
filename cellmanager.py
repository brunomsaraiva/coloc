"""Placeholder"""

import numpy as np
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage

class CellManager(object):
    """Placeholder"""

    def __init__(self):
        self.features = None
        self.labels = None
        self.cells = []

    @staticmethod
    def compute_distance_peaks(mask):
        """Method used when the selected algorithm for the feature computation
        is the Distance Peaks.
        Returns a list of the centers of the different identified regions,
        which should be used in the compute_features method"""

        distance = ndimage.morphology.distance_transform_edt(1-mask)

        mindist = 5
        minmargin = 10

        centers = peak_local_max(distance, min_distance=mindist,
                                 threshold_abs=5,
                                 exclude_border=True,
                                 num_peaks=10000,
                                 indices=True)

        placedmask = np.ones(distance.shape)
        lx, ly = distance.shape
        result = []
        rad = 5
        heights = []
        circles = []

        for c in centers:
            x, y = c

            if x >= minmargin and y >= minmargin and x <= lx - minmargin \
               and y <= ly - minmargin and placedmask[x, y]:
                placedmask[x - mindist:x + mindist +
                           1, y - mindist:y + mindist + 1] = 0
                s = distance[x, y]
                circles.append((x, y, rad, s))
                heights.append(s)

        ixs = np.argsort(heights)
        for ix in ixs:
            result.append(circles[ix])

        return result

    def compute_features(self, cells_mask):
        """Method used to compute the features of an image using the mask.
        requires a mask and an instance of the imageprocessingparams
        if the selected algorithm used is Distance Peak, used the method
        compute_distance_peaks to compute the features"""

        mask = 1 - cells_mask
        features = np.zeros(mask.shape)

        circles = self.compute_distance_peaks(mask)

        for ix, c in enumerate(circles):
            x, y, dum1, dum2 = c
            for f in range(3):
                features[x - 1 + f, y] = ix + 1
                features[x, y - 1 + f] = ix + 1

        self.features = features

    def compute_labels(self, cells_mask):
        """Computes the labels for each region based on the previous computed
        features. Requires the mask, th base mask, the features and an
        instance of the imageprocessingparams"""

        markers = self.features
        mask = 1- cells_mask
        inverted_mask = cells_mask

        distance = - ndimage.morphology.distance_transform_edt(inverted_mask)

        mindist = np.min(distance)
        markpoints = markers > 0
        distance[markpoints] = mindist
        labels = morphology.watershed(distance, markers, mask=inverted_mask)

        self.labels = labels

    def cell_regions_from_labels(self, labels, maskshape):
        """creates a list of N cells assuming self.labels has consecutive
        values from 1 to N create cell regions, frontiers and neighbours from
        labeled regions presumes that cell list is created and has enough
        elements for all different labels. Each cell is at index label-1
        """

        difLabels = []
        for line in labels:
            difLabels.extend(set(line))
        difLabels = sorted(set(difLabels))[1:]

        cells = {}

        for f in difLabels:
            cells[str(int(f))] = Cell(f)

        for y in range(1, len(labels[0, :]) - 1):
            old_label = 0
            x1 = -1
            x2 = -1

            for x in range(1, len(labels[:, 0]) - 1):
                l = int(labels[x, y])

                # check if line began or ended, add line
                if l != old_label:
                    if x1 > 0:
                        x2 = x - 1
                        cells[str(old_label)].add_line(y, x1, x2)
                        x1 = -1
                    if l > 0:
                        x1 = x
                    old_label = l

                # check neighbours
                if l > 0:
                    square = labels[x - 1:x + 2, y - 1:y + 2]
                    cells[str(l)].add_frontier_point(x, y, square)

        for key in cells.keys():
            cells[key].compute_box(maskshape)
            cells[key].compute_cell_mask(maskshape)

        self.cells = cells

    def process_cells(self, cells_mask):
        """Placeholder"""

        self.compute_features(cells_mask)
        self.compute_labels(cells_mask)
        self.cell_regions_from_labels(self.labels, cells_mask.shape)

        print len(self.cells.keys())

class Cell(object):
    """Placeholder"""

    def __init__(self, cell_id):
        self.label = cell_id
        self.lines = []
        self.outline = []
        self.long_axis = []
        self.short_axis = []
        self.length = 0
        self.width = 0
        self.eccentricity = 0
        self.irregularity = 0
        self.box = None
        self.cell_mask = None
        self.cell_image = None
        self.cell_image_overlays = None
        self.selection_state = 0

    def add_line(self, y, x1, x2):
        """
        Adds a line to the cell region and updates area
        """

        self.lines.append((y, x1, x2))

    def add_frontier_point(self, x, y, neighs):
        """
        Adds an external point. Neighs is the neighbourhood labels
        """
        # check if any neighbour not in labels
        # nlabels=np.unique(neighs[neighs <> self.label])

        nlabels = []
        notzero = []
        for line in neighs:
            for p in line:
                if p != self.label and not p in nlabels:
                    nlabels.append(p)

        if nlabels != []:
            self.outline.append((x, y))

    def compute_box(self, maskshape):
        """ computes the box
        """

        points = np.asarray(self.outline)  # in two columns, x, y
        bm = 30
        w, h = maskshape
        self.box = (max(min(points[:, 0]) - bm, 0),
                    max(min(points[:, 1]) - bm, 0),
                    min(max(points[:, 0]) + bm, w - 1),
                    min(max(points[:, 1]) + bm, h - 1))

    def compute_cell_mask(self, maskshape):
        x0, y0, x1, y1 = self.box
        mask = np.zeros((x1 - x0 + 1, y1 - y0 + 1))
        for lin in self.lines:
            y, st, en = lin
            mask[st - x0:en - x0 + 1, y - y0] = 1.0
        self.cell_mask = mask

        rotations = self.rotation_matrices(5)

        self.compute_axes(rotations, maskshape)

    @staticmethod
    def rotation_matrices(step):
        """ returns a list of rotation matrixes over 180 deg
        matrixes are transposed to use with 2 column point arrays (x,y),
        multiplying after the array
        TODO: optimize with np vectors
        """

        result = []
        ang = 0

        while ang < 180:
            sa = np.sin(ang / 180.0 * np.pi)
            ca = np.cos(ang / 180.0 * np.pi)
            # note .T, for column points
            result.append(np.matrix([[ca, -sa], [sa, ca]]).T)
            ang = ang + step

        return result

    @staticmethod
    def bounded_point(x0, x1, y0, y1, p):
        tx, ty = p

        if tx < x0:
            tx = x0
        elif tx > x1:
            tx = x1

        if ty < y0:
            ty = y0
        elif ty > y1:
            ty = y1

        return tx, ty

    @staticmethod
    def bound_rectangle(points):
        """ returns a tuple (x0,y0,x1,y1,width) of the bounding rectangle
        points must be a N,2 array of x,y coords
        """

        x0, y0 = np.amin(points, axis=0)
        x1, y1 = np.amax(points, axis=0)
        a = np.min([(x1 - x0), (y1 - y0)])
        return x0, y0, x1, y1, a

    def axes_from_rotation(self, x0, y0, x1, y1, rotation):
        """ sets the cell axes from the box and the rotation
        """

        # midpoints
        mx = (x1 + x0) / 2
        my = (y1 + y0) / 2

        # assumes long is X. This duplicates rotations but simplifies
        # using different algorithms such as brightness
        self.long_axis = [[x0, my], [x1, my]]
        self.short_axis = [[mx, y0], [mx, y1]]
        self.short_axis = \
            np.asarray(np.dot(self.short_axis, rotation.T), dtype=np.int32)
        self.long_axis = \
            np.asarray(np.dot(self.long_axis, rotation.T), dtype=np.int32)

        # check if axis fall outside area due to rounding errors
        bx0, by0, bx1, by1 = self.box
        self.short_axis[0] = \
            self.bounded_point(bx0, bx1, by0, by1, self.short_axis[0])
        self.short_axis[1] = \
            self.bounded_point(bx0, bx1, by0, by1, self.short_axis[1])
        self.long_axis[0] = \
            self.bounded_point(bx0, bx1, by0, by1, self.long_axis[0])
        self.long_axis[1] = \
            self.bounded_point(bx0, bx1, by0, by1, self.long_axis[1])

        self.length = \
            np.linalg.norm(self.long_axis[1] - self.long_axis[0])
        self.width = \
            np.linalg.norm(self.short_axis[1] - self.short_axis[0])

    def compute_axes(self, rotations, maskshape):
        """ scans rotation matrices for the narrowest rectangle
        stores the result in self.long_axis and self.short_axis, each a 2,2 array
        with one point per line (coords axes in columns)

        also computes the box for masks and images
        WARNING: Rotations cannot be empty and must include a null rotation
        """

        self.compute_box(maskshape)
        points = np.asarray(self.outline)  # in two columns, x, y
        width = len(points) + 1

        # no need to do more rotations, due to symmetry
        for rix in range(len(rotations) / 2 + 1):
            r = rotations[rix]
            nx0, ny0, nx1, ny1, nwidth = self.bound_rectangle(
                np.asarray(np.dot(points, r)))

            if nwidth < width:
                width = nwidth
                x0 = nx0
                x1 = nx1
                y0 = ny0
                y1 = ny1
                angle = rix

        self.axes_from_rotation(x0, y0, x1, y1, rotations[angle])

        if self.length < self.width:
            dum = self.length
            self.length = self.width
            self.width = dum
            dum = self.short_axis
            self.short_axis = self.long_axis
            self.long_axis = dum

        self.eccentricity = \
            ((self.length - self.width) / (self.length + self.width))
        self.irregularity = \
            (len(self.outline) / (np.sum(self.cell_mask) ** 0.5))