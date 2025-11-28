#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libc.stdint cimport int32_t, int64_t

import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "accessibility.h" namespace "MTC::accessibility":
    cdef cppclass Accessibility:
        Accessibility(int64_t, vector[vector[int64_t]], vector[vector[double]], bool) except +
        vector[string] aggregations
        vector[string] decays
        void initializeCategory(double, int64_t, string, vector[int64_t])
        pair[vector[vector[double]], vector[vector[int64_t]]] findAllNearestPOIs(
            float, int64_t, string, int64_t)
        void initializeAccVar(string, vector[int64_t], vector[double])
        vector[double] getAllAggregateAccessibilityVariables(
            float, string, string, string, int64_t)
        vector[int64_t] Route(int64_t, int64_t, int64_t)
        vector[vector[int64_t]] Routes(vector[int64_t], vector[int64_t], int64_t)
        double Distance(int64_t, int64_t, int64_t)
        vector[double] Distances(vector[int64_t], vector[int64_t], int64_t)
        vector[vector[pair[int64_t, float]]] Range(vector[int64_t], float, int64_t, vector[int64_t])
        void precomputeRangeQueries(double)


cdef np.ndarray[double] convert_vector_to_array_dbl(vector[double] vec):
    cdef Py_ssize_t n = vec.size()
    cdef np.ndarray[double] arr = np.empty(n, dtype=np.double)
    cdef Py_ssize_t i
    for i in range(n):
        arr[i] = vec[i]
    return arr


cdef np.ndarray[double, ndim=2] convert_2D_vector_to_array_dbl(vector[vector[double]] vec):
    cdef Py_ssize_t rows = vec.size()
    cdef Py_ssize_t cols = 0
    if rows > 0:
        cols = vec[0].size()

    cdef np.ndarray[double, ndim=2] arr = np.empty((rows, cols), dtype=np.double)

    cdef Py_ssize_t i, j
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = vec[i][j]

    return arr


cdef np.ndarray[int64_t, ndim=2] convert_2D_vector_to_array_int(vector[vector[int64_t]] vec):
    cdef Py_ssize_t rows = vec.size()
    cdef Py_ssize_t cols = 0
    if rows > 0:
        cols = vec[0].size()

    cdef np.ndarray[int64_t, ndim=2] arr = np.empty((rows, cols), dtype=np.int64)

    cdef Py_ssize_t i, j
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = vec[i][j]

    return arr


cdef class cyaccess:
    cdef Accessibility *access

    def __cinit__(
        self,
        np.ndarray[int64_t] node_ids,
        np.ndarray[double, ndim=2] node_xys,
        np.ndarray[int64_t, ndim=2] edges,
        np.ndarray[double, ndim=2] edge_weights,
        bool twoway=True
    ):
        """
        Construct the Accessibility network. node_ids and node_xys are unused,
        but retained for compatibility.
        """
        self.access = new Accessibility(len(node_ids), edges, edge_weights, twoway)

    def __dealloc__(self):
        del self.access

    def initialize_category(
        self,
        double maxdist,
        int maxitems,
        string category,
        np.ndarray[int64_t] node_ids
    ):
        cdef vector[int64_t] v_ids
        v_ids.reserve(node_ids.shape[0])

        cdef Py_ssize_t i
        for i in range(node_ids.shape[0]):
            v_ids.push_back(<int64_t> node_ids[i])

        self.access.initializeCategory(maxdist, maxitems, category, v_ids)

    def find_all_nearest_pois(
        self,
        double radius,
        int64_t num_of_pois,
        string category,
        int64_t impno=0
    ):
        ret = self.access.findAllNearestPOIs(radius, num_of_pois, category, impno)

        out_dists = convert_2D_vector_to_array_dbl(ret.first)
        out_ids   = convert_2D_vector_to_array_int(ret.second)

        return out_dists, out_ids

    def initialize_access_var(
        self,
        string category,
        np.ndarray[int64_t] node_ids,
        np.ndarray[double] values
    ):
        cdef vector[int64_t] v_ids
        v_ids.reserve(node_ids.shape[0])

        cdef Py_ssize_t i
        for i in range(node_ids.shape[0]):
            v_ids.push_back(<int64_t> node_ids[i])

        cdef vector[double] v_vals
        v_vals.reserve(values.shape[0])
        for i in range(values.shape[0]):
            v_vals.push_back(<double> values[i])

        self.access.initializeAccVar(category, v_ids, v_vals)

    def get_available_aggregations(self):
        return self.access.aggregations

    def get_available_decays(self):
        return self.access.decays

    def get_all_aggregate_accessibility_variables(
        self,
        double radius,
        category,
        aggtyp,
        decay,
        int64_t impno=0,
    ):
        ret = self.access.getAllAggregateAccessibilityVariables(
            radius, category, aggtyp, decay, impno)

        return convert_vector_to_array_dbl(ret)

    def shortest_path(self, int64_t srcnode, int64_t destnode, int64_t impno=0):
        return self.access.Route(srcnode, destnode, impno)

    def shortest_paths(
        self,
        np.ndarray[int64_t] srcnodes,
        np.ndarray[int64_t] destnodes,
        int64_t impno=0
    ):
        cdef vector[int64_t] v_src
        v_src.reserve(srcnodes.shape[0])
        cdef Py_ssize_t i
        for i in range(srcnodes.shape[0]):
            v_src.push_back(<int64_t> srcnodes[i])

        cdef vector[int64_t] v_dest
        v_dest.reserve(destnodes.shape[0])
        for i in range(destnodes.shape[0]):
            v_dest.push_back(<int64_t> destnodes[i])

        return self.access.Routes(v_src, v_dest, impno)

    def shortest_path_distance(self, int64_t srcnode, int64_t destnode, int64_t impno=0):
        return self.access.Distance(srcnode, destnode, impno)

    def shortest_path_distances(
        self,
        np.ndarray[int64_t] srcnodes,
        np.ndarray[int64_t] destnodes,
        int64_t impno=0
    ):
        cdef vector[int64_t] v_src
        v_src.reserve(srcnodes.shape[0])
        cdef Py_ssize_t i

        for i in range(srcnodes.shape[0]):
            v_src.push_back(<int64_t> srcnodes[i])

        cdef vector[int64_t] v_dest
        v_dest.reserve(destnodes.shape[0])
        for i in range(destnodes.shape[0]):
            v_dest.push_back(<int64_t> destnodes[i])

        return self.access.Distances(v_src, v_dest, impno)

    def precompute_range(self, double radius):
        self.access.precomputeRangeQueries(radius)

    def nodes_in_range(
        self,
        vector[int64_t] srcnodes,
        float radius,
        int64_t impno,
        np.ndarray[int64_t] ext_ids
    ):
        cdef vector[int64_t] v_ext
        v_ext.reserve(ext_ids.shape[0])

        cdef Py_ssize_t i
        for i in range(ext_ids.shape[0]):
            v_ext.push_back(<int64_t> ext_ids[i])

        return self.access.Range(srcnodes, radius, impno, v_ext)
