#ifndef POINT_TYPE_H
#define POINT_TYPE_H

#define PCL_NO_PRECOMPILE
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl_ext {

struct EIGEN_ALIGN16 PointXYZLO {
    PCL_ADD_POINT4D; // preferred way of adding a XYZ+padding
    int label; // the label of point
    int object; // the object instance of point
    PCL_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace pcl_ext

POINT_CLOUD_REGISTER_POINT_STRUCT(pcl_ext::PointXYZLO, // here we assume a XYZ + "label"+"object" (as fields)
    (float, x, x)(float, y, y)(float, z, z)(int, label, label)(int, object, object))

#endif