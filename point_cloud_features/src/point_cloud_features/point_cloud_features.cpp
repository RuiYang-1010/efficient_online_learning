/**
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, Zhi Yan
 * All rights reserved.
 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **/

#include "point_cloud_features/point_cloud_features.h"

/* f1 (1d): Number of points included in a cluster */
int numberOfPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr pc) {
  return pc->size();
}

/* f2 (1d): The minimum distance of the cluster to the sensor */
/* f1 and f2 could be used in pairs, since f1 varies with f2 changes */
float minDistance(pcl::PointCloud<pcl::PointXYZI>::Ptr pc) {
  float m = FLT_MAX;
  
  for(int i = 0; i < pc->size(); i++) {
    m = std::min(m, pc->points[i].x*pc->points[i].x + pc->points[i].y*pc->points[i].y + pc->points[i].z*pc->points[i].z);
  }
  
  return sqrt(m);
}

/* f3 (6d): 3D covariance matrix of the cluster */
void covarianceMat3D(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, std::vector<float> &res) {
  Eigen::Matrix3f covariance_3d;
  pcl::PCA<pcl::PointXYZI> pca;
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_projected(new pcl::PointCloud<pcl::PointXYZI>);
  Eigen::Vector4f centroid;
  
  pca.setInputCloud(pc);
  pca.project(*pc, *pc_projected);
  pcl::compute3DCentroid(*pc, centroid);
  pcl::computeCovarianceMatrixNormalized(*pc_projected, centroid, covariance_3d);

  // Only 6 elements are needed as covariance_3d is symmetric.
  res.push_back(covariance_3d(0,0));
  res.push_back(covariance_3d(0,1));
  res.push_back(covariance_3d(0,2));
  res.push_back(covariance_3d(1,1));
  res.push_back(covariance_3d(1,2));
  res.push_back(covariance_3d(2,2));
}

/* f4 (6d): The normalized moment of inertia tensor */
void normalizedMOIT(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, std::vector<float> &res) {
  Eigen::Matrix3f moment_3d;
  pcl::PCA<pcl::PointXYZI> pca;
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_projected(new pcl::PointCloud<pcl::PointXYZI>);
  
  moment_3d.setZero();
  pca.setInputCloud(pc);
  pca.project(*pc, *pc_projected);
  for(int i = 0; i < (*pc_projected).size(); i++) {
    moment_3d(0,0) += (*pc_projected)[i].y*(*pc_projected)[i].y + (*pc_projected)[i].z*(*pc_projected)[i].z;
    moment_3d(0,1) -= (*pc_projected)[i].x*(*pc_projected)[i].y;
    moment_3d(0,2) -= (*pc_projected)[i].x*(*pc_projected)[i].z;
    moment_3d(1,1) += (*pc_projected)[i].x*(*pc_projected)[i].x + (*pc_projected)[i].z*(*pc_projected)[i].z;
    moment_3d(1,2) -= (*pc_projected)[i].y*(*pc_projected)[i].z;
    moment_3d(2,2) += (*pc_projected)[i].x*(*pc_projected)[i].x + (*pc_projected)[i].y*(*pc_projected)[i].y;
  }
  
  // Only 6 elements are needed as moment_3d is symmetric.
  res.push_back(moment_3d(0,0));
  res.push_back(moment_3d(0,1));
  res.push_back(moment_3d(0,2));
  res.push_back(moment_3d(1,1));
  res.push_back(moment_3d(1,2));
  res.push_back(moment_3d(2,2));
}

/* f5 (n*2d): Slice feature for the cluster */
void sliceFeature(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, int n, std::vector<float> &res) {
  for(int i = 0; i < n*2; i++) {
    res.push_back(0);
  }

  Eigen::Vector4f pc_min, pc_max;
  pcl::getMinMax3D(*pc, pc_min, pc_max);
  
  pcl::PointCloud<pcl::PointXYZI>::Ptr blocks[n];
  float itv = (pc_max[2] - pc_min[2]) / n;
  
  if(itv > 0) {
    for(int i = 0; i < n; i++) {
      blocks[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
    }
    for(unsigned int i = 0, j; i < pc->size(); i++) {
      j = std::min((n-1), (int)((pc->points[i].z - pc_min[2]) / itv));
      blocks[j]->points.push_back(pc->points[i]);
    }
    
    Eigen::Vector4f block_min, block_max;
    for(int i = 0; i < n; i++) {
      if(blocks[i]->size() > 2) { // At least 3 points to perform pca.
	pcl::PCA<pcl::PointXYZI> pca;
	pcl::PointCloud<pcl::PointXYZI>::Ptr block_projected(new pcl::PointCloud<pcl::PointXYZI>);
	pca.setInputCloud(blocks[i]);
	pca.project(*blocks[i], *block_projected);
	pcl::getMinMax3D(*block_projected, block_min, block_max);
      } else {
	block_min.setZero();
	block_max.setZero();
      }
      res[i*2] = block_max[0] - block_min[0];
      res[i*2+1] = block_max[1] - block_min[1];
    }
  }
}

/* f6 (n+2d): Distribution of the reflection intensity, including the mean, the standard deviation and the normalized 1D histogram (n is the number of bins) */
void intensityDistribution(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, int n, std::vector<float> &res) {
  float sum = 0, min = FLT_MAX, max = -FLT_MAX, mean = 0, sum_dev = 0;
  
  for(int i = 0; i < n+2; i++) {
    res.push_back(0);
  }
  
  for(int i = 0; i < pc->size(); i++) {
    sum += pc->points[i].intensity;
    min = std::min(min, pc->points[i].intensity);
    max = std::max(max, pc->points[i].intensity);
  }
  mean = sum / pc->size();
  
  for(int i = 0; i < pc->size(); i++) {
    sum_dev += (pc->points[i].intensity - mean) * (pc->points[i].intensity - mean);
    
    int j = std::min(float(n-1), std::floor((pc->points[i].intensity-min) / ((max-min) / n)));
    res[j]++;
  }
  
  res[n] = sqrt(sum_dev / pc->size());
  res[n+1] = mean;
}
