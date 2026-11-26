// GPD adapter for AxisForge candidates (Phase VII extension).
//
// Scores externally given parallel-jaw grasps with the classifier of GPD (ten Pas et al., IJRR 2017; official
// source https://github.com/atenpas/gpd, commit 6327f20, BSD-2-Clause), unmodified: the point cloud is
// preprocessed as GPD does (voxel grid 3 mm, surface normals from the cloud), each grasp becomes a
// gpd::candidate::Hand in GPD's frame convention (columns: approach, binormal = closing direction,
// hand axis), its 15-channel grasp image is built by GPD's own Image15ChannelsStrategy (reached through a
// subclass, because createImage is protected) with GPD's shadow computation, and the pretrained LeNet
// (models/lenet/15channels) scores it. GPD's own candidate generation is NOT used: this is the
// ranking-controlled level (same candidates as the planner, only the order changes), an adaptation of
// GPD's classifier to candidates it did not generate, not a reproduction of the GPD pipeline.
//
// usage: gpd_score_candidates LEFT.pcd RIGHT.pcd CANDIDATES.txt OUT.txt WEIGHTS_DIR IMAGE_GEOMETRY.cfg
//        LEFT_X LEFT_Y LEFT_Z RIGHT_X RIGHT_Y RIGHT_Z
// (two depth-camera clouds and their camera positions, as GPD's two-camera Cloud constructor takes them)
// CANDIDATES.txt: one grasp per line: cx cy cz ax ay az kx ky kz width (world frame, metres; a = approach
// unit vector, k = closing unit vector). OUT.txt: one score per line, same order (higher = more likely
// a good grasp, as GPD ranks).
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <pcl/kdtree/kdtree_flann.h>

#include <gpd/candidate/antipodal.h>
#include <gpd/candidate/finger_hand.h>
#include <gpd/candidate/hand.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/candidate/hand_set.h>
#include <gpd/descriptor/image_15_channels_strategy.h>
#include <gpd/descriptor/image_geometry.h>
#include <gpd/net/classifier.h>
#include <gpd/util/cloud.h>
#include <gpd/util/point_list.h>

namespace {

class AdapterStrategy : public gpd::descriptor::Image15ChannelsStrategy {
 public:
  AdapterStrategy(const gpd::descriptor::ImageGeometry &g)
      : gpd::descriptor::Image15ChannelsStrategy(g, 1, 1, false), geometry_(g) {}
  std::unique_ptr<cv::Mat> image(const gpd::util::PointList &nn, const gpd::candidate::Hand &hand,
                                 const Eigen::Matrix3Xd &shadow) const {
    auto img = std::make_unique<cv::Mat>(geometry_.size_, geometry_.size_, CV_8UC(geometry_.num_channels_),
                                         cv::Scalar(0.0));
    createImage(nn, hand, shadow, *img);
    return img;
  }

 private:
  gpd::descriptor::ImageGeometry geometry_;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc != 13) {
    std::cerr << "usage: gpd_score_candidates LEFT.pcd RIGHT.pcd CANDIDATES.txt OUT.txt WEIGHTS_DIR IMAGE_GEOMETRY.cfg "
                 "LEFT_X LEFT_Y LEFT_Z RIGHT_X RIGHT_Y RIGHT_Z\n";
    return 2;
  }
  const std::string left_file = argv[1], right_file = argv[2], cand_file = argv[3], out_file = argv[4],
                    weights = argv[5], image_cfg = argv[6];
  Eigen::Matrix3Xd views(3, 2);
  for (int i = 0; i < 2; i++)
    for (int r = 0; r < 3; r++) views(r, i) = std::stod(argv[7 + 3 * i + r]);

  // GPD's default preprocessing (cfg/eigen_params.cfg): voxel grid 3 mm, normals with a 1 cm radius.
  gpd::util::Cloud cloud(left_file, right_file, views);
  cloud.voxelizeCloud(0.003f);
  cloud.calculateNormals(1, 0.01);
  Eigen::Matrix3Xd points =
      cloud.getCloudProcessed()->getMatrixXfMap().cast<double>().block(0, 0, 3, cloud.getCloudProcessed()->points.size());
  gpd::util::PointList point_list(points, cloud.getNormals(), cloud.getCameraSource(), cloud.getViewPoints());

  gpd::descriptor::ImageGeometry image_geometry(image_cfg);
  AdapterStrategy strategy(image_geometry);
  // A hand set only to reach GPD's shadow computation (HandSet::calculateShadow); its own search is not run.
  gpd::candidate::HandGeometry hand_geometry(0.01, 0.12, 0.06, 0.02, 0.01);
  gpd::candidate::Antipodal antipodal(20.0, 6);
  Eigen::VectorXd angles(1);
  angles << 0.0;
  gpd::candidate::HandSet hand_set(hand_geometry, angles, std::vector<int>{2}, 10, true, antipodal);
  Eigen::Vector3d dims(image_geometry.depth_, image_geometry.height_ / 2.0, image_geometry.outer_diameter_);
  const double radius = dims.maxCoeff();

  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud.getCloudProcessed());

  std::ifstream in(cand_file);
  std::vector<std::unique_ptr<cv::Mat>> images;
  std::string line;
  int n_empty = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    Eigen::Vector3d c, a, k;
    double width;
    ss >> c(0) >> c(1) >> c(2) >> a(0) >> a(1) >> a(2) >> k(0) >> k(1) >> k(2) >> width;
    a.normalize();
    k = (k - a * a.dot(k)).normalized();
    Eigen::Matrix3d frame;
    frame.col(0) = a;
    frame.col(1) = k;
    frame.col(2) = a.cross(k);
    // Fingers reach half the finger depth past the grasp centre (contacts at mid-pad); lateral centre 0.
    gpd::candidate::FingerHand fh(hand_geometry.finger_width_, hand_geometry.outer_diameter_, hand_geometry.depth_, 10);
    fh.evaluateFingers(Eigen::Matrix3Xd(3, 0), hand_geometry.depth_ / 2.0);
    fh.setCenter(0.0);
    gpd::candidate::Hand hand(c, frame, fh, width);
    pcl::PointXYZRGBA s;
    s.getVector3fMap() = c.cast<float>();
    std::vector<int> idx;
    std::vector<float> d2;
    gpd::util::PointList nn;
    if (kdtree.radiusSearch(s, radius, idx, d2) > 0) {
      nn = point_list.slice(idx);
    } else {
      n_empty++;
    }
    hand_set.setSample(c);
    Eigen::Matrix3Xd shadow = nn.size() > 0 ? hand_set.calculateShadow(nn, radius) : Eigen::Matrix3Xd(3, 0);
    images.push_back(nn.size() > 0 ? strategy.image(nn, hand, shadow)
                                   : std::make_unique<cv::Mat>(image_geometry.size_, image_geometry.size_,
                                                               CV_8UC(image_geometry.num_channels_), cv::Scalar(0.0)));
  }
  std::shared_ptr<gpd::net::Classifier> classifier =
      gpd::net::Classifier::create("", weights, gpd::net::Classifier::Device::eCPU, 1);
  std::vector<float> scores = classifier->classifyImages(images);
  std::ofstream out(out_file);
  for (float v : scores) out << v << "\n";
  std::cerr << "scored " << scores.size() << " candidates (" << n_empty << " with no cloud points nearby) from "
            << point_list.size() << " points\n";
  return 0;
}
