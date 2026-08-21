#include "utils.h"

FlowSet::FlowSet(const std::string& root, bool occ) {
    std::string image_dir = root + "/training/image_2";
    std::string flow_dir  = root + "/training/" + (occ ? "flow_occ" : "flow_noc");

    // Collect paths first, keeping the flow_occ path so ground truth stays
    // reachable for the accuracy evaluation.
    std::vector<std::tuple<std::string, std::string, std::string>> paths;
    for (auto& entry : fs::directory_iterator(flow_dir)) {
        std::string flow_path = entry.path().string();
        if (flow_path.find("_10.png") == std::string::npos) continue;

        std::string filename = entry.path().filename().string();
        std::string img1_path = image_dir + "/" + filename;
        std::string img2_path = image_dir + "/" +
                                filename.substr(0, filename.size() - 7) + "_11.png";

        if (fs::exists(img1_path) && fs::exists(img2_path)) {
            paths.emplace_back(img1_path, img2_path, flow_path);
        }
    }

    // Preload the image pairs; flow maps are decoded on demand.
    img1s.reserve(paths.size());
    img2s.reserve(paths.size());
    flow_paths.reserve(paths.size());

    for (const auto& [p1, p2, flow] : paths) {
        img1s.push_back(cv::imread(p1));
        img2s.push_back(cv::imread(p2));
        flow_paths.push_back(flow);
    }
}

cv::Mat FlowSet::groundTruthFlow(size_t index) const {
    // KITTI stores flow as a 16-bit 3-channel PNG. OpenCV reads it BGR, so the
    // channels are (valid, v_raw, u_raw); the raw components are unsigned with a
    // 2^15 bias at 1/64 px resolution, and valid is nonzero where flow exists.
    cv::Mat encoded = cv::imread(flow_paths[index], cv::IMREAD_UNCHANGED);

    cv::Mat flow(encoded.rows, encoded.cols, CV_32FC3);
    for (int y = 0; y < encoded.rows; ++y) {
        const cv::Vec3w* src = encoded.ptr<cv::Vec3w>(y);
        cv::Vec3f* dst = flow.ptr<cv::Vec3f>(y);
        for (int x = 0; x < encoded.cols; ++x) {
            float valid = src[x][0] != 0 ? 1.f : 0.f;
            dst[x][0] = (static_cast<float>(src[x][2]) - 32768.f) / 64.f; // u
            dst[x][1] = (static_cast<float>(src[x][1]) - 32768.f) / 64.f; // v
            dst[x][2] = valid;
        }
    }
    return flow;
}