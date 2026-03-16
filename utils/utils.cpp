#include "utils.h"

FlowSet::FlowSet(const std::string& root, bool occ) {
    std::string image_dir = root + "/training/image_2";
    std::string flow_dir  = root + "/training/" + (occ ? "flow_occ" : "flow_noc");

    // Collect paths first
    std::vector<std::tuple<std::string, std::string>> paths;
    for (auto& entry : fs::directory_iterator(flow_dir)) {
        std::string flow_path = entry.path().string();
        if (flow_path.find("_10.png") == std::string::npos) continue;

        std::string filename = entry.path().filename().string();
        std::string img1_path = image_dir + "/" + filename;
        std::string img2_path = image_dir + "/" +
                                filename.substr(0, filename.size() - 7) + "_11.png";

        if (fs::exists(img1_path) && fs::exists(img2_path)) {
            paths.emplace_back(img1_path, img2_path);
        }
    }

    // Preload everything
    img1s.reserve(paths.size());
    img2s.reserve(paths.size());
    //gray1s.reserve(paths.size());
    //gray2s.reserve(paths.size());

    for (const auto& [p1, p2] : paths) {
        cv::Mat img1 = cv::imread(p1);
        cv::Mat img2 = cv::imread(p2);

        //cv::Mat g1, g2;
        //cv::cvtColor(img1, g1, cv::COLOR_BGR2GRAY);
        //cv::cvtColor(img2, g2, cv::COLOR_BGR2GRAY);

        img1s.push_back(std::move(img1));
        img2s.push_back(std::move(img2));
        //gray1s.push_back(std::move(g1));
        //gray2s.push_back(std::move(g2));
    }
}