#pragma once

#include <opencv2/opencv.hpp>

#include <string>

namespace graph_binarization
{

struct Metrics
{
    double iou = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

struct ProcessingParams
{
    int crop_x1 = 20;
    int crop_y1 = 0;
    int crop_x2 = 1450;
    int crop_y2 = 2000;

    int output_width = 1480;
    int output_height = 2100;

    int center_keep_radius = 5;
    int line_half_width = 1;

    int dilation_size = 2;
    int min_component_area = 70;

    int max_line_thickness = 8;
    int min_line_length = 18;
};

cv::Mat preprocessImage(const cv::Mat& image, const ProcessingParams& params);

cv::Mat preprocessMask(const cv::Mat& mask, const ProcessingParams& params);

cv::Mat toGray(const cv::Mat& image);

cv::Mat applyFourierFilter(const cv::Mat& gray, const ProcessingParams& params);

cv::Mat binarizeTextMask(const cv::Mat& filteredImage, const ProcessingParams& params);

cv::Mat createErrorMap(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask);

Metrics calculateMetrics(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask);

bool saveImage(const std::string& path, const cv::Mat& image);

} 
