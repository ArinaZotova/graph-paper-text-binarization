#include "graph_binarization/binarization.hpp"

#include <filesystem>
#include <iostream>

namespace graph_binarization
{

cv::Mat preprocessImage(const cv::Mat& image, const ProcessingParams& params)
{
    if (image.empty())
    {
        return {};
    }

    int x1 = std::max(0, params.crop_x1);
    int y1 = std::max(0, params.crop_y1);
    int x2 = std::min(image.cols, params.crop_x2);
    int y2 = std::min(image.rows, params.crop_y2);

    if (x2 <= x1 || y2 <= y1)
    {
        return {};
    }

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat cropped = image(roi).clone();

    cv::Mat resized;
    cv::resize(
        cropped,
        resized,
        cv::Size(params.output_width, params.output_height),
        0,
        0,
        cv::INTER_LINEAR
    );

    return resized;
}

cv::Mat preprocessMask(const cv::Mat& mask, const ProcessingParams& params)
{
    if (mask.empty())
    {
        return {};
    }

    cv::Mat grayMask;

    if (mask.channels() == 3)
    {
        cv::cvtColor(mask, grayMask, cv::COLOR_BGR2GRAY);
    }
    else
    {
        grayMask = mask.clone();
    }

    int x1 = std::max(0, params.crop_x1);
    int y1 = std::max(0, params.crop_y1);
    int x2 = std::min(grayMask.cols, params.crop_x2);
    int y2 = std::min(grayMask.rows, params.crop_y2);

    if (x2 <= x1 || y2 <= y1)
    {
        return {};
    }

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat cropped = grayMask(roi).clone();

    cv::Mat resized;
    cv::resize(
        cropped,
        resized,
        cv::Size(params.output_width, params.output_height),
        0,
        0,
        cv::INTER_NEAREST
    );

    cv::Mat binaryMask;
    cv::threshold(resized, binaryMask, 127, 255, cv::THRESH_BINARY);

    return binaryMask;
}

cv::Mat toGray(const cv::Mat& image)
{
    if (image.empty())
    {
        return {};
    }

    if (image.channels() == 1)
    {
        return image.clone();
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    return gray;
}

cv::Mat applyFourierFilter(const cv::Mat& gray, const ProcessingParams& params)
{
    if (gray.empty())
    {
        return {};
    }

    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);

    cv::Mat planes[] = {
        grayFloat,
        cv::Mat::zeros(grayFloat.size(), CV_32F)
    };

    cv::Mat complexImage;
    cv::merge(planes, 2, complexImage);

    cv::dft(complexImage, complexImage);

    int cx = complexImage.cols / 2;
    int cy = complexImage.rows / 2;

    cv::Mat q0(complexImage, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(complexImage, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(complexImage, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(complexImage, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::Mat mask = cv::Mat::ones(complexImage.size(), CV_32F);

    int centerX = mask.cols / 2;
    int centerY = mask.rows / 2;

    int halfWidth = params.line_half_width;

    cv::rectangle(
        mask,
        cv::Point(0, centerY - halfWidth),
        cv::Point(mask.cols - 1, centerY + halfWidth),
        cv::Scalar(0),
        cv::FILLED
    );

    cv::rectangle(
        mask,
        cv::Point(centerX - halfWidth, 0),
        cv::Point(centerX + halfWidth, mask.rows - 1),
        cv::Scalar(0),
        cv::FILLED
    );

    cv::circle(
        mask,
        cv::Point(centerX, centerY),
        params.center_keep_radius,
        cv::Scalar(1),
        cv::FILLED
    );

    cv::Mat maskPlanes[] = {mask, mask};
    cv::Mat complexMask;
    cv::merge(maskPlanes, 2, complexMask);

    cv::multiply(complexImage, complexMask, complexImage);

    q0 = cv::Mat(complexImage, cv::Rect(0, 0, cx, cy));
    q1 = cv::Mat(complexImage, cv::Rect(cx, 0, cx, cy));
    q2 = cv::Mat(complexImage, cv::Rect(0, cy, cx, cy));
    q3 = cv::Mat(complexImage, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::Mat inverseTransform;
    cv::idft(complexImage, inverseTransform, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Mat normalized;
    cv::normalize(inverseTransform, normalized, 0, 255, cv::NORM_MINMAX);

    cv::Mat result;
    normalized.convertTo(result, CV_8U);

    return result;
}

cv::Mat binarizeTextMask(const cv::Mat& filteredImage, const ProcessingParams& params)
{
    if (filteredImage.empty())
    {
        return {};
    }

    cv::Mat binary;
    cv::threshold(
        filteredImage,
        binary,
        0,
        255,
        cv::THRESH_BINARY_INV | cv::THRESH_OTSU
    );

    cv::Mat kernel = cv::Mat::ones(
        params.dilation_size,
        params.dilation_size,
        CV_8U
    );

    cv::dilate(binary, binary, kernel, cv::Point(-1, -1), 1);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    int numberOfLabels = cv::connectedComponentsWithStats(
        binary,
        labels,
        stats,
        centroids,
        8,
        CV_32S
    );

    cv::Mat cleaned = cv::Mat::zeros(binary.size(), CV_8U);

    for (int label = 1; label < numberOfLabels; ++label)
    {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        if (area < params.min_component_area)
        {
            continue;
        }

        if (width > params.min_line_length && height <= params.max_line_thickness)
        {
            continue;
        }

        if (height > params.min_line_length && width <= params.max_line_thickness)
        {
            continue;
        }

        cleaned.setTo(255, labels == label);
    }

    return cleaned;
}

cv::Mat createErrorMap(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask)
{
    if (predictedMask.empty() || groundTruthMask.empty())
    {
        return {};
    }

    cv::Mat predBinary;
    cv::Mat gtBinary;

    cv::threshold(predictedMask, predBinary, 127, 1, cv::THRESH_BINARY);
    cv::threshold(groundTruthMask, gtBinary, 127, 1, cv::THRESH_BINARY);

    cv::Mat errorMap = cv::Mat::zeros(predBinary.size(), CV_8UC3);

    for (int y = 0; y < predBinary.rows; ++y)
    {
        for (int x = 0; x < predBinary.cols; ++x)
        {
            int pred = predBinary.at<uchar>(y, x);
            int gt = gtBinary.at<uchar>(y, x);

            if (pred == 0 && gt == 0)
            {
                errorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
            else if (pred == 1 && gt == 1)
            {
                errorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            }
            else if (pred == 1 && gt == 0)
            {
                errorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
            }
            else if (pred == 0 && gt == 1)
            {
                errorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
            }
        }
    }

    return errorMap;
}

Metrics calculateMetrics(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask)
{
    Metrics metrics;

    if (predictedMask.empty() || groundTruthMask.empty())
    {
        return metrics;
    }

    cv::Mat predBinary;
    cv::Mat gtBinary;

    cv::threshold(predictedMask, predBinary, 127, 1, cv::THRESH_BINARY);
    cv::threshold(groundTruthMask, gtBinary, 127, 1, cv::THRESH_BINARY);

    long long tp = 0;
    long long fp = 0;
    long long fn = 0;

    for (int y = 0; y < predBinary.rows; ++y)
    {
        for (int x = 0; x < predBinary.cols; ++x)
        {
            int pred = predBinary.at<uchar>(y, x);
            int gt = gtBinary.at<uchar>(y, x);

            if (pred == 1 && gt == 1)
            {
                ++tp;
            }
            else if (pred == 1 && gt == 0)
            {
                ++fp;
            }
            else if (pred == 0 && gt == 1)
            {
                ++fn;
            }
        }
    }

    metrics.iou = static_cast<double>(tp) / static_cast<double>(tp + fp + fn + 1e-9);
    metrics.precision = static_cast<double>(tp) / static_cast<double>(tp + fp + 1e-9);
    metrics.recall = static_cast<double>(tp) / static_cast<double>(tp + fn + 1e-9);
    metrics.f1 = 2.0 * metrics.precision * metrics.recall /
                 (metrics.precision + metrics.recall + 1e-9);

    return metrics;
}

bool saveImage(const std::string& path, const cv::Mat& image)
{
    if (image.empty())
    {
        return false;
    }

    std::filesystem::path filePath(path);
    std::filesystem::create_directories(filePath.parent_path());

    return cv::imwrite(path, image);
}

}
