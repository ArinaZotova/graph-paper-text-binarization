#include "graph_binarization/binarization.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage:\n";
        std::cout << "demo <input_image> <output_dir> [gt_mask] [--visualize]\n";
        return 1;
    }

    std::string inputImagePath = argv[1];
    std::string outputDir = argv[2];

    std::string groundTruthPath;
    bool hasGroundTruth = false;
    bool saveVisualizations = false;

    for (int i = 3; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--visualize")
        {
            saveVisualizations = true;
        }
        else
        {
            groundTruthPath = arg;
            hasGroundTruth = true;
        }
    }

    fs::create_directories(outputDir);

    cv::Mat image = cv::imread(inputImagePath, cv::IMREAD_COLOR);

    if (image.empty())
    {
        std::cerr << "Error: cannot read input image: " << inputImagePath << std::endl;
        return 1;
    }

    graph_binarization::ProcessingParams params;

    cv::Mat preprocessedImage = graph_binarization::preprocessImage(image, params);

    if (preprocessedImage.empty())
    {
        std::cerr << "Error: preprocessing failed" << std::endl;
        return 1;
    }

    cv::Mat grayImage = graph_binarization::toGray(preprocessedImage);
    cv::Mat filteredImage = graph_binarization::applyFourierFilter(grayImage, params);
    cv::Mat predictedMask = graph_binarization::binarizeTextMask(filteredImage, params);

    if (predictedMask.empty())
    {
        std::cerr << "Error: binarization failed" << std::endl;
        return 1;
    }

    graph_binarization::saveImage(
        outputDir + "/predicted_mask.png",
        predictedMask
    );

    if (saveVisualizations)
    {
        graph_binarization::saveImage(
            outputDir + "/rectified_image.png",
            preprocessedImage
        );

        graph_binarization::saveImage(
            outputDir + "/gray_image.png",
            grayImage
        );

        graph_binarization::saveImage(
            outputDir + "/back.png",
            filteredImage
        );
    }

    if (hasGroundTruth)
    {
        cv::Mat groundTruth = cv::imread(groundTruthPath, cv::IMREAD_GRAYSCALE);

        if (groundTruth.empty())
        {
            std::cerr << "Error: cannot read ground truth mask: "
                      << groundTruthPath << std::endl;
            return 1;
        }

        cv::Mat preprocessedGroundTruth =
            graph_binarization::preprocessMask(groundTruth, params);

        if (preprocessedGroundTruth.empty())
        {
            std::cerr << "Error: ground truth preprocessing failed" << std::endl;
            return 1;
        }

        graph_binarization::Metrics metrics =
            graph_binarization::calculateMetrics(
                predictedMask,
                preprocessedGroundTruth
            );

        std::cout << "Metrics:" << std::endl;
        std::cout << "IoU: " << metrics.iou << std::endl;
        std::cout << "Precision: " << metrics.precision << std::endl;
        std::cout << "Recall: " << metrics.recall << std::endl;
        std::cout << "F1-score: " << metrics.f1 << std::endl;

        cv::Mat errorMap = graph_binarization::createErrorMap(
            predictedMask,
            preprocessedGroundTruth
        );

        graph_binarization::saveImage(
            outputDir + "/error.png",
            errorMap
        );

        if (saveVisualizations)
        {
            graph_binarization::saveImage(
                outputDir + "/gt_mask.png",
                preprocessedGroundTruth
            );
        }
    }

    std::cout << "Done. Results saved to: " << outputDir << std::endl;

    return 0;
}
