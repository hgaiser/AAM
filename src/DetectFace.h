/*
 * DetectFace.h
 *
 *  Created on: 16 mei 2013
 *      Author: hansgaiser
 */

#ifndef DETECTFACE_H_
#define DETECTFACE_H_

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/ml/ml.hpp>

#include "MatlabIO.hpp"

#ifdef HAVE_CUDA
#include <opencv2/gpu/gpu.hpp>
#endif

#include <tinyxml.h>

#define TYPE float
#define MAT_TYPE(n) (typeid(TYPE) == typeid(float) ? CV_32FC(n) : CV_64FC(n))

namespace eva
{

class DetectFace
{
public:
	DetectFace(const char * workingDir);

	struct Config
	{
		std::string faceCascadeName;
		std::string noseCascadeName;
		double scale;
		double iterationThreshold;
		uint32_t translationIteration;
		uint32_t maxIteration;

		std::string imageTopic;
		bool showImages;
		bool cudaEnabled;
	};

	struct Model
	{
		std::string modelName;

		uint32_t width;
		uint32_t height;

		cv::Mat curr_points;

		cv::Mat size;

		cv::Mat shape_mean;
		cv::Mat shape_vectors;
		cv::Mat shape_transform;
		cv::Mat shape_mesh;

		cv::Mat app_mean;
		cv::Mat app_vectors;
		cv::Mat gradient;

		cv::Mat steepest_descent;
		cv::Mat H;
		cv::Mat invH;
		std::vector<cv::Mat> R;

		cv::Mat warp_map;

		uint32_t no_pixels;
		uint32_t trans_it;

	};

	void spin();

private:
	bool m_modelLoaded;
	std::string m_configPath;
	std::string m_workingDir;

	cv::CascadeClassifier m_faceCascade;
	cv::CascadeClassifier m_noseCascade;
#ifdef HAVE_CUDA
	cv::gpu::CascadeClassifier_GPU m_faceCascadeGpu;
	cv::gpu::CascadeClassifier_GPU m_noseCascadeGpu;
#endif
	Config m_config;
	Model m_model;

	bool loadConfig();
	void processImage(cv::Mat image);

	void drawPoints(cv::Mat & image, cv::Mat points, cv::Scalar color);
	void drawMesh(cv::Mat & image, cv::Mat points, cv::Scalar color, cv::Mat mesh);
	void drawText(cv::Mat & image, std::string text, cv::Scalar fontColor);

	void matchModel(cv::Mat image);

	void loadModel();
};

} /* namespace eva */
#endif /* DETECTFACE_H_ */
